import json
import logging
import os
import pickle
import time
from functools import partial

import jax
import jax.numpy as jnp
import jaxpruner
import orbax.checkpoint as ocp
import yaml
from jax import random
from jax.scipy.linalg import block_diag

from sparseRNNs.dataloaders import Datasets
from sparseRNNs.model.seq_model import (QBatchClassificationModel,
                                        QBatchRegressionModel)
from sparseRNNs.model.ssm import init_qS5SSM
from sparseRNNs.model.ssm_init import make_DPLR_HiPPO
from sparseRNNs.train_helpers import (capture_intermediates,
                                      capture_intermediates_ndns,
                                      create_train_state, train_epoch,
                                      train_epoch_ndns, validate,
                                      validate_ndns)
from sparseRNNs.utils.logging import (compute_eigenvalue_logs, logger,
                                      setup_experiment_logging_fns)
from sparseRNNs.utils.pruning import pruning_recipe_map
from sparseRNNs.utils.quantization import (
    _merge_trained_params_into_calibrated, _move_scales_to_params,
    quantization_recipe_map)


def _store_pytree(pytree, filename, datafolder):
    with open(os.path.join(datafolder, filename), "wb") as f:
        pickle.dump(pytree, f)


def convert(args):

    if args.debug:
        jax.config.update("jax_disable_jit", True)
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)

    logger.debug(args)

    LOAD_FROM_CHECKPOINT = (
        args.checkpoint_dir is not None and args.load_run_name is not None
    )
    VALIDATE_BASELINE = args.validate_baseline
    VALIDATE_NAIVESCAN = args.validate_naive_scan
    STORE_ACTIVATIONS = args.store_activations
    TRAIN_AQT = args.train_aqt
    VALIDATE_AQT = args.validate_aqt
    VALIDATE_STATIC_QUANT = args.validate_static_quant
    TRAIN_STATIC_QUANT = args.train_static_quant
    RESTORE_CHECKPOINT_STEP = args.checkpoint_restore_step

    args_summary = yaml.dump(vars(args), default_flow_style=False, sort_keys=False)
    logger.info("Starting conversion with args:", args_summary)

    # set up datafolder for artefacts (subfolder in the checkpoint folder)
    chkpt_path = os.path.join(args.checkpoint_dir, args.load_run_name)
    assert os.path.exists(chkpt_path), "checkpoint directory doesn't exist!"
    DATA_FOLDER = os.path.join(args.checkpoint_dir, args.run_name)
    store_pytree = partial(_store_pytree, datafolder=DATA_FOLDER)
    if os.path.exists(DATA_FOLDER):
        logger.warning(f"Data folder {DATA_FOLDER} already exists. Overwriting...")
    os.makedirs(DATA_FOLDER, exist_ok=True)

    # overwrite synaptic sparsity logging arg
    args.log_syn_sparsity = args.pruning != "no_prune"
    if not args.log_syn_sparsity:
        logger.debug("No pruning config detected. Not logging sparsity.")

    # store args into data folder
    with open(os.path.join(DATA_FOLDER, "args.json"), "w") as json_file:
        json.dump(vars(args), json_file, indent=4)
    with open(os.path.join(DATA_FOLDER, "fp_model.txt"), "w") as f:
        f.write(f"Original FP checkpoint from: {str(chkpt_path)}")

    ssm_size = args.ssm_size_base
    ssm_lr = args.ssm_lr_base

    # determine the size of initial blocks
    block_size = int(ssm_size / args.blocks)

    # Set global learning rate lr (e.g. encoders, etc.) as function of ssm_lr
    lr = args.lr_factor * ssm_lr

    # Set randomness...
    logger.info("Setting Randomness...")
    key = random.PRNGKey(args.jax_seed)
    init_rng, train_rng = random.split(key, num=2)

    # Get dataset creation function
    logger.info(f"Creating Dataset for `{args.dataset}`...")
    create_dataset_fn = Datasets[args.dataset]

    # Dataset dependent logic
    padded = args.dataset in [
        "imdb-classification",
        "listops-classification",
        "aan-classification",
    ]
    retrieval = args.dataset in ["aan-classification"]
    ndns = args.dataset in ["ndns"]

    if ndns:
        train_fn = train_epoch_ndns
        val_fn = validate_ndns
        capture_fn = capture_intermediates_ndns
    else:
        train_fn = train_epoch
        val_fn = validate
        capture_fn = capture_intermediates

    # Create dataset...
    init_rng, key = random.split(init_rng, num=2)
    (
        trainloader,
        valloader,
        testloader,
        aux_dataloaders,
        n_classes,
        seq_len,
        in_dim,
        train_size,
    ) = create_dataset_fn(args.dir_name, seed=args.jax_seed, bsz=args.bsz)
    steps_per_epoch = int(train_size / args.bsz)
    n_steps_total = steps_per_epoch * args.epochs
    n_steps_warmup = steps_per_epoch * args.warmup_end

    valloader = valloader if valloader is not None else testloader

    logger.info(f"Initializing...")

    # Initialize state matrix A using approximation to HiPPO-LegS matrix
    Lambda, _, B, V, B_orig = make_DPLR_HiPPO(block_size)

    if args.conj_sym:
        block_size = block_size // 2
        ssm_size = ssm_size // 2

    Lambda = Lambda[:block_size]
    V = V[:, :block_size]
    Vc = V.conj().T

    # If initializing state matrix A as block-diagonal, put HiPPO approximation
    # on each block
    Lambda = (Lambda * jnp.ones((args.blocks, block_size))).ravel()
    V = block_diag(*([V] * args.blocks))
    Vinv = block_diag(*([Vc] * args.blocks))

    q_config = quantization_recipe_map[args.quantization]()
    print("original qconfig:", q_config)

    pruning_cfg = pruning_recipe_map[args.pruning](
        epochs=args.epochs, steps_per_epoch=steps_per_epoch
    )

    ssm_init_fn_partial = partial(
        init_qS5SSM,
        H=args.d_model,
        P=ssm_size,
        Lambda_re_init=Lambda.real,
        Lambda_im_init=Lambda.imag,
        V=V,
        Vinv=Vinv,
        C_init=args.C_init,
        discretization=args.discretization,
        dt_min=args.dt_min,
        dt_max=args.dt_max,
        conj_sym=args.conj_sym,
        clip_eigs=args.clip_eigs,
        bidirectional=args.bidirectional,
        relufication=args.relufication,
    )

    if retrieval:
        raise NotImplementedError("Retrieval model not implemented")
    elif ndns:
        task = QBatchRegressionModel
    else:
        task = partial(QBatchClassificationModel, mode=args.mode)

    model_cls_partial = partial(
        task,
        n_layers=args.n_layers,
        d_model=args.d_model,
        dropout=args.p_dropout,
        batchnorm=args.batchnorm,
        prenorm=args.prenorm,
        bn_momentum=args.bn_momentum,
        glu_variant=args.glu_variant,
        relufication=args.relufication,
        fuse_batchnorm_linear=args.fuse_batchnorm_linear,
        d_output=n_classes,
        padded=padded,
        use_batchnorm_bias=args.batchnorm_use_bias,
        use_batchnorm_scale=args.batchnorm_use_scale,
        quant_input=args.quant_input,
    )

    ssm_init_fn = ssm_init_fn_partial(
        q_config=q_config,
        associative_scan=False,
    )
    model_cls = partial(
        model_cls_partial,
        mixer_cls=ssm_init_fn,
        q_config=q_config,
        training=False,
    )

    # initialize training state
    logger.info("Initializing Training State...")
    state, sparsity_updater = create_train_state(
        model_cls=model_cls,
        rng=init_rng,
        padded=padded,
        retrieval=retrieval,
        n_steps_total=n_steps_total,
        in_dim=in_dim,
        bsz=args.bsz,
        seq_len=seq_len,
        weight_decay=args.weight_decay,
        opt_config=args.opt_config,
        ssm_lr=ssm_lr,
        lr=lr,
        grad_clip_threshold=args.grad_clip_threshold,
        dt_global=args.dt_global,
        n_warmup=n_steps_warmup,
        lr_min=args.lr_min,
        batchnorm=args.batchnorm,
        pruner_cfg=pruning_cfg,
        training=False,
    )

    logger.info("Training State Initialized...")

    chkpt_metadata = {
        "best_test_loss": 100000000,
        "best_test_acc": -10000.0,
        "wandb_id": None,
        "last_step": 0,
        "next_epoch": 0,
    }

    restored_state = None
    restored_metadata = None

    results_dict = {}

    if VALIDATE_AQT or VALIDATE_BASELINE or VALIDATE_NAIVESCAN or VALIDATE_STATIC_QUANT:
        # Setup logging for experiment tracking, with wandb or mlflow or disable tracking
        log_metrics, log_best_metrics, log_artifacts, end_logging = (
            setup_experiment_logging_fns(args, chkpt_metadata)
        )

    #########################################################################
    # loading the model from the checkpoint
    #########################################################################

    if LOAD_FROM_CHECKPOINT:
        logger.info(f"Loading checkpoint from {chkpt_path}...")
        chkpt_options = ocp.CheckpointManagerOptions(
            save_interval_steps=args.checkpoint_interval_steps,
        )
        chkpt_mngr = ocp.CheckpointManager(
            directory=chkpt_path,
            item_names=("state", "metadata"),
            options=chkpt_options,
        )
        abstract_state = jax.tree_util.tree_map(ocp.utils.to_shape_dtype_struct, state)
        restore_step = (
            chkpt_mngr.best_step()
            if RESTORE_CHECKPOINT_STEP is None
            else RESTORE_CHECKPOINT_STEP
        )
        restored = chkpt_mngr.restore(
            restore_step,
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(abstract_state),
                metadata=ocp.args.JsonRestore(),
            ),
        )
        restored_state = restored["state"]
        restored_metadata = restored["metadata"]

        assert restored_state is not None
        assert restored_metadata is not None
        state = restored_state

        best_test_loss = restored_metadata["best_test_loss"]
        best_test_acc = restored_metadata["best_test_acc"]
        step = restored_metadata["last_step"]
        epoch_start = restored_metadata["next_epoch"]
        logger.info(f"Loaded step {step} and epoch {epoch_start}")
        logger.info(f"Best test loss: {best_test_loss}")
        logger.info(f"Best test accuracy: {best_test_acc}")

        sparsity_summary = jaxpruner.summarize_sparsity(
            state.params, only_total_sparsity=True
        )
        logger.info(f"Total Sparsity: {sparsity_summary['_total_sparsity']:.4f}")
        logger.info(
            "Number of Nonzero Parameters:" f" {sparsity_summary['_nparams_nnz']:.0f}"
        )
        logger.info(f"Number of Parameters: {sparsity_summary['_nparams']:.0f}")

        if sparsity_summary["_total_sparsity"] < 0.01:
            logger.warning("Detected low sparsity -> re-apply sparsity mask")
            state = state.replace(
                params=sparsity_updater.pre_forward_update(
                    state.params, state.opt_state
                )
            )
            sparsity_summary = jaxpruner.summarize_sparsity(
                state.params, only_total_sparsity=True
            )
            logger.info(f"Total Sparsity: {sparsity_summary['_total_sparsity']:.4f}")
            logger.info(
                "Number of Nonzero Parameters:"
                f" {sparsity_summary['_nparams_nnz']:.0f}"
            )
            logger.info(f"Number of Parameters: {sparsity_summary['_nparams']:.0f}")

    #########################################################################
    # validate loaded baseline model (no modifications) + store activations
    #########################################################################

    if VALIDATE_BASELINE:
        logger.info("Running Validation of baseline model...")
        val_tuple = val_fn(
            state,
            key,
            model_cls,
            valloader,
            seq_len,
            in_dim,
            args.batchnorm,
            log_act_sparsity=False,
        )
        val_loss, val_acc = val_tuple
        logger.info(f"Validation Loss: {val_loss:.4f}")
        logger.info(f"Validation Accuracy: {val_acc:.4f}")
        log_best_metrics(
            {
                "Best Val Loss - baseline": val_loss,
                "Best Val Acc - baseline": val_acc,
            },
            step=step,
        )

    if STORE_ACTIVATIONS:
        logger.info("Capture intermediate activations...")
        intermediates, inputs, labels = capture_fn(
            state,
            key,
            model_cls,
            valloader,
            seq_len,
            in_dim,
            args.batchnorm,
        )
        with open(os.path.join(DATA_FOLDER, "activations_fp.pkl"), "wb") as f:
            pickle.dump(intermediates, f)
        jnp.save(os.path.join(DATA_FOLDER, "inputs.npy"), inputs)
        jnp.save(os.path.join(DATA_FOLDER, "labels.npy"), labels)
        del intermediates, inputs, labels
        logger.info("Successfully stored intermediate activations")

    #########################################################################
    # validate the loaded model with naive scans (non-parallel)
    #########################################################################

    if VALIDATE_NAIVESCAN:
        ssm_init_fn = ssm_init_fn_partial(
            q_config=q_config,
            associative_scan=False,
        )
        model_cls = partial(
            model_cls_partial,
            mixer_cls=ssm_init_fn,
            q_config=q_config,
        )
        logger.info("Running validation using naive scans...")
        val_tuple = val_fn(
            state,
            train_rng,
            model_cls,
            valloader,
            seq_len,
            in_dim,
            args.batchnorm,
            log_act_sparsity=True,
        )
        val_loss, val_acc, act_sparsity_logs = val_tuple
        logger.info(f"Validation Loss: {val_loss:.4f}")
        logger.info(f"Validation Accuracy: {val_acc:.4f}")
        log_best_metrics(
            {
                "Best Val Loss - naivescan": val_loss,
                "Best Val Acc - naivescan": val_acc,
            },
            step=step,
        )

    #########################################################################
    # update configuration for quantization (non-static, W8A16, parallel fwd)
    #########################################################################

    if VALIDATE_AQT:
        q_config = quantization_recipe_map[args.convert_quantization]()
        ssm_init_fn = ssm_init_fn_partial(
            q_config=q_config,
            associative_scan=True,
        )
        model_cls = partial(
            model_cls_partial,
            mixer_cls=ssm_init_fn,
            q_config=q_config,
            training=False,
        )
        logger.info("Running validation using AQT...")
        logger.info(f"{q_config}")
        val_tuple = val_fn(
            state,
            train_rng,
            model_cls,
            valloader,
            seq_len,
            in_dim,
            args.batchnorm,
            log_act_sparsity=True,
        )
        val_loss, val_acc, act_sparsity_logs = val_tuple
        logger.info(f"Validation Loss: {val_loss:.4f}")
        logger.info(f"Validation Accuracy: {val_acc:.4f}")
        log_best_metrics(
            {
                "Best Val Loss - AQT": val_loss,
                "Best Val Acc - AQT": val_acc,
            },
            step=step,
        )

    if TRAIN_AQT:
        lr_factor_qaft = args.aqt_qaft_lr_factor
        q_config = quantization_recipe_map[args.convert_quantization]()
        ssm_init_fn = ssm_init_fn_partial(
            q_config=q_config,
            associative_scan=True,
        )
        model_cls = partial(
            model_cls_partial,
            mixer_cls=ssm_init_fn,
            q_config=q_config,
            training=True,
        )
        logger.info("Training the model with AQT...")
        logger.info(f"{q_config}")

        # NOTE: set additional params
        args.epochs = args.epochs + 10
        steps_per_epoch = int(train_size / args.bsz)
        n_steps_total = steps_per_epoch * args.epochs
        n_steps_warmup = steps_per_epoch * args.warmup_end
        logger.debug(f"Training for total {args.epochs} epochs ({n_steps_total} steps)")
        logger.debug(f"continuing at epoch {epoch_start}")

        fresh_training_state, sparsity_updater = create_train_state(
            model_cls,
            init_rng,
            padded,
            retrieval,
            n_steps_total,
            in_dim=in_dim,
            bsz=args.bsz,
            seq_len=seq_len,
            weight_decay=args.weight_decay,
            batchnorm=args.batchnorm,
            opt_config=args.opt_config,
            ssm_lr=ssm_lr * lr_factor_qaft,
            lr=lr * lr_factor_qaft,
            lr_min=args.lr_min,
            grad_clip_threshold=args.grad_clip_threshold,
            dt_global=args.dt_global,
            pruner_cfg=pruning_cfg,
            n_warmup=n_steps_warmup,
            training=True,
            static_quant=False,
            calibrating=False,
        )

        # NOTE: need to merge the optimizer states for fine-tuning
        # - params <- calibrated params
        # - step <- restored step
        # - batch_stats <- restored batch_stats
        # - opt_state:
        #   - count <- from restored_state
        #   - inner_state <- merge the inner_state.inner_states
        #   - masks <- from fresh_training_state (assertion below)
        #   - target_sparsities <- from fresh_training_state (assertion below)
        merged_masks = _merge_trained_params_into_calibrated(
            restored_state.opt_state.masks,
            fresh_training_state.opt_state.masks,
        )
        assert jax.tree_util.tree_all(
            jax.tree.map(
                jnp.allclose,
                fresh_training_state.opt_state.masks,
                merged_masks,
            )
        )
        mts = _merge_trained_params_into_calibrated(
            restored_state.opt_state.target_sparsities,
            fresh_training_state.opt_state.target_sparsities,
        )
        assert jax.tree_util.tree_all(
            jax.tree.map(
                jnp.allclose,
                fresh_training_state.opt_state.target_sparsities,
                mts,
            )
        )
        state = fresh_training_state.replace(
            opt_state=fresh_training_state.opt_state._replace(
                count=restored_state.opt_state.count,
                inner_state=fresh_training_state.opt_state.inner_state._replace(
                    inner_states=_merge_trained_params_into_calibrated(
                        restored_state.opt_state.inner_state.inner_states,
                        fresh_training_state.opt_state.inner_state.inner_states,
                    )
                ),
            ),
            params=restored_state.params,
            step=restored_state.step,
            batch_stats=restored_state.batch_stats,  # NOTE: only if batchnorm is used!
        )

        # log initial eigenvalues
        if epoch_start == 0:
            eigenvalue_logs = compute_eigenvalue_logs(state, clip_eigs=args.clip_eigs)
            log_metrics(eigenvalue_logs, step=0)

        # NOTE: setting logging options here
        args.log_act_sparsity = "both"
        args.log_grads = True

        # Training loop over epochs
        best_loss, best_acc, best_epoch = (
            100000000,
            -100000000.0,
            0,
        )  # This best loss is val_loss
        count, best_val_loss = (
            0,
            100000000,
        )  # This line is for early stopping purposes
        for epoch in range(epoch_start, args.epochs):
            logger.info(f"[*] Starting Training Epoch {epoch + 1}...")
            step = epoch * len(trainloader) + 1

            train_rng, skey = random.split(train_rng)
            train_return_tuple = train_fn(
                state,
                skey,
                model_cls,
                trainloader,
                seq_len,
                in_dim,
                args.batchnorm,
                sparsity_updater=sparsity_updater,
                return_grads=args.log_grads,
                log_metrics_fn=log_metrics,
                step=step,
                log_act_sparsity=args.log_act_sparsity in ("both", "train"),
            )
            if args.log_grads and args.log_act_sparsity in ("both", "train"):
                (
                    state,
                    train_loss,
                    sparsity_summary,
                    grad_logs,
                    act_sparsity_logs,
                ) = train_return_tuple
            elif args.log_grads:
                state, train_loss, sparsity_summary, grad_logs = train_return_tuple
            elif args.log_act_sparsity in ("both", "train"):
                state, train_loss, sparsity_summary, act_sparsity_logs = (
                    train_return_tuple
                )
            else:
                state, train_loss, sparsity_summary = train_return_tuple

            # log current sparsity levels, current learning rates, and eigenvalue statistics
            # assert state.opt_state.inner_state.inner_states['none'].inner_state.hyperparams['learning_rate'].item() == 0.0
            eigenvalue_logs = compute_eigenvalue_logs(state, clip_eigs=args.clip_eigs)
            act_sparsity_logs = (
                {f"act_sparsity/train/{k}": v for k, v in act_sparsity_logs.items()}
                if args.log_act_sparsity in ("both", "train")
                else {}
            )
            sparsity_logs = {
                "sparsity/" + k: float(v) for k, v in sparsity_summary.items()
            }

            lr_logs = {}
            # ssm lr
            # opt_state > jaxpruner.SparseState
            # opt_state.inner_state > optax.MultiTransformState
            # opt_state.inner_state.inner_states > dict[str, state]
            # opt_state.inner_state.inner_states['ssm'] > optax.MaskedState
            # opt_state.inner_state.inner_states['ssm'].inner_state > optax.InjectStatefulHyperparamsState
            # opt_state.inner_state.inner_states['ssm'].inner_state.hyperparams > dict[str, np.ndarray]
            if hasattr(
                state.opt_state.inner_state.inner_states["ssm"].inner_state,
                "hyperparams",
            ):
                lr_logs["ssm_lr"] = (
                    state.opt_state.inner_state.inner_states["ssm"]
                    .inner_state.hyperparams["learning_rate"]
                    .item()
                )
            elif isinstance(
                state.opt_state.inner_state.inner_states["ssm"].inner_state,
                tuple,
            ):
                lr_logs["ssm_lr"] = (
                    state.opt_state.inner_state.inner_states["ssm"]
                    .inner_state[0]
                    .hyperparams["learning_rate"]
                    .item()
                )
            else:
                logger.warning("ssm lr not found")
            # regular lr
            if hasattr(
                state.opt_state.inner_state.inner_states["regular"].inner_state,
                "hyperparams",
            ):
                lr_logs["lr"] = (
                    state.opt_state.inner_state.inner_states["regular"]
                    .inner_state.hyperparams["learning_rate"]
                    .item()
                )
            elif isinstance(
                state.opt_state.inner_state.inner_states["regular"].inner_state,
                tuple,
            ):
                lr_logs["lr"] = (
                    state.opt_state.inner_state.inner_states["regular"]
                    .inner_state[0]
                    .hyperparams["learning_rate"]
                    .item()
                )
            else:
                logger.warning("[warning] lr not found")

            log_metrics(
                {
                    **lr_logs,
                    **act_sparsity_logs,
                    **(sparsity_logs if args.log_act_sparsity else {}),
                    **(eigenvalue_logs if args.log_eigenvalues else {}),
                    **(grad_logs if args.log_grads else {}),
                },
                step=step,
            )

            new_params = sparsity_updater.pre_forward_update(
                state.params, state.opt_state
            )
            forward_state = state.replace(params=new_params)

            assert valloader is not None, "Validation loader is None"
            print(f"[*] Running Epoch {epoch + 1} Validation...")
            val_tuple = val_fn(
                forward_state,
                skey,
                model_cls,
                valloader,
                seq_len,
                in_dim,
                args.batchnorm,
                log_act_sparsity=args.log_act_sparsity in ("both", "val"),
            )
            if args.log_act_sparsity in ("both", "val"):
                val_loss, val_acc, act_sparsity_logs = val_tuple
                log_metrics(
                    {f"act_sparsity/val/{k}": v for k, v in act_sparsity_logs.items()},
                    step=step,
                )
            else:
                val_loss, val_acc = val_tuple

            print(f"[*] Running Epoch {epoch + 1} Test...")
            test_tuple = val_fn(
                forward_state,
                skey,
                model_cls,
                testloader,
                seq_len,
                in_dim,
                args.batchnorm,
                log_act_sparsity=args.log_act_sparsity in ("both", "val"),
            )
            if args.log_act_sparsity in ("both", "val"):
                test_loss, test_acc, act_sparsity_logs = test_tuple
                log_metrics(
                    {f"act_sparsity/test/{k}": v for k, v in act_sparsity_logs.items()},
                    step=step,
                )
            else:
                test_loss, test_acc = test_tuple

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {train_loss:.5f} -- Val Loss: {val_loss:.5f}"
                f" --Test Loss: {test_loss:.5f} --"
                f" Val Accuracy: {val_acc:.4f}"
                f" Test Accuracy: {test_acc:.4f}"
            )

            # For early stopping purposes
            if val_loss < best_val_loss:
                count = 0
                best_val_loss = val_loss
            else:
                count += 1

            if val_acc > best_acc:
                # Increment counters etc.
                count = 0
                best_loss, best_acc, best_epoch = val_loss, val_acc, epoch
                if valloader is not None:
                    best_test_loss, best_test_acc = test_loss, test_acc
                else:
                    best_test_loss, best_test_acc = best_loss, best_acc

                # Do some validation on improvement.
                # Evaluate on resolution 2 val and test sets
                print(f"[*] Running Epoch {epoch + 1} Res 2 Validation...")
                val2_tuple = val_fn(
                    forward_state,
                    skey,
                    model_cls,
                    aux_dataloaders["valloader2"],
                    int(seq_len // 2),
                    in_dim,
                    args.batchnorm,
                    step_rescale=2.0,
                    log_act_sparsity=args.log_act_sparsity in ("both", "val"),
                )
                if args.log_act_sparsity in ("both", "val"):
                    val2_loss, val2_acc, act_sparsity_log = val2_tuple
                    log_metrics(
                        {
                            f"act_sparsity/val2/{k}": v
                            for k, v in act_sparsity_log.items()
                        },
                        step=step,
                    )
                else:
                    val2_loss, val2_acc = val2_tuple

                print(f"[*] Running Epoch {epoch + 1} Res 2 Test...")
                test2_tuple = val_fn(
                    forward_state,
                    skey,
                    model_cls,
                    aux_dataloaders["testloader2"],
                    int(seq_len // 2),
                    in_dim,
                    args.batchnorm,
                    step_rescale=2.0,
                    log_act_sparsity=args.log_act_sparsity in ("both", "val"),
                )
                if args.log_act_sparsity in ("both", "val"):
                    test2_loss, test2_acc, act_sparsity_log = test2_tuple
                    log_metrics(
                        {
                            f"act_sparsity/test2/{k}": v
                            for k, v in act_sparsity_log.items()
                        },
                        step=step,
                    )
                else:
                    test2_loss, test2_acc = test2_tuple
                print(f"\n=>> Epoch {epoch + 1} Res 2 Metrics ===")
                print(
                    f"\tVal2 Loss: {val2_loss:.5f} --Test2 Loss:"
                    f" {test2_loss:.5f} -- Val Accuracy: {val2_acc:.4f} Test"
                    f" Accuracy: {test2_acc:.4f}"
                )

            # Print best accuracy & loss so far...
            print(
                f"\tBest Val Loss: {best_loss:.5f} -- Best Val Accuracy:"
                f" {best_acc:.4f} at Epoch {best_epoch + 1}\n\tBest Test Loss:"
                f" {best_test_loss:.5f} -- Best Test Accuracy:"
                f" {best_test_acc:.4f} at Epoch {best_epoch + 1}\n"
            )

            if valloader is not None:
                log_metrics(
                    {
                        "Training Loss": train_loss,
                        "Val loss": val_loss,
                        "Val Accuracy": val_acc,
                        "Test Loss": test_loss,
                        "Test Accuracy": test_acc,
                        # "Val2 loss": val2_loss,
                        # "Val2 Accuracy": val2_acc,
                        # "Test2 Loss": test2_loss,
                        # "Test2 Accuracy": test2_acc,
                    },
                    step=step,
                )
            else:
                log_metrics(
                    {
                        "Training Loss": train_loss,
                        "Val loss": val_loss,
                        "Val Accuracy": val_acc,
                    },
                    step=step,
                )
            log_best_metrics(
                {
                    "Best Val Loss": best_loss,
                    "Best Val Accuracy": best_acc,
                    "Best Epoch": best_epoch,
                    "Best Test Loss": best_test_loss,
                    "Best Test Accuracy": best_test_acc,
                },
                step=step,
            )

            # save checkpoint
            if chkpt_mngr is not None:
                chkpt_metadata["best_test_loss"] = best_test_loss.item()
                chkpt_metadata["best_test_acc"] = best_test_acc.item()
                chkpt_metadata["last_step"] = step
                chkpt_metadata["next_epoch"] = epoch + 1
                chkpt_mngr.save(
                    step=epoch + 1,
                    args=ocp.args.Composite(
                        state=ocp.args.StandardSave(state),
                        metadata=ocp.args.JsonSave(chkpt_metadata),
                    ),
                )
                chkpt_mngr.wait_until_finished()
                log_artifacts(os.path.join(args.checkpoint_dir, args.run_name))

            if count > args.early_stop_patience:
                break

    #########################################################################
    # update configuration for static quantization + naive fwd
    #########################################################################

    q_config = quantization_recipe_map[args.convert_quantization](
        static_quant=True, calibrating=True
    )
    ssm_init_fn = ssm_init_fn_partial(
        q_config=q_config,
        associative_scan=False,
    )
    logger.info("Updating configuration for static quantization...")
    logger.info(f"{q_config}")

    #########################################################################
    # calibrate the model on validation data
    #########################################################################

    if VALIDATE_STATIC_QUANT or TRAIN_STATIC_QUANT:
        if os.path.exists(os.path.join(DATA_FOLDER, "sc_calibrated_params.pkl")):
            logger.info("Loading calibrated parameters from file...")
            with open(os.path.join(DATA_FOLDER, "sc_calibrated_params.pkl"), "rb") as f:
                calibrated_params = pickle.load(f)
        else:
            logger.info("Creating calibrated parameters...")
            model_cls = partial(
                model_cls_partial,
                mixer_cls=ssm_init_fn,
                q_config=q_config,
                training=False,
            )
            cal_state, sparsity_updater = create_train_state(
                model_cls,
                init_rng,
                padded,
                retrieval,
                n_steps_total,
                in_dim=in_dim,
                bsz=args.bsz,
                seq_len=seq_len,
                weight_decay=args.weight_decay,
                batchnorm=args.batchnorm,
                opt_config=args.opt_config,
                ssm_lr=ssm_lr,
                lr=lr,
                grad_clip_threshold=args.grad_clip_threshold,
                dt_global=args.dt_global,
                pruner_cfg=pruning_cfg,
                n_warmup=n_steps_warmup,
                lr_min=args.lr_min,
                training=False,
                static_quant=True,
                calibrating=True,
            )
            logger.info("Merging trained params into calibration state")
            ############## state <- state.params (structure), state.batch_stats (structure, removed again)
            # merge trained params into calibration state
            merged_params = _merge_trained_params_into_calibrated(
                state.params, cal_state.params
            )
            merge_batchstats = _merge_trained_params_into_calibrated(
                state.batch_stats, cal_state.batch_stats
            )
            # replace the state.params with the merged params
            cal_state = state.replace(
                params=merged_params, batch_stats=merge_batchstats
            )

            if not os.path.exists(os.path.join(DATA_FOLDER, "sc_cal_stats.pkl")):
                logger.info("Calibrating the model on validation data...")
                c_loss, c_acc, cal_state_new = val_fn(
                    cal_state,
                    init_rng,
                    model_cls,
                    valloader,  # calibration done on validation (quicker!)
                    seq_len,
                    in_dim,
                    args.batchnorm,
                    return_state=True,
                )
                logger.info(f"Validation Loss: {c_loss:.4f}")
                logger.info(f"Validation Accuracy: {c_acc:.4f}")
                params_close = jax.tree_util.tree_all(
                    jax.tree.map(jnp.allclose, cal_state.params, cal_state_new.params)
                )
                batch_stats_close = jax.tree_util.tree_all(
                    jax.tree.map(
                        jnp.allclose,
                        cal_state.batch_stats,
                        cal_state_new.batch_stats,
                    )
                )
                if not params_close:
                    logger.warning(
                        "Parameters have changed after calibration without" " training!"
                    )
                if batch_stats_close:
                    logger.warning(
                        "Batch statistics have not changed after calibration!"
                    )
                store_pytree(cal_state_new.batch_stats, "sc_cal_stats.pkl")
                cal_stats = cal_state_new.batch_stats
            else:
                with open(os.path.join(DATA_FOLDER, "sc_cal_stats.pkl"), "rb") as f:
                    cal_stats = pickle.load(f)

            logger.info("Moving scales from batch_stats into params")
            calibrated_params = _move_scales_to_params(cal_state.params, cal_stats)
            store_pytree(calibrated_params, "sc_calibrated_params.pkl")

    #########################################################################
    # validate the calibrated model (static quantization)
    #########################################################################

    if VALIDATE_STATIC_QUANT or TRAIN_STATIC_QUANT:
        logger.info("Creating frozen calibrated model...")
        # NOTE: ignoring retrieval for now
        assert not retrieval, "Retrieval not supported for calibration"
        # NOTE: fix this
        q_config = quantization_recipe_map[args.convert_quantization](
            static_quant=True, calibrating=False
        )
        logger.info(f"{q_config}")
        ssm_init_fn = ssm_init_fn_partial(
            q_config=q_config,
            associative_scan=False,
        )
        model_cls = partial(
            model_cls_partial,
            mixer_cls=ssm_init_fn,
            q_config=q_config,
            training=False,
        )
        frozen_state = state.replace(params=calibrated_params)

    if VALIDATE_STATIC_QUANT:
        logger.info("Running Validation using calibrated model...")
        val_tuple = val_fn(
            frozen_state,
            train_rng,
            model_cls,
            valloader,
            seq_len,
            in_dim,
            args.batchnorm,
            log_act_sparsity=True,
        )
        val_loss, val_acc, act_sparsity_logs = val_tuple
        logger.info(f"Validation Loss: {val_loss:.4f}")
        logger.info(f"Validation Accuracy: {val_acc:.4f}")
        # write validation metrics to file simple json
        results_dict["static_val_loss"] = float(val_loss.item())
        results_dict["static_val_acc"] = float(val_acc.item())
        try:
            with open(os.path.join(DATA_FOLDER, "val_metrics.json"), "w") as f:
                json.dump(results_dict, f)
        except Exception as e:
            print(e)
        # write batch stats and params
        with open(os.path.join(DATA_FOLDER, "frozen_stats.pkl"), "wb") as f:
            pickle.dump(frozen_state.batch_stats, f)
        with open(os.path.join(DATA_FOLDER, "frozen_params.pkl"), "wb") as f:
            pickle.dump(frozen_state.params, f)
        # store to mlflow/wandb
        log_best_metrics(
            {
                "Best Val Loss - static quant": val_loss,
                "Best Val Acc - static quant": val_acc,
            },
            step=step,
        )

    if VALIDATE_STATIC_QUANT and STORE_ACTIVATIONS:
        logger.info("Capture intermediate activations...")
        intermediates, inputs, labels = capture_fn(
            frozen_state,
            key,
            model_cls,
            valloader,
            seq_len,
            in_dim,
            args.batchnorm,
        )
        with open(os.path.join(DATA_FOLDER, "activations_int.pkl"), "wb") as f:
            pickle.dump(intermediates, f)
        if not os.path.exists(os.path.join(DATA_FOLDER, "inputs.npy")):
            jnp.save(os.path.join(DATA_FOLDER, "inputs.npy"), inputs)
        if not os.path.exists(os.path.join(DATA_FOLDER, "labels.npy")):
            jnp.save(os.path.join(DATA_FOLDER, "labels.npy"), labels)
        del intermediates, inputs, labels

    #########################################################################
    # train the calibrated model with static quantization
    #########################################################################

    if TRAIN_STATIC_QUANT:
        log_metrics, log_best_metrics, log_artifacts, end_logging = (
            setup_experiment_logging_fns(args, chkpt_metadata)
        )

        # NOTE: set additional params
        args.epochs = args.epochs + 10
        steps_per_epoch = int(train_size / args.bsz)
        n_steps_total = steps_per_epoch * args.epochs
        n_steps_warmup = steps_per_epoch * args.warmup_end
        print(f"Training for total {args.epochs} epochs ({n_steps_total} steps)")
        print(f"continuing at epoch {epoch_start}")

        # NOTE: ignoring retrieval for now
        assert not retrieval, "Retrieval not supported for calibration"
        model_cls = partial(
            model_cls_partial,
            mixer_cls=ssm_init_fn,
            q_config=q_config,
            training=True,
        )
        fresh_training_state, sparsity_updater = create_train_state(
            model_cls,
            init_rng,
            padded,
            retrieval,
            n_steps_total,
            in_dim=in_dim,
            bsz=args.bsz,
            seq_len=seq_len,
            weight_decay=args.weight_decay,
            batchnorm=args.batchnorm,
            opt_config=args.opt_config,
            ssm_lr=ssm_lr,
            lr=lr,
            lr_min=args.lr_min,
            # ssm_lr=ssm_lr * 0.01,
            # lr=lr * 0.01,
            # lr_min=args.lr_min * 0.01,
            grad_clip_threshold=args.grad_clip_threshold,
            dt_global=args.dt_global,
            pruner_cfg=pruning_cfg,
            n_warmup=n_steps_warmup,
            training=True,
            static_quant=True,
            calibrating=False,
        )

        # NOTE: need to merge the optimizer states for fine-tuning
        # - params <- calibrated params
        # - step <- restored step
        # - batch_stats <- restored batch_stats
        # - opt_state:
        #   - count <- from restored_state
        #   - inner_state <- merge the inner_state.inner_states
        #   - masks <- from fresh_training_state (assertion below)
        #   - target_sparsities <- from fresh_training_state (assertion below)
        merged_masks = _merge_trained_params_into_calibrated(
            restored_state.opt_state.masks,
            fresh_training_state.opt_state.masks,
        )
        assert jax.tree_util.tree_all(
            jax.tree.map(
                jnp.allclose,
                fresh_training_state.opt_state.masks,
                merged_masks,
            )
        )
        mts = _merge_trained_params_into_calibrated(
            restored_state.opt_state.target_sparsities,
            fresh_training_state.opt_state.target_sparsities,
        )
        assert jax.tree_util.tree_all(
            jax.tree.map(
                jnp.allclose,
                fresh_training_state.opt_state.target_sparsities,
                mts,
            )
        )
        training_state = fresh_training_state.replace(
            opt_state=fresh_training_state.opt_state._replace(
                count=restored_state.opt_state.count,
                inner_state=fresh_training_state.opt_state.inner_state._replace(
                    inner_states=_merge_trained_params_into_calibrated(
                        restored_state.opt_state.inner_state.inner_states,
                        fresh_training_state.opt_state.inner_state.inner_states,
                    )
                ),
            ),
            params=calibrated_params,
            step=restored_state.step,
            batch_stats=restored_state.batch_stats,  # NOTE: only if batchnorm is used!
        )

        # log initial eigenvalues
        if epoch_start == 0:
            eigenvalue_logs = compute_eigenvalue_logs(state, clip_eigs=args.clip_eigs)
            log_metrics(eigenvalue_logs, step=0)

        # NOTE: setting logging options here
        args.log_act_sparsity = "both"
        args.log_grads = True

        # Training loop over epochs
        best_loss, best_acc, best_epoch = (
            100000000,
            -100000000.0,
            0,
        )  # This best loss is val_loss
        count, best_val_loss = (
            0,
            100000000,
        )  # This line is for early stopping purposes
        for epoch in range(epoch_start, args.epochs):
            print(f"[*] Starting Training Epoch {epoch + 1}...")
            step = epoch * len(trainloader) + 1

            train_rng, skey = random.split(train_rng)
            train_return_tuple = train_fn(
                training_state,
                skey,
                model_cls,
                trainloader,
                seq_len,
                in_dim,
                args.batchnorm,
                sparsity_updater=sparsity_updater,
                return_grads=args.log_grads,
                log_metrics_fn=log_metrics,
                step=step,
                log_act_sparsity=args.log_act_sparsity in ("both", "train"),
            )
            if args.log_grads and args.log_act_sparsity in ("both", "train"):
                (
                    state,
                    train_loss,
                    sparsity_summary,
                    grad_logs,
                    act_sparsity_logs,
                ) = train_return_tuple
            elif args.log_grads:
                state, train_loss, sparsity_summary, grad_logs = train_return_tuple
            elif args.log_act_sparsity in ("both", "train"):
                state, train_loss, sparsity_summary, act_sparsity_logs = (
                    train_return_tuple
                )
            else:
                state, train_loss, sparsity_summary = train_return_tuple

            # log current sparsity levels, current learning rates, and eigenvalue statistics
            # assert state.opt_state.inner_state.inner_states['none'].inner_state.hyperparams['learning_rate'].item() == 0.0
            eigenvalue_logs = compute_eigenvalue_logs(state, clip_eigs=args.clip_eigs)
            act_sparsity_logs = (
                {f"act_sparsity/train/{k}": v for k, v in act_sparsity_logs.items()}
                if args.log_act_sparsity in ("both", "train")
                else {}
            )
            sparsity_logs = {
                "sparsity/" + k: float(v) for k, v in sparsity_summary.items()
            }

            lr_logs = {}
            # ssm lr
            # opt_state > jaxpruner.SparseState
            # opt_state.inner_state > optax.MultiTransformState
            # opt_state.inner_state.inner_states > dict[str, state]
            # opt_state.inner_state.inner_states['ssm'] > optax.MaskedState
            # opt_state.inner_state.inner_states['ssm'].inner_state > optax.InjectStatefulHyperparamsState
            # opt_state.inner_state.inner_states['ssm'].inner_state.hyperparams > dict[str, np.ndarray]
            if hasattr(
                state.opt_state.inner_state.inner_states["ssm"].inner_state,
                "hyperparams",
            ):
                lr_logs["ssm_lr"] = (
                    state.opt_state.inner_state.inner_states["ssm"]
                    .inner_state.hyperparams["learning_rate"]
                    .item()
                )
            elif isinstance(
                state.opt_state.inner_state.inner_states["ssm"].inner_state,
                tuple,
            ):
                lr_logs["ssm_lr"] = (
                    state.opt_state.inner_state.inner_states["ssm"]
                    .inner_state[0]
                    .hyperparams["learning_rate"]
                    .item()
                )
            else:
                print("[warning] ssm lr not found")
            # regular lr
            if hasattr(
                state.opt_state.inner_state.inner_states["regular"].inner_state,
                "hyperparams",
            ):
                lr_logs["lr"] = (
                    state.opt_state.inner_state.inner_states["regular"]
                    .inner_state.hyperparams["learning_rate"]
                    .item()
                )
            elif isinstance(
                state.opt_state.inner_state.inner_states["regular"].inner_state,
                tuple,
            ):
                lr_logs["lr"] = (
                    state.opt_state.inner_state.inner_states["regular"]
                    .inner_state[0]
                    .hyperparams["learning_rate"]
                    .item()
                )
            else:
                print("[warning] lr not found")

            log_metrics(
                {
                    **lr_logs,
                    **act_sparsity_logs,
                    **(sparsity_logs if args.log_act_sparsity else {}),
                    **(eigenvalue_logs if args.log_eigenvalues else {}),
                    **(grad_logs if args.log_grads else {}),
                },
                step=step,
            )

            new_params = sparsity_updater.pre_forward_update(
                training_state.params, training_state.opt_state
            )
            forward_state = training_state.replace(params=new_params)

            assert valloader is not None, "Validation loader is None"
            print(f"[*] Running Epoch {epoch + 1} Validation...")
            val_tuple = val_fn(
                forward_state,
                skey,
                model_cls,
                valloader,
                seq_len,
                in_dim,
                args.batchnorm,
                log_act_sparsity=args.log_act_sparsity in ("both", "val"),
            )
            if args.log_act_sparsity in ("both", "val"):
                val_loss, val_acc, act_sparsity_logs = val_tuple
                log_metrics(
                    {f"act_sparsity/val/{k}": v for k, v in act_sparsity_logs.items()},
                    step=step,
                )
            else:
                val_loss, val_acc = val_tuple

            print(f"[*] Running Epoch {epoch + 1} Test...")
            test_tuple = val_fn(
                forward_state,
                skey,
                model_cls,
                testloader,
                seq_len,
                in_dim,
                args.batchnorm,
                log_act_sparsity=args.log_act_sparsity in ("both", "val"),
            )
            if args.log_act_sparsity in ("both", "val"):
                test_loss, test_acc, act_sparsity_logs = test_tuple
                log_metrics(
                    {f"act_sparsity/test/{k}": v for k, v in act_sparsity_logs.items()},
                    step=step,
                )
            else:
                test_loss, test_acc = test_tuple

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {train_loss:.5f} -- Val Loss: {val_loss:.5f}"
                f" --Test Loss: {test_loss:.5f} --"
                f" Val Accuracy: {val_acc:.4f}"
                f" Test Accuracy: {test_acc:.4f}"
            )

            # For early stopping purposes
            if val_loss < best_val_loss:
                count = 0
                best_val_loss = val_loss
            else:
                count += 1

            if val_acc > best_acc:
                # Increment counters etc.
                count = 0
                best_loss, best_acc, best_epoch = val_loss, val_acc, epoch
                if valloader is not None:
                    best_test_loss, best_test_acc = test_loss, test_acc
                else:
                    best_test_loss, best_test_acc = best_loss, best_acc

                # Do some validation on improvement.
                # Evaluate on resolution 2 val and test sets
                print(f"[*] Running Epoch {epoch + 1} Res 2 Validation...")
                val2_tuple = val_fn(
                    forward_state,
                    skey,
                    model_cls,
                    aux_dataloaders["valloader2"],
                    int(seq_len // 2),
                    in_dim,
                    args.batchnorm,
                    step_rescale=2.0,
                    log_act_sparsity=args.log_act_sparsity in ("both", "val"),
                )
                if args.log_act_sparsity in ("both", "val"):
                    val2_loss, val2_acc, act_sparsity_log = val2_tuple
                    log_metrics(
                        {
                            f"act_sparsity/val2/{k}": v
                            for k, v in act_sparsity_log.items()
                        },
                        step=step,
                    )
                else:
                    val2_loss, val2_acc = val2_tuple

                print(f"[*] Running Epoch {epoch + 1} Res 2 Test...")
                test2_tuple = val_fn(
                    forward_state,
                    skey,
                    model_cls,
                    aux_dataloaders["testloader2"],
                    int(seq_len // 2),
                    in_dim,
                    args.batchnorm,
                    step_rescale=2.0,
                    log_act_sparsity=args.log_act_sparsity in ("both", "val"),
                )
                if args.log_act_sparsity in ("both", "val"):
                    test2_loss, test2_acc, act_sparsity_log = test2_tuple
                    log_metrics(
                        {
                            f"act_sparsity/test2/{k}": v
                            for k, v in act_sparsity_log.items()
                        },
                        step=step,
                    )
                else:
                    test2_loss, test2_acc = test2_tuple
                print(f"\n=>> Epoch {epoch + 1} Res 2 Metrics ===")
                print(
                    f"\tVal2 Loss: {val2_loss:.5f} --Test2 Loss:"
                    f" {test2_loss:.5f} -- Val Accuracy: {val2_acc:.4f} Test"
                    f" Accuracy: {test2_acc:.4f}"
                )

            # Print best accuracy & loss so far...
            print(
                f"\tBest Val Loss: {best_loss:.5f} -- Best Val Accuracy:"
                f" {best_acc:.4f} at Epoch {best_epoch + 1}\n\tBest Test Loss:"
                f" {best_test_loss:.5f} -- Best Test Accuracy:"
                f" {best_test_acc:.4f} at Epoch {best_epoch + 1}\n"
            )

            if valloader is not None:
                log_metrics(
                    {
                        "Training Loss": train_loss,
                        "Val loss": val_loss,
                        "Val Accuracy": val_acc,
                        "Test Loss": test_loss,
                        "Test Accuracy": test_acc,
                        # "Val2 loss": val2_loss,
                        # "Val2 Accuracy": val2_acc,
                        # "Test2 Loss": test2_loss,
                        # "Test2 Accuracy": test2_acc,
                    },
                    step=step,
                )
            else:
                log_metrics(
                    {
                        "Training Loss": train_loss,
                        "Val loss": val_loss,
                        "Val Accuracy": val_acc,
                    },
                    step=step,
                )
            log_best_metrics(
                {
                    "Best Val Loss": best_loss,
                    "Best Val Accuracy": best_acc,
                    "Best Epoch": best_epoch,
                    "Best Test Loss": best_test_loss,
                    "Best Test Accuracy": best_test_acc,
                },
                step=step,
            )

            # save checkpoint
            if chkpt_mngr is not None:
                chkpt_metadata["best_test_loss"] = best_test_loss.item()
                chkpt_metadata["best_test_acc"] = best_test_acc.item()
                chkpt_metadata["last_step"] = step
                chkpt_metadata["next_epoch"] = epoch + 1
                chkpt_mngr.save(
                    step=epoch + 1,
                    args=ocp.args.Composite(
                        state=ocp.args.StandardSave(training_state),
                        metadata=ocp.args.JsonSave(chkpt_metadata),
                    ),
                )
                chkpt_mngr.wait_until_finished()
                log_artifacts(os.path.join(args.checkpoint_dir, args.run_name))

            if count > args.early_stop_patience:
                break

    logger.info("ending mlflow gracefully")
    time.sleep(10)
    end_logging()
