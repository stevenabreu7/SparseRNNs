import argparse
import logging
import os
from functools import partial

import jax
from jax import numpy as jnp
from jax import random
from jax.scipy.linalg import block_diag
from orbax import checkpoint as ocp

from sparseRNNs.dataloaders import Datasets
from sparseRNNs.model.seq_model import (QBatchClassificationModel,
                                        QBatchRegressionModel, QRetrievalModel)
from sparseRNNs.model.ssm import init_qS5SSM
from sparseRNNs.model.ssm_init import make_DPLR_HiPPO
from sparseRNNs.train_helpers import (create_train_state, train_epoch,
                                      train_epoch_ndns, validate,
                                      validate_ndns)
from sparseRNNs.utils.logging import (compute_eigenvalue_logs, logger,
                                      setup_experiment_logging_fns)
from sparseRNNs.utils.pruning import pruning_recipe_map
from sparseRNNs.utils.quantization import quantization_recipe_map


def train(args):
    """
    Main function to train over a certain number of epochs
    """
    if args.debug:
        jax.config.update("jax_disable_jit", True)
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)

    logger.debug(args)

    args.log_syn_sparsity = args.pruning != "no_prune"
    if not args.log_syn_sparsity:
        logger.debug("No pruning config detected. Not logging sparsity.")

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
    else:
        train_fn = train_epoch
        val_fn = validate

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
    args.d_input = in_dim
    args.d_output = n_classes

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

    if args.reset_optimizer:
        pruning_cfg = pruning_recipe_map["no_prune"](
            epochs=args.epochs, steps_per_epoch=steps_per_epoch
        )
    else:
        pruning_cfg = pruning_recipe_map[args.pruning](
            epochs=args.epochs, steps_per_epoch=steps_per_epoch
        )

    ssm_init_fn = init_qS5SSM(
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
        q_config=q_config,
    )

    if retrieval:
        task = QRetrievalModel
    elif ndns:
        task = QBatchRegressionModel
    else:
        task = partial(QBatchClassificationModel, mode=args.mode)

    model_cls = partial(
        task,
        mixer_cls=ssm_init_fn,
        n_layers=args.n_layers,
        d_model=args.d_model,
        dropout=args.p_dropout,
        batchnorm=args.batchnorm,
        prenorm=args.prenorm,
        bn_momentum=args.bn_momentum,
        glu_variant=args.glu_variant,
        relufication=args.relufication,
        fuse_batchnorm_linear=args.fuse_batchnorm_linear,
        q_config=q_config,
        d_output=n_classes,
        padded=padded,
        use_batchnorm_bias=args.batchnorm_use_bias,
        use_batchnorm_scale=args.batchnorm_use_scale,
        topk=args.topk,
        approx_topk=args.approx_topk,
        quant_input=args.quant_input,
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
    )

    logger.info("Training State Initialized...")

    # Setup checkpointing & wandb
    chkpt_metadata = {
        "best_test_loss": 100000000,
        "best_test_acc": -10000.0,
        "wandb_id": None,
        "last_step": 0,
        "next_epoch": 0,
    }
    restored_state = None

    abstract_state = None
    # create checkpoint manager
    chkpt_mngr = None
    if args.load_run_name is not None and args.checkpoint_dir is not None:
        # create directory for model checkpoints
        chkpt_path = os.path.join(args.checkpoint_dir, args.load_run_name)
        os.makedirs(chkpt_path, exist_ok=True)

        # create checkpoint manager
        chkpt_options = ocp.CheckpointManagerOptions(
            save_interval_steps=args.checkpoint_interval_steps,
        )
        chkpt_mngr = ocp.CheckpointManager(
            directory=chkpt_path,
            item_names=("state", "metadata"),
            options=chkpt_options,
        )

        # check if we should load a checkpoint
        if chkpt_mngr.latest_step() is not None:
            abstract_state = jax.tree_util.tree_map(
                ocp.utils.to_shape_dtype_struct, state
            )
            restored = chkpt_mngr.restore(
                chkpt_mngr.latest_step(),
                args=ocp.args.Composite(
                    state=ocp.args.StandardRestore(abstract_state),
                    metadata=ocp.args.JsonRestore(),
                ),
            )
            restored_state = restored["state"]
            restored_metadata = restored["metadata"]

        # remove this checkpoint manager (avoid overwriting)
        chkpt_mngr = None

    if args.run_name is not None and args.checkpoint_dir is not None:
        # create directory for model checkpoints
        chkpt_path = os.path.join(args.checkpoint_dir, args.run_name)
        os.makedirs(chkpt_path, exist_ok=True)

        # create checkpoint manager
        chkpt_options = ocp.CheckpointManagerOptions(
            save_interval_steps=args.checkpoint_interval_steps,
        )
        chkpt_mngr = ocp.CheckpointManager(
            directory=chkpt_path,
            item_names=("state", "metadata"),
            options=chkpt_options,
        )

        # check if we should load a checkpoint (make sure we didn't already load one above)
        if chkpt_mngr.latest_step() is not None and restored_state is None:
            abstract_state = jax.tree_util.tree_map(
                ocp.utils.to_shape_dtype_struct, state
            )
            restored = chkpt_mngr.restore(
                chkpt_mngr.latest_step(),
                args=ocp.args.Composite(
                    state=ocp.args.StandardRestore(abstract_state),
                    metadata=ocp.args.JsonRestore(),
                ),
            )
            restored_state = restored["state"]
            restored_metadata = restored["metadata"]

    # Restore training state and other metadata
    if restored_state is not None:
        logger.info("Restoring train state from checkpoint...")
        if args.reset_optimizer:
            pruning_cfg = pruning_recipe_map[args.pruning](
                epochs=args.epochs, steps_per_epoch=steps_per_epoch
            )
            model_cls = partial(
                task,
                mixer_cls=ssm_init_fn,
                n_layers=args.n_layers,
                d_model=args.d_model,
                dropout=args.p_dropout,
                batchnorm=args.batchnorm,
                prenorm=args.prenorm,
                bn_momentum=args.bn_momentum,
                glu_variant=args.glu_variant,
                relufication=args.relufication,
                fuse_batchnorm_linear=args.fuse_batchnorm_linear,
                q_config=q_config,
                d_output=n_classes,
                padded=padded,
                use_batchnorm_bias=args.batchnorm_use_bias,
                use_batchnorm_scale=args.batchnorm_use_scale,
                topk=args.topk,
                approx_topk=args.approx_topk,
                quant_input=args.quant_input,
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
            )

            state = state.replace(
                # opt_state=state.opt_state._replace(
                #     inner_state=state.opt_state.inner_state._replace(
                #         inner_states=restored_state.opt_state.inner_state.inner_states
                #     ),
                # ),
                params=restored_state.params,
                # step=restored_state.step,
                batch_stats=restored_state.batch_stats,  # NOTE: only if batchnorm is used!
            )
            chkpt_metadata = {
                "best_test_loss": 100000000,
                "best_test_acc": -10000.0,
                "wandb_id": None,
                "last_step": 0,
                "next_epoch": 0,
            }
            restored_state = None
        else:
            state = restored_state
            chkpt_metadata = restored_metadata

    best_test_loss = chkpt_metadata["best_test_loss"]
    best_test_acc = chkpt_metadata["best_test_acc"]
    step = chkpt_metadata["last_step"]
    epoch_start = chkpt_metadata["next_epoch"]

    # Setup logging for experiment tracking, with wandb
    log_metrics, log_best_metrics, log_artifacts, end_logging = (
        setup_experiment_logging_fns(args, chkpt_metadata)
    )

    # log initial eigenvalues
    if epoch_start == 0 and args.log_eigenvalues:
        eigenvalue_logs = compute_eigenvalue_logs(state, clip_eigs=args.clip_eigs)
        log_metrics(eigenvalue_logs, step=0)

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
        logger.info(f"Starting Training Epoch {epoch + 1}...")
        step = epoch * len(trainloader) + 1
        train_rng, skey = random.split(train_rng)
        train_return_tuple = train_fn(
            state=state,
            rng=skey,
            model=model_cls,
            trainloader=trainloader,
            seq_len=seq_len,
            in_dim=in_dim,
            batchnorm=True,  # args.batchnorm,
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
            state, train_loss, sparsity_summary, act_sparsity_logs = train_return_tuple
        else:
            state, train_loss, sparsity_summary = train_return_tuple

        eigenvalue_logs = compute_eigenvalue_logs(state, clip_eigs=args.clip_eigs)
        act_sparsity_logs = (
            {f"act_sparsity/train/{k}": v for k, v in act_sparsity_logs.items()}
            if args.log_act_sparsity in ("both", "train")
            else {}
        )
        sparsity_logs = {"sparsity/" + k: float(v) for k, v in sparsity_summary.items()}

        lr_logs = {}
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
            state.opt_state.inner_state.inner_states["ssm"].inner_state, tuple
        ):
            lr_logs["ssm_lr"] = (
                state.opt_state.inner_state.inner_states["ssm"]
                .inner_state[0]
                .hyperparams["learning_rate"]
                .item()
            )
        else:
            logger.info("[warning] ssm lr not found")
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
            logger.info("[warning] lr not found")

        log_metrics(
            {
                **lr_logs,
                **act_sparsity_logs,
                **(sparsity_logs if args.log_syn_sparsity else {}),
                **(eigenvalue_logs if args.log_eigenvalues else {}),
                **(grad_logs if args.log_grads else {}),
            },
            step=step,
        )

        new_params = sparsity_updater.pre_forward_update(state.params, state.opt_state)
        forward_state = state.replace(params=new_params)

        if valloader is not None:
            logger.info(f"Running Epoch {epoch + 1} Validation...")
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

            logger.info(f"Running Epoch {epoch + 1} Test...")
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

            logger.info(f"=== Epoch {epoch + 1} Metrics ===")
            logger.info(f"Train Loss: {train_loss:.5f}")
            logger.info(f"Val Loss: {val_loss:.5f} -- Val Accuracy: {val_acc:.4f}")
            logger.info(f"Test Loss: {test_loss:.5f} -- Test Accuracy: {test_acc:.4f}")
        else:
            # use test set as validation set (e.g. IMDB)
            logger.info(f"Running Epoch {epoch + 1} Validation...")
            val_tuple = val_fn(
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
                # if ndns:
                #     val_loss, val_acc, val_dnsmos, act_sparsity_logs = val_tuple
                # else:
                #     val_loss, val_acc, act_sparsity_logs = val_tuple
                val_loss, val_acc, act_sparsity_logs = val_tuple
                log_metrics(
                    {f"act_sparsity/val/{k}": v for k, v in act_sparsity_logs.items()},
                    step=step,
                )
            else:
                # if ndns:
                #     val_loss, val_acc, val_dnsmos = val_tuple
                # else:
                #     val_loss, val_acc = val_tuple
                val_loss, val_acc = val_tuple

            logger.info(f"=== Epoch {epoch + 1} Metrics ===")
            logger.info(f"Train Loss: {train_loss:.5f}")
            logger.info(f"Val Loss: {val_loss:.5f} -- Val Accuracy: {val_acc:.4f}")

        # For early stopping purposes
        if val_loss < best_val_loss:
            count = 0
            best_val_loss = val_loss
        else:
            count += 1

        if val_acc > best_acc:
            count = 0
            best_loss, best_acc, best_epoch = val_loss, val_acc, epoch
            if valloader is not None:
                best_test_loss, best_test_acc = test_loss, test_acc
            else:
                best_test_loss, best_test_acc = best_loss, best_acc

        logger.info(
            f"Best Val Loss: {best_loss:.5f} -- Best Val Accuracy:"
            f" {best_acc:.4f} at Epoch {best_epoch + 1}"
        )
        logger.info(
            f"Best Test Loss: {best_test_loss:.5f} -- Best Test Accuracy:"
            f" {best_test_acc:.4f} at Epoch {best_epoch + 1}"
        )

        if valloader is not None:
            log_metrics(
                {
                    "Training Loss": train_loss,
                    "Val Loss": val_loss,
                    "Val Accuracy": val_acc,
                    "Test Loss": test_loss,
                    "Test Accuracy": test_acc,
                    # "Val DNSMOS OVRL": val_dnsmos[0],
                    # "Val DNSMOS SIG": val_dnsmos[1],
                    # "Val DNSMOS BAK": val_dnsmos[2],
                },
                step=step,
            )
        else:
            log_metrics(
                {
                    "Training Loss": train_loss,
                    "Val Loss": val_loss,
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

            logger.info(f"Saving Checkpoint for Epoch {epoch + 1}...")
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

    logger.info("Done!")
    end_logging()


def tune(args):
    import math

    import ray
    from hyperopt import hp
    from ray.tune.search.hyperopt import HyperOptSearch

    def train_raytune(args, tuned_params):
        args_copy = argparse.Namespace(**vars(args))
        for key, value in tuned_params.items():
            setattr(args_copy, key, value)
        train(args_copy)

    trainer = partial(train_raytune, args)
    trainer_gpu = ray.tune.with_resources(trainer, {"cpu": 12, "gpu": 1})
    space = {
        "ssm_lr_base": hp.loguniform("ssm_lr_base", math.log(1e-5), math.log(1e-1)),
    }

    # Docs at https://docs.ray.io/en/latest/tune/api/doc/ray.tune.se    arch.hyperopt.HyperOptSearch.html
    hyperopt_search = HyperOptSearch(
        space=space,
        metric="val_accuracy",
        n_initial_points=6,
        points_to_evaluate=[
            {
                "ssm_lr_base": args.ssm_lr_base,
            }
        ],
        random_state_seed=0,
        mode="max",
    )

    tuner = ray.tune.Tuner(
        trainable=trainer_gpu,
        tune_config=ray.tune.TuneConfig(
            num_samples=12,
            search_alg=hyperopt_search,
        ),
    )
    results = tuner.fit()
    best_result = results.get_best_result(metric="val_accuracy", mode="max")
    logger.info("Best hyperparameters found were: ", best_result.config)
