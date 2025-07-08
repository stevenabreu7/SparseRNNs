from functools import partial
from typing import Any, Tuple

import jax
import jax.numpy as np
import jaxpruner
import optax
from flax.core.frozen_dict import unfreeze
from flax.training import train_state
from jax.nn import one_hot
from tqdm import tqdm

from sparseRNNs.utils.logging import logger


@jax.jit
def si_snr_jax(target, estimate):
    """Calculates SI-SNR estiamte from target audio and estiamte audio. The
    audio sequene is expected to be a tensor/array of dimension more than 1.
    The last dimension is interpreted as time.

    The implementation is based on the example here:
    https://www.tutorialexample.com/wp-content/uploads/2021/12/SI-SNR-definition.png

    Parameters
    ----------
    target : Union[torch.tensor, np.ndarray]
        Target audio waveform.
    estimate : Union[torch.tensor, np.ndarray]
        Estimate audio waveform.

    Returns
    -------
    torch.tensor
        SI-SNR of each target and estimate pair.
    """
    EPS = 1e-8

    # zero mean to ensure scale invariance
    s_target = target - np.mean(target, axis=-1, keepdims=True)
    s_estimate = estimate - np.mean(estimate, axis=-1, keepdims=True)

    # <s, s'> / ||s||**2 * s
    pair_wise_dot = np.sum(s_target * s_estimate, axis=-1, keepdims=True)
    s_target_norm = np.sum(s_target**2, axis=-1, keepdims=True)
    pair_wise_proj = pair_wise_dot * s_target / s_target_norm

    e_noise = s_estimate - pair_wise_proj

    pair_wise_sdr = np.sum(pair_wise_proj**2, axis=-1) / (
        np.sum(e_noise**2, axis=-1) + EPS
    )
    return 10 * np.log10(pair_wise_sdr + EPS)


def flatten_pytree(pytree, prefix=""):
    flat_dict = {}

    # Recursive function to traverse the PyTree
    def _flatten(subtree, path):
        if isinstance(subtree, dict):
            for key, val in subtree.items():
                _flatten(val, f"{path}/{key}" if path else key)
        elif isinstance(subtree, (list, tuple)):
            for i, val in enumerate(subtree):
                _flatten(val, f"{path}/{i}" if path else str(i))
        else:  # It's a leaf
            flat_dict[path] = subtree

    _flatten(pytree, prefix)
    return flat_dict


def _compute_act_sparsity(act, atol=1e-6):
    return np.isclose(act, 0, atol=atol).sum() / act.size


def act_sparsity_filter_fn(mdl, method_name):
    modnames = {}
    return isinstance(mdl.name, str) and (mdl.name in modnames)


class TrainState(train_state.TrainState):
    batch_stats: Any


# LR schedulers
def linear_warmup(step, base_lr, end_step, lr_min=None):
    return base_lr * (step + 1) / end_step


def cosine_annealing(step, base_lr, end_step, lr_min=1e-6):
    # https://github.com/deepmind/optax/blob/master/optax/_src/schedule.py#L207#L240
    count = np.minimum(step, end_step)
    cosine_decay = 0.5 * (1 + np.cos(np.pi * count / end_step))
    decayed = (base_lr - lr_min) * cosine_decay + lr_min
    return decayed


def reduce_lr_on_plateau(input, factor=0.2, patience=20, lr_min=1e-6):
    lr, ssm_lr, count, new_acc, opt_acc = input
    if new_acc > opt_acc:
        count = 0
        opt_acc = new_acc
    else:
        count += 1

    if count > patience:
        lr = factor * lr
        ssm_lr = factor * ssm_lr
        count = 0

    if lr < lr_min:
        lr = lr_min
    if ssm_lr < lr_min:
        ssm_lr = lr_min

    return lr, ssm_lr, count, opt_acc


def constant_lr(step, base_lr, end_step, lr_min=None):
    return base_lr


def update_learning_rate_per_step(lr_params, state):
    decay_function, ssm_lr, lr, step, end_step, opt_config, lr_min = lr_params

    # Get decayed value
    lr_val = decay_function(step, lr, end_step, lr_min)
    ssm_lr_val = decay_function(step, ssm_lr, end_step, lr_min)
    step += 1

    # Update state
    state.opt_state.inner_states["regular"].inner_state.hyperparams["learning_rate"] = (
        np.array(lr_val, dtype=np.float32)
    )
    state.opt_state.inner_states["ssm"].inner_state.hyperparams["learning_rate"] = (
        np.array(ssm_lr_val, dtype=np.float32)
    )
    if opt_config in ["BandCdecay"]:
        # In this case we are applying the ssm learning rate to B, even though
        # we are also using weight decay on B
        state.opt_state.inner_states["none"].inner_state.hyperparams[
            "learning_rate"
        ] = np.array(ssm_lr_val, dtype=np.float32)

    return state, step


def map_nested_fn(fn):
    """
    Recursively apply `fn to the key-value pairs of a nested dict / pytree.
    We use this for some of the optax definitions below.
    """

    def map_fn(nested_dict):
        return {
            k: map_fn(v) if hasattr(v, "keys") else fn(k, v)
            for k, v in nested_dict.items()
        }

    return map_fn


def create_train_state(
    model_cls,
    rng,
    padded,
    retrieval,
    n_steps_total,
    training=True,
    n_warmup=0,
    in_dim=1,
    bsz=128,
    seq_len=784,
    weight_decay=0.01,
    batchnorm=False,
    opt_config="standard",
    ssm_lr=1e-3,
    lr=1e-3,
    lr_min=0.0,
    grad_clip_threshold=None,
    dt_global=False,
    pruner_cfg=None,
    static_quant=False,
    calibrating=False,
):
    """
    Initializes the training state using optax

    :param model_cls:
    :param rng:
    :param padded:
    :param retrieval:
    :param in_dim:
    :param bsz:
    :param seq_len:
    :param weight_decay:
    :param batchnorm:
    :param opt_config:
    :param ssm_lr:
    :param lr:
    :param grad_clip_threshold:
    :param dt_global:
    :return:
    """

    def inject_hyperparams_clipped(opt_cls):
        def inner_func(**kwargs):
            if grad_clip_threshold is None:
                return optax.inject_hyperparams(opt_cls)(**kwargs)
            else:
                return optax.chain(
                    optax.inject_hyperparams(opt_cls)(**kwargs),
                    optax.clip_by_global_norm(grad_clip_threshold),
                )

        return inner_func

    ssm_lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=ssm_lr / n_warmup if n_warmup > 0 else ssm_lr,
        peak_value=ssm_lr,
        warmup_steps=n_warmup,
        decay_steps=n_steps_total,
        end_value=lr_min,
        exponent=1.0,
    )
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=lr / n_warmup if n_warmup > 0 else lr,
        peak_value=lr,
        warmup_steps=n_warmup,
        decay_steps=n_steps_total,
        end_value=lr_min,
        exponent=1.0,
    )

    if padded:
        if retrieval:
            # For retrieval tasks we have two different sets of "documents"
            dummy_input = (
                np.ones((2 * bsz, seq_len, in_dim)),
                np.ones(2 * bsz),
            )
            integration_timesteps = np.ones(
                (
                    2 * bsz,
                    seq_len,
                )
            )
        else:
            dummy_input = (np.ones((bsz, seq_len, in_dim)), np.ones(bsz))
            integration_timesteps = np.ones(
                (
                    bsz,
                    seq_len,
                )
            )
    else:
        dummy_input = np.ones((bsz, seq_len, in_dim))
        integration_timesteps = np.ones(
            (
                bsz,
                seq_len,
            )
        )

    model = model_cls(training=training)
    init_rng, dropout_rng = jax.random.split(rng, num=2)
    variables = model.init(
        {"params": init_rng, "dropout": dropout_rng},
        dummy_input,
        integration_timesteps,
    )
    if batchnorm or (static_quant and calibrating):
        # params = variables["params"].unfreeze()
        params = unfreeze(variables["params"])
        batch_stats = variables["batch_stats"]
    else:
        # params = variables["params"].unfreeze()
        params = unfreeze(variables["params"])
        # Note: `unfreeze()` is for using Optax.

    sparsity_updater = jaxpruner.create_updater_from_config(pruner_cfg)

    if opt_config in ["qaft"]:
        logger.debug(
            "using quantization-aware fine-tuning optimization setup (SGD, no" " decay)"
        )
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: (
                    "ssm"
                    if k in ["B", "Lambda_re", "Lambda_im", "norm"]
                    else ("none" if k in [] else "regular")
                )
            )

        else:
            ssm_fn = map_nested_fn(
                lambda k, _: (
                    "ssm"
                    if k in ["B", "Lambda_re", "Lambda_im", "log_step", "norm"]
                    else ("none" if k in [] else "regular")
                )
            )
        tx = optax.multi_transform(
            {
                "none": inject_hyperparams_clipped(optax.sgd)(learning_rate=0.0),
                "ssm": inject_hyperparams_clipped(optax.sgd)(
                    learning_rate=ssm_lr_schedule, momentum=0.9
                ),
                "regular": inject_hyperparams_clipped(optax.sgd)(
                    learning_rate=lr_schedule, momentum=0.9
                ),
            },
            ssm_fn,
        )
    elif opt_config in ["standard"]:
        """This option applies weight decay to C, but B is kept with the
        SSM parameters with no weight decay.
        """
        logger.debug("Configuring standard optimization setup")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: (
                    "ssm"
                    if k in ["B", "Lambda_re", "Lambda_im", "norm"]
                    else ("none" if k in [] else "regular")
                )
            )

        else:
            ssm_fn = map_nested_fn(
                lambda k, _: (
                    "ssm"
                    if k in ["B", "Lambda_re", "Lambda_im", "log_step", "norm"]
                    else ("none" if k in [] else "regular")
                )
            )
        tx = optax.multi_transform(
            {
                "none": inject_hyperparams_clipped(optax.sgd)(learning_rate=0.0),
                "ssm": inject_hyperparams_clipped(optax.adam)(
                    learning_rate=ssm_lr_schedule
                ),
                "regular": inject_hyperparams_clipped(optax.adamw)(
                    learning_rate=lr_schedule, weight_decay=weight_decay
                ),
            },
            ssm_fn,
        )
    elif opt_config in ["BandCdecay"]:
        """This option applies weight decay to both C and B. Note we still apply the
        ssm learning rate to B.
        """
        logger.debug("Configuring optimization with B in AdamW setup")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: (
                    "ssm"
                    if k in ["Lambda_re", "Lambda_im", "norm"]
                    else ("none" if k in ["B"] else "regular")
                )
            )

        else:
            ssm_fn = map_nested_fn(
                lambda k, _: (
                    "ssm"
                    if k in ["Lambda_re", "Lambda_im", "log_step", "norm"]
                    else ("none" if k in ["B"] else "regular")
                )
            )
        tx = optax.multi_transform(
            {
                "none": inject_hyperparams_clipped(optax.adamw)(
                    learning_rate=ssm_lr_schedule, weight_decay=weight_decay
                ),
                "ssm": inject_hyperparams_clipped(optax.adam)(
                    learning_rate=ssm_lr_schedule
                ),
                "regular": inject_hyperparams_clipped(optax.adamw)(
                    learning_rate=lr_schedule, weight_decay=weight_decay
                ),
            },
            ssm_fn,
        )
    elif opt_config in ["BfastandCdecay"]:
        """This option applies weight decay to both C and B. Note here we apply
        faster global learning rate to B also.
        """
        logger.debug("Configuring optimization with B in AdamW setup with lr")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: (
                    "ssm"
                    if k in ["Lambda_re", "Lambda_im", "norm"]
                    else ("none" if k in [] else "regular")
                )
            )
        else:
            ssm_fn = map_nested_fn(
                lambda k, _: (
                    "ssm"
                    if k in ["Lambda_re", "Lambda_im", "log_step", "norm"]
                    else ("none" if k in [] else "regular")
                )
            )
        tx = optax.multi_transform(
            {
                "none": inject_hyperparams_clipped(optax.adamw)(learning_rate=0.0),
                "ssm": inject_hyperparams_clipped(optax.adam)(
                    learning_rate=ssm_lr_schedule
                ),
                "regular": inject_hyperparams_clipped(optax.adamw)(
                    learning_rate=lr_schedule, weight_decay=weight_decay
                ),
            },
            ssm_fn,
        )
    elif opt_config in ["noBCdecay"]:
        """This option does not apply weight decay to B or C. C is included
        with the SSM parameters and uses ssm learning rate.
        """
        logger.debug("Configuring optimization with C not in AdamW setup")
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: (
                    "ssm"
                    if k
                    in [
                        "B",
                        "C",
                        "C1",
                        "C2",
                        "D",
                        "Lambda_re",
                        "Lambda_im",
                        "norm",
                    ]
                    else ("none" if k in [] else "regular")
                )
            )
        else:
            ssm_fn = map_nested_fn(
                lambda k, _: (
                    "ssm"
                    if k
                    in [
                        "B",
                        "C",
                        "C1",
                        "C2",
                        "D",
                        "Lambda_re",
                        "Lambda_im",
                        "log_step",
                        "norm",
                    ]
                    else ("none" if k in [] else "regular")
                )
            )
        tx = optax.multi_transform(
            {
                "none": inject_hyperparams_clipped(optax.sgd)(learning_rate=0.0),
                "ssm": inject_hyperparams_clipped(optax.adam)(
                    learning_rate=ssm_lr_schedule
                ),
                "regular": inject_hyperparams_clipped(optax.adamw)(
                    learning_rate=lr_schedule, weight_decay=weight_decay
                ),
            },
            ssm_fn,
        )
    elif opt_config in ["constant"]:
        ssm_lr_schedule = optax.constant_schedule(ssm_lr)
        lr_schedule = optax.constant_schedule(lr)
        if dt_global:
            ssm_fn = map_nested_fn(
                lambda k, _: (
                    "ssm"
                    if k in ["B", "Lambda_re", "Lambda_im", "norm"]
                    else ("none" if k in [] else "regular")
                )
            )
        else:
            ssm_fn = map_nested_fn(
                lambda k, _: (
                    "ssm"
                    if k in ["B", "Lambda_re", "Lambda_im", "log_step", "norm"]
                    else ("none" if k in [] else "regular")
                )
            )
        tx = optax.multi_transform(
            {
                "none": inject_hyperparams_clipped(optax.sgd)(learning_rate=0.0),
                "ssm": inject_hyperparams_clipped(optax.adam)(
                    learning_rate=ssm_lr_schedule
                ),
                "regular": inject_hyperparams_clipped(optax.adamw)(
                    learning_rate=lr_schedule, weight_decay=weight_decay
                ),
            },
            ssm_fn,
        )
    else:
        raise ValueError(f"Optimization configuration {opt_config} not recognized.")

    tx = sparsity_updater.wrap_optax(tx)
    fn_is_complex = lambda x: x.dtype in [np.complex64, np.complex128]
    param_sizes = map_nested_fn(
        lambda k, param: param.size * (2 if fn_is_complex(param) else 1)
    )(params)
    logger.info(f"Trainable Parameters: {sum(jax.tree_leaves(param_sizes))}")

    if batchnorm or (static_quant and calibrating):
        return (
            TrainState.create(
                apply_fn=model.apply,
                params=params,
                tx=tx,
                batch_stats=batch_stats,
            ),
            sparsity_updater,
        )
    else:
        return (
            train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx),
            sparsity_updater,
        )


# Train and eval steps
@partial(np.vectorize, signature="(c),()->()")
def cross_entropy_loss(logits, label):
    one_hot_label = jax.nn.one_hot(label, num_classes=logits.shape[0])
    return -np.sum(one_hot_label * logits)


@partial(np.vectorize, signature="(c),()->()")
def compute_accuracy(logits, label):
    return np.argmax(logits) == label


def prep_batch(
    batch: tuple, seq_len: int, in_dim: int
) -> Tuple[np.ndarray, np.ndarray, np.array]:
    """
    Take a batch and convert it to a standard x/y format.
    :param batch:       (x, y, aux_data) as returned from dataloader.
    :param seq_len:     (int) length of sequence.
    :param in_dim:      (int) dimension of input.
    :return:
    """
    if len(batch) == 2:
        inputs, targets = batch
        aux_data = {}
    elif len(batch) == 3:
        inputs, targets, aux_data = batch
    else:
        raise RuntimeError("Err... not sure what I should do... Unhandled data type. ")

    # Convert to JAX.
    if not isinstance(inputs, np.ndarray):
        inputs = np.asarray(inputs.numpy())

    # Grab lengths from aux if it is there.
    lengths = aux_data.get("lengths", None)

    # Make all batches have same sequence length
    num_pad = seq_len - inputs.shape[1]
    if num_pad > 0:
        # Assuming vocab padding value is zero
        inputs = np.pad(
            inputs, ((0, 0), (0, num_pad)), "constant", constant_values=(0,)
        )

    # Inputs is either [n_batch, seq_len] or [n_batch, seq_len, in_dim].
    # If there are not three dimensions and trailing dimension is not equal to in_dim then
    # transform into one-hot.  This should be a fairly reliable fix.
    if (inputs.ndim < 3) and (inputs.shape[-1] != in_dim):
        inputs = one_hot(np.asarray(inputs), in_dim)

    # If there are lengths, bundle them up.
    if lengths is not None:
        lengths = np.asarray(lengths.numpy())
        full_inputs = (inputs.astype(float), lengths.astype(float))
    else:
        full_inputs = inputs.astype(float)

    # Convert and apply.
    if not isinstance(targets, np.ndarray):
        targets = np.array(targets.numpy())

    # If there is an aux channel containing the integration times, then add that.
    if "timesteps" in aux_data.keys():
        integration_timesteps = np.diff(np.asarray(aux_data["timesteps"].numpy()))
    else:
        integration_timesteps = np.ones((len(inputs), seq_len))

    return full_inputs, targets.astype(float), integration_timesteps


def train_epoch(
    state,
    rng,
    model,
    trainloader,
    seq_len,
    in_dim,
    batchnorm,
    loss_fn=cross_entropy_loss,
    sparsity_updater=None,
    return_grads=False,
    MAX_GRAD_NORM: float = 1.0,
    log_metrics_fn=None,
    step: int = None,
    log_act_sparsity=False,
    profiling: bool = False,
):
    """
    Training function for an epoch that loops over batches.
    """

    model = model(training=True)
    batch_losses = []
    batch_gradnorms = None
    batch_gradsparsity = 0.0
    batch_gradsparsity_tree = None
    batch_grads_logs = {}
    act_sparsities = []

    is_ste = isinstance(
        sparsity_updater,
        (jaxpruner.SteMagnitudePruning, jaxpruner.SteRandomPruning),
    )
    pre_op = jax.jit(sparsity_updater.pre_forward_update)
    post_op = jax.jit(sparsity_updater.post_gradient_update)
    grad_report = ""

    for batch_idx, batch in enumerate(tqdm(trainloader)):
        if profiling:
            jax.profiler.start_trace(
                "/home/sabreu/ncl-dl-stack/tmp/tensorboard",
                create_perfetto_trace=True,
            )
        inputs, labels, integration_times = prep_batch(batch, seq_len, in_dim)
        forward_state = state
        new_params = pre_op(state.params, state.opt_state)
        forward_state = state.replace(params=new_params)
        step_tuple = train_step(
            forward_state,
            rng,
            inputs,
            labels,
            integration_times,
            model,
            batchnorm,
            loss_act=loss_fn,
            return_grads=return_grads,
            log_act_sparsity=log_act_sparsity,
        )
        if return_grads and log_act_sparsity:
            state, loss, grads, act_sparsity = step_tuple
            act_sparsities.append(act_sparsity)
        elif return_grads:
            state, loss, grads = step_tuple
        elif log_act_sparsity:
            state, loss, act_sparsity = step_tuple
            act_sparsities.append(act_sparsity)
        else:
            state, loss = step_tuple

        if return_grads:
            # check that the scale parameters are zero
            scale_grad_zero = all(
                [
                    v.sum().item() == 0.0
                    for kp, v in jax.tree_util.tree_flatten_with_path(grads)[0]
                    if kp[-1].key.endswith("_scale")
                ]
            )
            if not scale_grad_zero:
                grad_report += f"Batch {batch_idx}: " + ",".join(
                    [
                        f"{v.sum().item():.4f}{'.'.join([e.key for e in kp])}"
                        for kp, v in jax.tree_util.tree_flatten_with_path(grads)[0]
                        if kp[-1].key.endswith("_scale")
                        if v.sum().item() != 0.0
                    ]
                )
            # compute gradient sparsity
            sparsity_tree = jax.tree_util.tree_map(
                lambda leaf: np.sum(leaf == 0) / leaf.size, grads
            )
            total_sparsity = np.mean(
                np.array(jax.tree_util.tree_leaves(sparsity_tree))
            ).item()
            batch_gradsparsity += total_sparsity
            if total_sparsity > 1e-3:
                # only log full sparsity tree if the average sparsity is non-negligible
                batch_gradsparsity_tree = (
                    sparsity_tree
                    if batch_gradsparsity_tree is None
                    else jax.tree_util.tree_map(
                        lambda x, y: x + y,
                        batch_gradsparsity_tree,
                        sparsity_tree,
                    )
                )
            # compute gradient norms
            grad_norms = jax.tree_util.tree_map(
                lambda x: np.linalg.norm(x.flatten()).item(), grads
            )
            batch_gradnorms = (
                grad_norms
                if batch_gradnorms is None
                else jax.tree_util.tree_map(
                    lambda x, y: x + y, batch_gradnorms, grad_norms
                )
            )

            # intermediate logging if gradient norms get large, or if sparsity is high
            if log_metrics_fn is not None and step is not None:
                max_grad_norm = np.max(
                    np.array(jax.tree_util.tree_leaves(grad_norms))
                ).item()
                if max_grad_norm > MAX_GRAD_NORM:
                    grad_norms
                    log_metrics_fn(
                        {
                            **flatten_pytree(grad_norms, "grad_norms"),
                            "grad_sparsity_avg": total_sparsity,
                        },
                        step=step + batch_idx,
                    )
                if total_sparsity > 0.01:
                    log_metrics_fn(
                        flatten_pytree(sparsity_tree, "grad_sparsity"),
                        step=step + batch_idx,
                    )

        post_params = post_op(state.params, state.opt_state)
        state = state.replace(params=post_params)
        batch_losses.append(loss)

        # jax.clear_backends()  # NOTE: attempt to fix memory leak
        # jax.clear_caches()  # NOTE: attempt to fix memory leak
        if profiling:
            jax.profiler.stop_trace()
            jax.profiler.save_device_memory_profile(
                f"profiling/memory_{batch_idx}.prof"
            )

    if len(grad_report) > 0:
        print("DETECTED NON-ZERO GRADIENTS FOR SCALE PARAMETERS")
        print(grad_report)
    if return_grads:
        avg_fn = lambda x: x / len(batch_losses)
        batch_gradnorms = jax.tree_util.tree_map(avg_fn, batch_gradnorms)
        batch_gradnorms = flatten_pytree(batch_gradnorms, "grad_norms")
        batch_gradsparsity = avg_fn(batch_gradsparsity)
        batch_grad_sparsity_tree = (
            {}
            if batch_gradsparsity < 1e-3
            else flatten_pytree(
                jax.tree_util.tree_map(avg_fn, batch_gradsparsity_tree),
                "grad_sparsity",
            )
        )
        batch_grads_logs = {
            "grad_sparsity_avg": batch_gradsparsity,
            **batch_gradnorms,
            **batch_grad_sparsity_tree,
        }

    sparsity_summary = (
        jaxpruner.summarize_sparsity(new_params, only_total_sparsity=False)
        if is_ste
        else jaxpruner.summarize_sparsity(state.params, only_total_sparsity=False)
    )

    if log_act_sparsity:
        # aggregate mean
        act_sparsities = jax.tree.map(
            lambda *xs: np.mean(np.stack(xs)), *act_sparsities
        )
        act_sparsities = jax.tree.map(
            lambda x: x if x is None else x.item(), act_sparsities
        )
        act_sparsities = jax.tree.map(
            lambda x: x[0] if isinstance(x, tuple) else x, act_sparsities
        )
        # clean up
        clean_up_fn = lambda x: (
            x["__call__"] if isinstance(x, dict) and "__call__" in x else x
        )
        is_leaf_fn = lambda node: isinstance(node, dict) and "__call__" in node
        act_sparsities = jax.tree.map(clean_up_fn, act_sparsities, is_leaf=is_leaf_fn)
        # flatten
        act_sparsities = flatten_pytree(act_sparsities)

    batch_grads_logs = (batch_grads_logs,) if return_grads else ()
    act_sparsity_logs = (act_sparsities,) if log_act_sparsity else ()
    return (
        (state, np.mean(np.array(batch_losses)), sparsity_summary)
        + batch_grads_logs
        + act_sparsity_logs
    )


def train_epoch_ndns(
    state,
    rng,
    model,
    trainloader,
    seq_len: int,
    in_dim: int,
    batchnorm: bool,
    loss_fn=cross_entropy_loss,
    sparsity_updater=None,
    return_grads=False,
    MAX_GRAD_NORM: float = 1.0,
    log_metrics_fn=None,
    step: int = None,
    log_act_sparsity=False,
):
    """
    Training function for an epoch that loops over batches.
    """

    model = model(training=True)
    batch_losses = []
    batch_gradnorms = None
    batch_gradsparsity = 0.0
    batch_gradsparsity_tree = None
    batch_grads_logs = {}
    act_sparsities = []

    is_ste = isinstance(
        sparsity_updater,
        (jaxpruner.SteMagnitudePruning, jaxpruner.SteRandomPruning),
    )
    pre_op = jax.jit(sparsity_updater.pre_forward_update)
    post_op = jax.jit(sparsity_updater.post_gradient_update)

    for batch_idx, batch in enumerate(tqdm(trainloader)):

        # inputs, labels, integration_times = prep_batch(batch, seq_len, in_dim)
        noisy, clean, noise = batch
        # convert from pytorch to jax

        # Convert PyTorch tensors to NumPy arrays
        noisy_np = noisy.numpy()
        clean_np = clean.numpy()
        noise_np = noise.numpy()

        # Convert NumPy arrays to JAX arrays
        noisy = np.array(noisy_np)
        clean = np.array(clean_np)
        noise = np.array(noise_np)

        batch_size = len(noisy)
        integration_times = np.ones((batch_size, seq_len))

        noisy_mag, noisy_phase = stft_splitter(noisy)
        clean_mag, clean_phase = stft_splitter(clean)

        forward_state = state
        new_params = pre_op(state.params, state.opt_state)
        forward_state = state.replace(params=new_params)

        step_tuple = train_step_ndns(
            forward_state,
            rng,
            noisy_mag,
            noisy_phase,
            clean_mag,
            clean,
            integration_times,
            model,
            batchnorm,
            loss_act=loss_fn,
            return_grads=return_grads,
            log_act_sparsity=log_act_sparsity,
        )
        if return_grads and log_act_sparsity:
            state, loss, grads, act_sparsity = step_tuple
            act_sparsities.append(act_sparsity)
        elif return_grads:
            state, loss, grads = step_tuple
        elif log_act_sparsity:
            state, loss, act_sparsity = step_tuple
            act_sparsities.append(act_sparsity)
        else:
            state, loss = step_tuple

        if return_grads:
            # compute gradient sparsity
            sparsity_tree = jax.tree_util.tree_map(
                lambda leaf: np.sum(leaf == 0) / leaf.size, grads
            )
            total_sparsity = np.mean(
                np.array(jax.tree_util.tree_leaves(sparsity_tree))
            ).item()
            batch_gradsparsity += total_sparsity
            if total_sparsity > 1e-3:
                # only log full sparsity tree if the average sparsity is non-negligible
                batch_gradsparsity_tree = (
                    sparsity_tree
                    if batch_gradsparsity_tree is None
                    else jax.tree_util.tree_map(
                        lambda x, y: x + y,
                        batch_gradsparsity_tree,
                        sparsity_tree,
                    )
                )
            # compute gradient norms
            grad_norms = jax.tree_util.tree_map(
                lambda x: np.linalg.norm(x.flatten()).item(), grads
            )
            batch_gradnorms = (
                grad_norms
                if batch_gradnorms is None
                else jax.tree_util.tree_map(
                    lambda x, y: x + y, batch_gradnorms, grad_norms
                )
            )

            # intermediate logging if gradient norms get large, or if sparsity is high
            if log_metrics_fn is not None and step is not None:
                max_grad_norm = np.max(
                    np.array(jax.tree_util.tree_leaves(grad_norms))
                ).item()
                if max_grad_norm > MAX_GRAD_NORM:
                    grad_norms
                    log_metrics_fn(
                        {
                            **flatten_pytree(grad_norms, "grad_norms"),
                            "grad_sparsity_avg": total_sparsity,
                        },
                        step=step + batch_idx,
                    )
                if total_sparsity > 0.01:
                    log_metrics_fn(
                        flatten_pytree(sparsity_tree, "grad_sparsity"),
                        step=step + batch_idx,
                    )

        post_params = post_op(state.params, state.opt_state)
        state = state.replace(params=post_params)
        batch_losses.append(loss)

    if return_grads:
        avg_fn = lambda x: x / len(batch_losses)
        batch_gradnorms = jax.tree_util.tree_map(avg_fn, batch_gradnorms)
        batch_gradnorms = flatten_pytree(batch_gradnorms, "grad_norms")
        batch_gradsparsity = avg_fn(batch_gradsparsity)
        batch_grad_sparsity_tree = (
            {}
            if batch_gradsparsity < 1e-3
            else flatten_pytree(
                jax.tree_util.tree_map(avg_fn, batch_gradsparsity_tree),
                "grad_sparsity",
            )
        )
        batch_grads_logs = {
            "grad_sparsity_avg": batch_gradsparsity,
            **batch_gradnorms,
            **batch_grad_sparsity_tree,
        }

    sparsity_summary = (
        jaxpruner.summarize_sparsity(new_params, only_total_sparsity=False)
        if is_ste
        else jaxpruner.summarize_sparsity(state.params, only_total_sparsity=False)
    )

    if log_act_sparsity:
        # aggregate mean
        act_sparsities = jax.tree.map(
            lambda *xs: np.mean(np.stack(xs)), *act_sparsities
        )
        act_sparsities = jax.tree.map(
            lambda x: x if x is None else x.item(), act_sparsities
        )
        act_sparsities = jax.tree.map(
            lambda x: x[0] if isinstance(x, tuple) else x, act_sparsities
        )
        # clean up
        clean_up_fn = lambda x: (
            x["__call__"] if isinstance(x, dict) and "__call__" in x else x
        )
        is_leaf_fn = lambda node: isinstance(node, dict) and "__call__" in node
        act_sparsities = jax.tree.map(clean_up_fn, act_sparsities, is_leaf=is_leaf_fn)
        # flatten
        act_sparsities = flatten_pytree(act_sparsities)

    batch_grads_logs = (batch_grads_logs,) if return_grads else ()
    act_sparsity_logs = (act_sparsities,) if log_act_sparsity else ()
    return (
        (state, np.mean(np.array(batch_losses)), sparsity_summary)
        + batch_grads_logs
        + act_sparsity_logs
    )


def validate(
    state,
    skey,
    model,
    testloader,
    seq_len,
    in_dim,
    batchnorm,
    loss_fn=cross_entropy_loss,
    calculate_acc=True,
    step_rescale=1.0,
    log_act_sparsity=False,
    return_state=False,
):
    """Validation function that loops over batches"""
    model = model(training=False, step_rescale=step_rescale)
    losses, accuracies, preds = np.array([]), np.array([]), np.array([])
    act_sparsities = []
    for batch_idx, batch in enumerate(tqdm(testloader)):
        inputs, labels, integration_timesteps = prep_batch(batch, seq_len, in_dim)
        eval_tuple = eval_step(
            inputs,
            labels,
            skey,
            integration_timesteps,
            state,
            model,
            batchnorm,
            loss_act=loss_fn,
            log_act_sparsity=log_act_sparsity,
            return_state=return_state,
        )
        if return_state:
            *eval_tuple, state = eval_tuple
        if log_act_sparsity:
            *eval_tuple, act_sp = eval_tuple
            act_sparsities.append(act_sp)
        loss, acc, _ = eval_tuple

        losses = np.append(losses, loss)
        if calculate_acc:
            accuracies = np.append(accuracies, acc)

    if log_act_sparsity:
        # aggregate mean
        act_sparsities = jax.tree.map(
            lambda *xs: np.mean(np.stack(xs)), *act_sparsities
        )
        act_sparsities = jax.tree.map(
            lambda x: x if x is None else x.item(), act_sparsities
        )
        act_sparsities = jax.tree.map(
            lambda x: x[0] if isinstance(x, tuple) else x, act_sparsities
        )
        # clean up
        clean_up_fn = lambda x: (
            x["__call__"] if isinstance(x, dict) and "__call__" in x else x
        )
        is_leaf_fn = lambda node: isinstance(node, dict) and "__call__" in node
        act_sparsities = jax.tree.map(clean_up_fn, act_sparsities, is_leaf=is_leaf_fn)
        # flatten
        act_sparsities = flatten_pytree(act_sparsities)

    aveloss = np.mean(losses)
    aveaccu = np.mean(accuracies) if compute_accuracy else None

    ret_act_sparsity = (act_sparsities,) if log_act_sparsity else ()
    ret_state = (state,) if return_state else ()
    return (aveloss, aveaccu) + ret_act_sparsity + ret_state


def capture_intermediates(
    state,
    skey,
    model,
    dataloader,
    seq_len,
    in_dim,
    batchnorm,
    step_rescale=1.0,
):
    model = model(training=False, step_rescale=step_rescale)
    batch = next(iter(dataloader))
    inputs, labels, integration_timesteps = prep_batch(batch, seq_len, in_dim)
    intermediates = capture_intermediates_step(
        inputs,
        skey,
        integration_timesteps,
        state,
        model,
        batchnorm,
    )
    return intermediates, inputs, labels


@partial(jax.jit, static_argnums=(4, 5))
def capture_intermediates_step(
    batch_inputs,
    skey,
    batch_integration_timesteps,
    state,
    model,
    batchnorm,
):
    forward_state = {"params": state.params}
    if batchnorm:
        forward_state["batch_stats"] = state.batch_stats
    _, mod_vars = model.apply(
        forward_state,
        batch_inputs,
        batch_integration_timesteps,
        rngs={"params": skey},
        mutable=["intermediates"],
        capture_intermediates=True,
    )
    return mod_vars["intermediates"]


def capture_intermediates_ndns(
    state,
    skey,
    model,
    dataloader,
    seq_len,
    in_dim,
    batchnorm,
    step_rescale=1.0,
):
    model = model(training=False, step_rescale=step_rescale)
    for _, batch in enumerate(tqdm(dataloader)):
        # inputs, labels, integration_timesteps = prep_batch(batch, seq_len, in_dim)

        noisy, clean, noise = batch
        # convert from pytorch to jax

        # Convert PyTorch tensors to NumPy arrays
        noisy_np = noisy.numpy()
        clean_np = clean.numpy()
        noise_np = noise.numpy()

        # Convert NumPy arrays to JAX arrays
        noisy = np.array(noisy_np)
        clean = np.array(clean_np)
        noise = np.array(noise_np)

        batch_size = len(noisy)
        integration_times = np.ones((batch_size, seq_len))

        noisy_mag, noisy_phase = stft_splitter(noisy)
        clean_mag, clean_phase = stft_splitter(clean)

        intermediates = capture_intermediates_ndns_step(
            noisy_mag,
            noisy_phase,
            clean_mag,
            clean,
            skey,
            integration_times,
            state,
            model,
            batchnorm,
        )

        # create inputs and labels
        stft_mag_mean = 0.0007
        noisy_mag_mean_sub = noisy_mag - stft_mag_mean
        noisy_mag_mean_sub = np.transpose(noisy_mag_mean_sub, (0, 2, 1))
        inputs = noisy_mag_mean_sub
        labels = clean_mag

        # NOTE: only returning intermediates for a single batch
        return intermediates, inputs, labels


@partial(jax.jit, static_argnums=(7, 8))
def capture_intermediates_ndns_step(
    noisy_mag,
    noisy_phase,
    clean_mag,
    clean,
    skey,
    integration_times,
    state,
    model,
    batchnorm,
):
    capture_intermediates = True
    stft_mag_mean = 0.0007
    noisy_mag_mean_sub = noisy_mag - stft_mag_mean
    noisy_mag_mean_sub = np.transpose(noisy_mag_mean_sub, (0, 2, 1))

    mutable = ["intermediates"]
    modeldict = {"params": state.params}
    if batchnorm:
        modeldict = {"params": state.params, "batch_stats": state.batch_stats}

    _, mod_vars = model.apply(
        modeldict,
        noisy_mag_mean_sub,
        integration_times,
        rngs={"params": skey},
        mutable=mutable,
        capture_intermediates=capture_intermediates,
    )
    return mod_vars["intermediates"]


def validate_ndns(
    state,
    skey,
    model,
    testloader,
    seq_len,
    in_dim,
    batchnorm,
    loss_fn=cross_entropy_loss,
    step_rescale=1.0,
    log_act_sparsity=False,
    return_state=False,
):
    """Validation function that loops over batches"""

    # dnsmos = DNSMOS()

    dnsmos_cleaned = np.zeros(3)

    model = model(training=False, step_rescale=step_rescale)
    losses, accuracies, preds = np.array([]), np.array([]), np.array([])
    act_sparsities = []
    for batch_idx, batch in enumerate(tqdm(testloader)):

        noisy, clean, noise = batch

        # Convert PyTorch tensors to NumPy arrays
        noisy_np = noisy.numpy()
        clean_np = clean.numpy()
        noise_np = noise.numpy()

        # Convert NumPy arrays to JAX arrays
        noisy = np.array(noisy_np)
        clean = np.array(clean_np)
        noise = np.array(noise_np)

        batch_size = len(noisy)
        integration_times = np.ones((batch_size, seq_len))

        noisy_mag, noisy_phase = stft_splitter(noisy)
        clean_mag, clean_phase = stft_splitter(clean)

        eval_tuple = eval_step_ndns(
            noisy_mag,
            noisy_phase,
            clean_mag,
            clean,
            skey,
            integration_times,
            state,
            model,
            batchnorm,
            loss_act=loss_fn,
            log_act_sparsity=log_act_sparsity,
            return_state=return_state,
        )
        if return_state:
            *eval_tuple, state = eval_tuple
        if log_act_sparsity:
            *eval_tuple, act_sp = eval_tuple
            act_sparsities.append(act_sp)
        loss, acc, pred = eval_tuple

        losses = np.append(losses, loss)
        accuracies = np.append(accuracies, acc)

    if log_act_sparsity:
        # aggregate mean
        act_sparsities = jax.tree.map(
            lambda *xs: np.mean(np.stack(xs)), *act_sparsities
        )
        act_sparsities = jax.tree.map(
            lambda x: x if x is None else x.item(), act_sparsities
        )
        act_sparsities = jax.tree.map(
            lambda x: x[0] if isinstance(x, tuple) else x, act_sparsities
        )
        # clean up
        clean_up_fn = lambda x: (
            x["__call__"] if isinstance(x, dict) and "__call__" in x else x
        )
        is_leaf_fn = lambda node: isinstance(node, dict) and "__call__" in node
        act_sparsities = jax.tree.map(clean_up_fn, act_sparsities, is_leaf=is_leaf_fn)
        # flatten
        act_sparsities = flatten_pytree(act_sparsities)

    aveloss = np.mean(losses)
    aveaccu = np.mean(accuracies)

    avednsmos_cleaned = dnsmos_cleaned / len(testloader.dataset)

    ret_act_sparsity = (act_sparsities,) if log_act_sparsity else ()
    ret_state = (state,) if return_state else ()
    return (aveloss, aveaccu) + ret_act_sparsity + ret_state


@partial(jax.jit, static_argnums=(5, 6, 7, 8, 9))
def train_step(
    state,
    rng,
    batch_inputs,
    batch_labels,
    batch_integration_timesteps,
    model,
    batchnorm,
    loss_act=cross_entropy_loss,
    return_grads=False,
    log_act_sparsity=False,
):
    """Performs a single training step given a batch of data"""
    rng, drop_rng = jax.random.split(rng)

    def loss_fn(params):
        filter_fn = act_sparsity_filter_fn
        capture_intermediates = filter_fn if log_act_sparsity else None
        mutables = []
        apply_state = {"params": params}
        if log_act_sparsity:
            mutables.append("intermediates")
        if batchnorm:
            mutables.append("batch_stats")
            apply_state["batch_stats"] = state.batch_stats

        logits, mod_vars = model.apply(
            apply_state,
            batch_inputs,
            batch_integration_timesteps,
            rngs={"dropout": drop_rng, "params": rng},
            mutable=mutables,
            capture_intermediates=capture_intermediates,
        )
        loss = np.mean(loss_act(logits, batch_labels))
        return loss, (mod_vars, logits)

    (loss, (mod_vars, logits)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params
    )

    keys_map_fn = lambda x: {k: k for k in x.keys()}
    is_leaf_dict_fn = lambda x: isinstance(x, dict) and not isinstance(
        list(x.values())[0], dict
    )
    gnames = jax.tree.map(keys_map_fn, grads, is_leaf=is_leaf_dict_fn)
    grads = jax.tree.map(
        lambda pn, p: np.zeros_like(p) if pn.endswith("_scale") else p,
        gnames,
        grads,
    )

    if log_act_sparsity:
        act_sparsity_logs = jax.tree_util.tree_map(
            lambda x: _compute_act_sparsity(x), mod_vars["intermediates"]
        )

    if batchnorm:
        state = state.apply_gradients(grads=grads, batch_stats=mod_vars["batch_stats"])
    else:
        state = state.apply_gradients(grads=grads)

    return (
        (state, loss)
        + ((grads,) if return_grads else ())
        + ((act_sparsity_logs,) if log_act_sparsity else ())
    )


@jax.jit
def stft_splitter(audio):
    nfft = 512
    hop_length = 128
    noverlap = nfft - hop_length
    _, _, jax_stft = jax.scipy.signal.stft(
        audio,
        nperseg=nfft,
        nfft=nfft,
        noverlap=noverlap,
        window="boxcar",
        return_onesided=True,
    )
    jax_stft_mag = np.abs(jax_stft)
    jax_stft_phase = np.angle(jax_stft)
    return jax_stft_mag, jax_stft_phase


@jax.jit
def stft_mixer(stft_mag, stft_angle):
    nfft = 512
    hop_length = 128
    noverlap = nfft - hop_length
    _, audio = jax.scipy.signal.istft(
        stft_mag * np.exp(1j * stft_angle),
        nperseg=nfft,
        nfft=nfft,
        window="boxcar",
        noverlap=noverlap,
        input_onesided=True,
    )
    return audio


@partial(jax.jit, static_argnums=(7, 8, 9, 10, 11))
def train_step_ndns(
    state,
    rng,
    noisy_mag,
    noisy_phase,
    clean_mag,
    clean,
    batch_integration_timesteps,
    model,
    batchnorm,
    loss_act=None,
    return_grads=False,
    log_act_sparsity=False,
):
    """Performs a single training step given a batch of data"""
    rng, drop_rng = jax.random.split(rng)  # moved here from train_epoch

    def loss_fn(params):
        filter_fn = act_sparsity_filter_fn
        capture_intermediates = filter_fn if log_act_sparsity else None

        stft_mag_mean = 0.0007
        noisy_mag_mean_sub = noisy_mag - stft_mag_mean
        noisy_mag_mean_sub = np.transpose(noisy_mag_mean_sub, (0, 2, 1))

        if batchnorm:
            # adding params rng to make aqt/flax happy
            mask, mod_vars = model.apply(
                {"params": params, "batch_stats": state.batch_stats},
                noisy_mag_mean_sub,
                batch_integration_timesteps,
                rngs={"dropout": drop_rng, "params": rng},
                mutable=["intermediates", "batch_stats"],
                capture_intermediates=capture_intermediates,
            )
        else:
            mask, mod_vars = model.apply(
                {"params": params},
                noisy_mag_mean_sub,
                batch_integration_timesteps,
                rngs={"dropout": drop_rng, "params": rng},
                mutable=["intermediates"],
                capture_intermediates=capture_intermediates,
            )

        mask = np.transpose(mask, (0, 2, 1))
        cleaned_mag = noisy_mag * (1.0 + mask)
        cleaned = stft_mixer(cleaned_mag, noisy_phase)
        si_snr_score = si_snr_jax(cleaned, clean)

        lam = 0.001
        loss = lam * np.mean((cleaned_mag - clean_mag) ** 2) + (
            100 - np.mean(si_snr_score)
        )

        return loss, (mod_vars, mask)

    (loss, (mod_vars, mask)), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        state.params
    )

    if log_act_sparsity:
        act_sparsity_logs = jax.tree_util.tree_map(
            lambda x: _compute_act_sparsity(x), mod_vars["intermediates"]
        )

    if batchnorm:
        state = state.apply_gradients(grads=grads, batch_stats=mod_vars["batch_stats"])
    else:
        state = state.apply_gradients(grads=grads)

    return (
        (state, loss)
        + ((grads,) if return_grads else ())
        + ((act_sparsity_logs,) if log_act_sparsity else ())
    )


@partial(jax.jit, static_argnums=(5, 6, 7, 8, 9, 10))
def eval_step(
    batch_inputs,
    batch_labels,
    skey,
    batch_integration_timesteps,
    state,
    model,
    batchnorm,
    loss_act=cross_entropy_loss,
    calculate_acc=True,
    log_act_sparsity=False,
    return_state=False,
):
    filter_fn = act_sparsity_filter_fn
    capture_intermediates = filter_fn if log_act_sparsity else False
    mutable = []
    if log_act_sparsity:
        mutable.append("intermediates")
    if return_state:
        mutable.append("batch_stats")

    if batchnorm or return_state:
        logits, mod_vars = model.apply(
            {"params": state.params, "batch_stats": state.batch_stats},
            batch_inputs,
            batch_integration_timesteps,
            rngs={"params": skey},
            mutable=mutable,
            capture_intermediates=capture_intermediates,
        )
        if return_state:
            state = state.replace(batch_stats=mod_vars["batch_stats"])
    else:
        logits, mod_vars = model.apply(
            {"params": state.params},
            batch_inputs,
            batch_integration_timesteps,
            rngs={"params": skey},
            mutable=mutable,
            capture_intermediates=capture_intermediates,
        )

    if log_act_sparsity:
        act_sparsity_logs = jax.tree_util.tree_map(
            lambda x: _compute_act_sparsity(x), mod_vars["intermediates"]
        )

    losses = loss_act(logits, batch_labels)
    accs = None
    if calculate_acc:
        accs = compute_accuracy(logits, batch_labels)

    ret_act_sparsity = (act_sparsity_logs,) if log_act_sparsity else ()
    ret_state = (state,) if return_state else ()
    return (losses, accs, logits) + ret_act_sparsity + ret_state


@partial(jax.jit, static_argnums=(7, 8, 9, 10, 11))
def eval_step_ndns(
    noisy_mag,
    noisy_phase,
    clean_mag,
    clean,
    skey,
    batch_integration_timesteps,
    state,
    model,
    batchnorm,
    loss_act=None,
    log_act_sparsity=False,
    return_state=False,
):
    filter_fn = act_sparsity_filter_fn
    capture_intermediates = filter_fn if log_act_sparsity else None

    stft_mag_mean = 0.0007
    noisy_mag_mean_sub = noisy_mag - stft_mag_mean

    noisy_mag_mean_sub = np.transpose(noisy_mag_mean_sub, (0, 2, 1))

    mutable = ["intermediates"]
    if return_state:
        mutable.append("batch_stats")

    if batchnorm or return_state:
        mask, mod_vars = model.apply(
            {"params": state.params, "batch_stats": state.batch_stats},
            noisy_mag_mean_sub,
            batch_integration_timesteps,
            rngs={"params": skey},
            mutable=mutable,
            capture_intermediates=capture_intermediates,
        )
        if return_state:
            state = state.replace(batch_stats=mod_vars["batch_stats"])
    else:
        mask, mod_vars = model.apply(
            {"params": state.params},
            noisy_mag_mean_sub,
            batch_integration_timesteps,
            rngs={"params": skey},
            mutable=mutable,
            capture_intermediates=capture_intermediates,
        )

    if log_act_sparsity:
        act_sparsity_logs = jax.tree_util.tree_map(
            lambda x: _compute_act_sparsity(x), mod_vars["intermediates"]
        )

    mask = np.transpose(mask, (0, 2, 1))
    cleaned_mag = noisy_mag * (1.0 + mask)
    cleaned = stft_mixer(cleaned_mag, noisy_phase)
    si_snr_score = si_snr_jax(cleaned, clean)

    lam = 0.001
    losses = lam * np.mean((cleaned_mag - clean_mag) ** 2, axis=(1, 2)) + (
        100 - si_snr_score
    )
    accs = si_snr_score

    ret_act_sparsity = (act_sparsity_logs,) if log_act_sparsity else ()
    ret_state = (state,) if return_state else ()
    return (losses, accs, cleaned) + ret_act_sparsity + ret_state
