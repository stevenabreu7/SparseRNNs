import logging
import os
from functools import partial, reduce

import jax.numpy as jnp

logger = logging.getLogger("ncl-dl-stack")
logger.setLevel(logging.INFO)
logger.propagate = False
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
ch.setFormatter(formatter)
logger.addHandler(ch)


def compute_eigenvalue_logs(state, clip_eigs=False):
    """Calculate statistics for the eigenvalues of the A matrix and log them"""
    logs = []
    for layername in state.params["encoder"].keys():
        if "layers_" in layername:
            real = state.params["encoder"][layername]["mixer"]["Lambda_re"]
            real = jnp.clip(real, None, -1e-4) if clip_eigs else real
            imag = state.params["encoder"][layername]["mixer"]["Lambda_im"]
            logs.append(
                {
                    f"eigenvalues/{layername}/real_max": jnp.max(real).item(),
                    f"eigenvalues/{layername}/real_min": jnp.min(real).item(),
                    f"eigenvalues/{layername}/imag_max": jnp.max(imag).item(),
                    f"eigenvalues/{layername}/imag_min": jnp.min(imag).item(),
                    f"eigenvalues/{layername}/real_mean": (jnp.mean(real).item()),
                    f"eigenvalues/{layername}/imag_mean": (jnp.mean(imag).item()),
                    f"eigenvalues/{layername}/real_std": jnp.std(real).item(),
                    f"eigenvalues/{layername}/imag_std": jnp.std(imag).item(),
                }
            )
    return reduce(lambda a, b: {**a, **b}, logs)


def setup_experiment_logging_fns(args, chkpt_metadata: dict = None):
    
    import wandb
    wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        job_type="model_training",
        config=vars(args),
        id=chkpt_metadata.get("wandb_id", None) if chkpt_metadata else None,
        resume="allow"
    )
    if chkpt_metadata is not None:
        chkpt_metadata["wandb_id"] = wandb.run.id

    log_metrics = wandb.log

    def _log_best_metrics(d: dict, step: int):
        for k, v in d.items():
            wandb.run.summary[k] = v

    log_best_metrics = _log_best_metrics
    log_artifacts = lambda *args, **kwargs: None
    logger.warning("Will not log artifacts to WandB")
    end_logging = wandb.finish

    return log_metrics, log_best_metrics, log_artifacts, end_logging
