from functools import partial
from typing import Callable, Dict

from ml_collections import ConfigDict


def iterative_ste_magnitude_pruning(
    epochs: int, steps_per_epoch: int, target_sparsity: float
) -> ConfigDict:
    return ConfigDict(
        {
            "algorithm": "magnitude_ste",
            "update_freq": int(steps_per_epoch / 2),
            "update_end_step": int(0.9 * epochs * steps_per_epoch),
            "update_start_step": int(0.05 * epochs * steps_per_epoch),
            "sparsity": target_sparsity,
            "dist_type": "erk",
        }
    )


pruning_recipe_map: Dict[str, Callable[[int, int], ConfigDict]] = {
    "no_prune": lambda epochs, steps_per_epoch: ConfigDict({"algorithm": "no_prune"}),
    "iterative-ste-mag-0.1": partial(
        iterative_ste_magnitude_pruning, target_sparsity=0.1
    ),
    "iterative-ste-mag-0.2": partial(
        iterative_ste_magnitude_pruning, target_sparsity=0.2
    ),
    "iterative-ste-mag-0.3": partial(
        iterative_ste_magnitude_pruning, target_sparsity=0.3
    ),
    "iterative-ste-mag-0.4": partial(
        iterative_ste_magnitude_pruning, target_sparsity=0.4
    ),
    "iterative-ste-mag-0.5": partial(
        iterative_ste_magnitude_pruning, target_sparsity=0.5
    ),
    "iterative-ste-mag-0.6": partial(
        iterative_ste_magnitude_pruning, target_sparsity=0.6
    ),
    "iterative-ste-mag-0.7": partial(
        iterative_ste_magnitude_pruning, target_sparsity=0.7
    ),
    "iterative-ste-mag-0.8": partial(
        iterative_ste_magnitude_pruning, target_sparsity=0.8
    ),
    "iterative-ste-mag-0.9": partial(
        iterative_ste_magnitude_pruning, target_sparsity=0.9
    ),
    "iterative-ste-mag-0.95": partial(
        iterative_ste_magnitude_pruning, target_sparsity=0.95
    ),
}
