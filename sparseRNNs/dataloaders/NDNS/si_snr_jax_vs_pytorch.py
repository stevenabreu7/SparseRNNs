# This standalone script compares the SI-SNR implementation in JAX vs PyTorch.
# The PyTorch SI-SNR implementation is taken from https://github.com/IntelLabs/IntelNeuromorphicDNSChallenge/blob/main/snr.py.

import jax.numpy as jnp
import torch


def si_snr(target, estimate) -> torch.tensor:
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

    if not torch.is_tensor(target):
        target: torch.tensor = torch.tensor(target)
    if not torch.is_tensor(estimate):
        estimate: torch.tensor = torch.tensor(estimate)

    # zero mean to ensure scale invariance
    s_target = target - torch.mean(target, dim=-1, keepdim=True)
    s_estimate = estimate - torch.mean(estimate, dim=-1, keepdim=True)

    # <s, s'> / ||s||**2 * s
    pair_wise_dot = torch.sum(s_target * s_estimate, dim=-1, keepdim=True)
    s_target_norm = torch.sum(s_target**2, dim=-1, keepdim=True)
    pair_wise_proj = pair_wise_dot * s_target / s_target_norm

    e_noise = s_estimate - pair_wise_proj

    pair_wise_sdr = torch.sum(pair_wise_proj**2, dim=-1) / (
        torch.sum(e_noise**2, dim=-1) + EPS
    )
    return 10 * torch.log10(pair_wise_sdr + EPS)


def si_snr_jax(target, estimate) -> torch.tensor:
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
    s_target = target - jnp.mean(target, axis=-1, keepdims=True)
    s_estimate = estimate - jnp.mean(estimate, axis=-1, keepdims=True)

    # <s, s'> / ||s||**2 * s
    pair_wise_dot = jnp.sum(s_target * s_estimate, axis=-1, keepdims=True)
    s_target_norm = jnp.sum(s_target**2, axis=-1, keepdims=True)
    pair_wise_proj = pair_wise_dot * s_target / s_target_norm

    e_noise = s_estimate - pair_wise_proj

    pair_wise_sdr = jnp.sum(pair_wise_proj**2, axis=-1) / (
        jnp.sum(e_noise**2, axis=-1) + EPS
    )
    return 10 * jnp.log10(pair_wise_sdr + EPS)


if __name__ == "__main__":

    torch.manual_seed(0)
    clean = torch.randn(3, 16000)
    noisy = torch.randn(3, 16000)

    import jax.numpy as jnp

    clean_jax = jnp.array(clean.numpy())
    noisy_jax = jnp.array(noisy.numpy())

    print(f"{si_snr(clean, noisy)}")
    print(f"{si_snr_jax(clean_jax, noisy_jax)}")
    print("done")
