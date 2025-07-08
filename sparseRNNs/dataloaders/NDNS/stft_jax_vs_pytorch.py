# This script compares the STFT and inverse STFT implementations in JAX and PyTorch.
# Importantly, noverlap in JAX is different than hop_length in PyTorch: noverlap = nfft - hop_length.

import jax
import jax.numpy as jnp
import torch
from jax.scipy.signal import istft

# Define parameters
batch_size = 1  # Number of sequences
sequence_length = 500  # Length of each sequence
low = 0  # Lower bound of random numbers (inclusive)
high = 100  # Upper bound of random numbers (exclusive)

# Set random seed for reproducibility
torch.manual_seed(0)

# Generate batch of sequences
random_sequences = high * torch.rand((batch_size, sequence_length))

random_sequences_jax = jnp.array(random_sequences.numpy())

nfft = 12
hop_length = 5


def stft_splitter(audio, n_fft=512):
    with torch.no_grad():
        audio_stft = torch.stft(audio, n_fft=n_fft, onesided=True, return_complex=True)
        return audio_stft.abs(), audio_stft.angle()


def stft_mixer(stft_abs, stft_angle, n_fft=512):
    return torch.istft(
        torch.complex(
            stft_abs * torch.cos(stft_angle), stft_abs * torch.sin(stft_angle)
        ),
        n_fft=n_fft,
        onesided=True,
    )


# default is hop_length = n_fft / 4
torch_stft = torch.stft(
    random_sequences,
    hop_length=hop_length,
    win_length=nfft,
    n_fft=nfft,
    onesided=True,
    return_complex=True,
)

noverlap = nfft - hop_length

# Perform STFT using JAX
f, t, jax_stft = jax.scipy.signal.stft(
    random_sequences_jax,
    nperseg=nfft,
    nfft=nfft,
    noverlap=noverlap,
    window="boxcar",
    return_onesided=True,
)

jax_stft_mag = jnp.abs(jax_stft)
jax_stft_phase = jnp.angle(jax_stft)

# Perform inverse STFT using JAX
_, jax_reconstructed = istft(
    jax_stft,
    nperseg=nfft,
    nfft=nfft,
    window="boxcar",
    noverlap=noverlap,
    input_onesided=True,
)

_, jax_reconstructed_from_split = istft(
    jax_stft_mag * jnp.exp(1j * jax_stft_phase),
    nperseg=nfft,
    nfft=nfft,
    window="boxcar",
    noverlap=noverlap,
    input_onesided=True,
)

# Print the generated sequences
print(f"{random_sequences_jax=}")
print(f"{jax_reconstructed}")

original_seq_length = random_sequences_jax.shape[1]
jax_reconstructed_seq_length = jax_reconstructed.shape[1]

max_error = jnp.max(
    jnp.abs(random_sequences_jax - jax_reconstructed[:, :original_seq_length])
)
max_error_from_split = jnp.max(
    jnp.abs(
        random_sequences_jax - jax_reconstructed_from_split[:, :original_seq_length]
    )
)

print(f"{max_error=}")
print(f"{max_error_from_split=}")
print(f"{original_seq_length=}")
print(f"{jax_reconstructed_seq_length=}")
print(f"{jax_stft.shape=}")

# torch stft doesn't seem to match jax stft, but jax istft reproduces the original sequence at least

# Print the generated sequences
print(jax_stft)
