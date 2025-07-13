import numpy as np

# TODO: Determine if we need to quantize the sum/mean operation in here...
# If so, we might have to slightly retool this as a higher order function which returns the masked_meanpool w/ proper dot_operators...
def masked_meanpool(x, lengths):
    """
    Helper function to perform mean pooling across the sequence length
    when sequences have variable lengths. We only want to pool across
    the prepadded sequence length.
    Args:
         x (float32): input sequence (L, d_model)
         lengths (int32):   the original length of the sequence before padding
    Returns:
        mean pooled output sequence (float32): (d_model)
    """
    L = x.shape[0]
    mask = np.arange(L) < lengths
    return np.sum(mask[..., None] * x, axis=0) / lengths
