"""
Explanation for the FxpSigmoid implementation:
==============================================
- Create a lookup table with 2^n_exp entries (default: 8 entries)
- Sample the true sigmoid at evenly spaced points from 0 to 2^(x_exp + x_extra)
- Store quantized sigmoid values: round(sigmoid(x) * 2^y_exp) - 2^(y_exp-1)

For positive inputs, we perform linear interpolation between LUT entries:
1. Index calculation: ind = min(xx >> x_exp, n_sub_2) finds the LUT segment
2. Fractional part: mu = xx & (delta-1) gets the position within the segment
3. Linear interpolation: yy = ((delta - mu) * lut[ind] + mu * lut[ind+1]) >> x_exp
    (to blends between lut[ind] and lut[ind+1] based on mu)

For negative inputs:
- Use symmetry: sigmoid(-x) = 1 - sigmoid(x)
- For negative inputs, compute sigmoid(|x|) then apply: 1 - sigmoid(|x|)
- Final result: (1 << (y_exp-1)) + sign * lut_sigmoid_half(|xx|)
"""
from sparseRNNs.fxparray import (ComplexFxpArray, FxpArray, RoundingMode,
                                 fxp_add, fxp_complex_add, fxp_complex_mul,
                                 fxp_from_fp, fxp_log_softmax, fxp_matmul,
                                 fxp_mean, fxp_mul, fxp_sub)

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class FxpSigmoid:
    """Fixed-point approximation for the sigmoid function using a lookup table."""

    def __init__(
        self, x_exp: int = 6, y_exp: int = 8, x_extra: int = 3, n_exp: int = 3
    ):
        """
        Initialize the fixed-point sigmoid approximation and build the lookup table.

        Parameters:
        - x_exp (int): Number of fractional bits for input. Defaults to 6.
        - y_exp (int): Number of fractional bits for output. Defaults to 8.
        - x_extra (int): Extra bits to refine LUT sampling resolution. Defaults to 3.
        - n_exp (int): Exponent that controls LUT size. Defaults to 3.
        """
        self.x_exp = x_exp
        self.y_exp = y_exp
        self.x_extra = x_extra
        self.n_exp = n_exp
        x = np.linspace(0, 1 << self.x_exp + self.x_extra, (1 << self.n_exp) + 1)[
            :-1
        ].astype(int)
        self.lut = (
            np.round(sigmoid(x / (1 << self.x_exp)) * (1 << self.y_exp))
            - (1 << self.y_exp - 1)
        ).astype(int)

    def lut_sigmoid_half(self, xx):
        """
        Computes the half-range (positive side) sigmoid approximation using the LUT.

        Parameters:
        xx (int or np.ndarray): Non-negative scaled input in fixed-point format.

        Returns:
        int or np.ndarray: Sigmoid approximation in fixed-point format.
        """
        # Takes values in [0, 960]. xx has exponent 6, so xx=(value*64).
        # constants
        delta = 1 << self.x_exp  # default: 64
        delta_sub_1 = delta - 1  # default: 63
        n_sub_2 = (1 << self.n_exp) - 2  # default: 8-2 = 6

        xx_div8 = xx >> self.x_exp  # default: xx / 8
        ind = np.minimum(xx_div8, n_sub_2)
        mu = np.bitwise_and(xx, delta_sub_1)
        yy = ((delta - mu) * self.lut[ind] >> self.x_exp) + (
            mu * self.lut[ind + 1] >> self.x_exp
        )
        return yy

    def apply(self, x, output_fxp: bool = True):
        """
        Applies the fixed-point sigmoid approximation to x.

        Parameters:
        - x (float, int, np.ndarray, FxpArray, ComplexFxpArray): Input value(s), can be fp or fxp.
        - output_fxp (bool): If True, returns the result in fxp format, otherwise in fp.

        Returns:
        (int, np.ndarray, float, FxpArray, ComplexFxpArray): fxp output if output_fxp=True, else fp output.
        """
        if isinstance(x, (FxpArray, ComplexFxpArray)):
            xx = x.change_exp(self.x_exp, warn_on_clip=False).data
        else:
            isfloat = isinstance(x, float) or (
                hasattr(x, "dtype") and np.issubdtype(x.dtype, np.floating)
            )
            xx = (x * (1 << self.x_exp)).astype(int) if isfloat else x
        sign = 2 * (xx > 0) - 1
        yy = (1 << (self.y_exp - 1)) + sign * self.lut_sigmoid_half(np.abs(xx))
        output = yy if output_fxp else yy / (1 << self.y_exp)
        if isinstance(x, (FxpArray, ComplexFxpArray)):
            output = FxpArray(data=output, bits=x.bits, exp=self.y_exp, signed=True)
        return output

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    x = np.linspace(-10, 10, 2000)
    fxpsigmoid = FxpSigmoid()
    yapprox = fxpsigmoid.apply(x, output_fxp=False)
    ytrue = sigmoid(x)
    
    # Calculate LUT sampling points
    lut_x_points = np.linspace(0, 1 << fxpsigmoid.x_exp + fxpsigmoid.x_extra, 
                               (1 << fxpsigmoid.n_exp) + 1)[:-1]
    lut_x_scaled = lut_x_points / (1 << fxpsigmoid.x_exp)
    lut_x_both_sides = np.concatenate([-lut_x_scaled[::-1], lut_x_scaled])
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(x, ytrue, 'b-', label='True Sigmoid', linewidth=2)
    plt.plot(x, yapprox, 'r-', label='FXP Sigmoid', linewidth=2)
    for lut_x in lut_x_both_sides:
        if -10 <= lut_x <= 10:
            plt.axvline(x=lut_x, color='gray', linestyle=':', alpha=0.7, linewidth=1)
    plt.xlabel('x')
    plt.ylabel('sigmoid(x)')
    plt.title('Sigmoid Comparison (gray lines: LUT points)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    error = np.abs(ytrue - yapprox)
    plt.plot(x, error, 'g-', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('Absolute Error')
    plt.title('Absolute Error')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.subplot(2, 2, 3)
    zoom_mask = (x >= -5) & (x <= 5)
    plt.plot(x[zoom_mask], ytrue[zoom_mask], 'b-', label='True Sigmoid', linewidth=2)
    plt.plot(x[zoom_mask], yapprox[zoom_mask], 'r-', label='FXP Sigmoid', linewidth=2)
    for lut_x in lut_x_both_sides:
        if -5 <= lut_x <= 5:
            plt.axvline(x=lut_x, color='gray', linestyle=':', alpha=0.7, linewidth=1)
    plt.xlabel('x')
    plt.ylabel('sigmoid(x)')
    plt.title('Sigmoid Comparison (Zoomed, gray lines: LUT points)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.plot(x[zoom_mask], error[zoom_mask], 'g-', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('Absolute Error')
    plt.title('Absolute Error (Zoomed)')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('sigmoid_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Maximum absolute error: {np.max(error):.6f}")
    print(f"Mean absolute error: {np.mean(error):.6f}")
    print(f"RMS error: {np.sqrt(np.mean(error**2)):.6f}")