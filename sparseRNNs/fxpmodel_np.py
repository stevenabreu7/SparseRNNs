import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import partial
from typing import Optional, Union

import jax
# import jax.numpy as np
import numpy as np
from jaxtyping import Array, DTypeLike, PyTree

from sparseRNNs.fxparray_np import (ComplexFxpArray, FxpArray, RoundingMode,
                                 fxp_add, fxp_complex_add, fxp_complex_mul,
                                 fxp_from_fp, fxp_log_softmax, fxp_matmul,
                                 fxp_mean, fxp_mul, fxp_sub)
from sparseRNNs.model.seq_model import masked_meanpool
from sparseRNNs.model.ssm import discretize_bilinear, discretize_zoh
from sparseRNNs.utils.logging import logger
from sparseRNNs.utils.quantization import QuantizationConfig

Dtype = DTypeLike

sys.path.append("$HOME/ncl-dl-stack")

GLU_VARIANTS = ["full", "half1", "half2", "none"]

def relu(x: Array) -> Array:
    return np.maximum(x, 0)

def fxp_relu(
    x: Union[FxpArray, ComplexFxpArray],
) -> Union[FxpArray, ComplexFxpArray]:
    if isinstance(x, ComplexFxpArray):
        new_data = relu(x.real.data + 1j * x.imag.data)
        relu_x = ComplexFxpArray(
            real=FxpArray(
                data=new_data.real.astype(np.int32),
                bits=x.real.bits,
                exp=x.real.exp,
                signed=x.real.signed,  # could also be set to False
            ),
            imag=FxpArray(
                data=new_data.imag.astype(np.int32),
                bits=x.imag.bits,
                exp=x.imag.exp,
                signed=x.imag.signed,  # could also be set to False
            ),
        )
        assert relu_x.real.bits == x.real.bits
        assert relu_x.imag.bits == x.imag.bits
        assert relu_x.real.exp == x.real.exp
        assert relu_x.imag.exp == x.imag.exp
        assert relu_x.real.signed == x.real.signed
        assert relu_x.imag.signed == x.imag.signed
        return relu_x
    elif isinstance(x, FxpArray):
        relu_x = FxpArray(
            data=np.maximum(x.data, 0),
            bits=x.bits,
            exp=x.exp,
            signed=x.signed,  # could also be set to False
        )
        assert relu_x.bits == x.bits
        assert relu_x.exp == x.exp
        assert relu_x.signed == x.signed
        return relu_x


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


def make_ssm_step_fn(Atre_exp, Atim_exp, Butre_exp, Butim_exp, xtre_exp, xtim_exp):
    def step(carry, inputs):
        At, But = inputs
        Butreal, Butimag = np.unstack(But, axis=-1)
        Atreal, Atimag = np.unstack(At, axis=-1)
        xprev = carry
        xprev_re, xprev_im = np.unstack(xprev, axis=-1)

        Axt_re = ((Atreal * xprev_re) >> Atre_exp) - ((Atimag * xprev_im) >> Atre_exp)
        Axt_im = ((Atreal * xprev_im) >> Atim_exp) + ((Atimag * xprev_re) >> Atim_exp)

        Butre = (
            (Butreal >> (Butre_exp - xtre_exp))
            if Butre_exp > xtre_exp
            else (Butreal << (xtre_exp - Butre_exp))
        )
        Butim = (
            (Butimag >> (Butim_exp - xtim_exp))
            if Butim_exp > xtim_exp
            else (Butimag << (xtim_exp - Butim_exp))
        )

        xt = jax.numpy.stack([Axt_re + Butre, Axt_im + Butim], axis=-1)
        return xt, xt

    return step


def recurrent_loop(
    Bu_elements_real: Array,
    Bu_elements_imag: Array,
    Lambda_bar_real: Array,
    Lambda_bar_imag: Array,
    xt_real: Array,
    xt_imag: Array,
    Bu_elements_real_exp: int,
    Bu_elements_imag_exp: int,
    Lambda_bar_real_exp: int,
    Lambda_bar_imag_exp: int,
    xt_real_exp: int,
    xt_imag_exp: int,
):
    ones_shape = (Bu_elements_real.shape[-2], Lambda_bar_real.shape[0])
    Lambda_elements = jax.numpy.stack(
        [
            Lambda_bar_real * np.ones(ones_shape, dtype=np.int32),
            Lambda_bar_imag * np.ones(ones_shape, dtype=np.int32),
        ],
        axis=-1,
    )
    Bu_elements_scan = jax.numpy.stack([Bu_elements_real, Bu_elements_imag], axis=-1)
    step = make_ssm_step_fn(
        Atre_exp=Lambda_bar_real_exp,
        Atim_exp=Lambda_bar_imag_exp,
        Butre_exp=Bu_elements_real_exp,
        Butim_exp=Bu_elements_imag_exp,
        xtre_exp=xt_real_exp,
        xtim_exp=xt_imag_exp,
    )
    xt_scan = jax.numpy.stack([xt_real, xt_imag], axis=-1)
    print(type(xt_scan))
    print(type(Lambda_elements))
    print(type(Bu_elements_scan))
    breakpoint()
    _, xs_raw = jax.lax.scan(step, xt_scan, (Lambda_elements, Bu_elements_scan))
    return xs_raw


@dataclass(kw_only=True)
class FxpS5Config:
    # ssm args
    H: int
    P: int
    discretization: str
    conj_sym: bool = True
    bidirectional: bool = False
    relufication: bool = True
    associative_scan: bool = False
    q_config: QuantizationConfig = QuantizationConfig.none()
    # model args
    n_layers: int
    d_model: int
    batchnorm: bool = True
    prenorm: bool = False
    bn_momentum: float = 0.9
    glu_variant: str = "none"
    step_rescale: float = 1.0
    relufication: bool = False
    fuse_batchnorm_linear: bool = False
    dropout: float = 0.2  # n/a in inference-only
    training: bool = True  # n/a in inference-only
    # args for regression model
    d_output: int = None
    padded: bool = False
    # mode: str = "pool"

    def __post_init__(self):
        assert self.discretization in [
            "zoh",
            "foh",
        ], f"Invalid discretization: {self.discretization}"
        assert isinstance(
            self.q_config, QuantizationConfig
        ), f"Invalid q_config: {self.q_config}"
        assert (
            self.glu_variant in GLU_VARIANTS
        ), f"Invalid GLU variant: {self.glu_variant}"
        # assert self.mode in ["pool", "last"], f"Invalid mode: {self.mode}"

    def to_dict(self):
        return {
            key: getattr(self, key).to_dict() if key == "q_config" else value
            for key, value in asdict(self).items()
        }


@dataclass
class FxpModule:
    modeldict: PyTree
    fxp_qconfig: PyTree
    scope: str
    store_intermediates: bool

    def setup(self):
        pass

    def forward(self, *args, **kwargs):
        pass

    def sow(self, top_key: str, key: str, value: FxpArray):
        if self.store_intermediates:
            assert top_key == "intermediates", f"Invalid top_key for sow: {top_key}"
            self.intermediates[key].append(value.copy())

    def __post_init__(self):
        self.intermediates = defaultdict(list)
        self.setup()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def last_intermediates(self):
        return {k: v[-1] if len(v) > 0 else None for k, v in self.intermediates.items()}

    def export(self) -> PyTree:
        pass


@dataclass
class FxpDense(FxpModule):
    dtype: Dtype = np.int32
    # see setup()
    weight: Optional[FxpArray] = None
    bias: Optional[FxpArray] = None

    def setup(self):
        # set the fxp config from the qconfig
        self.weight_exp = self.fxp_qconfig["w_exp"]
        self.weight_bits = self.fxp_qconfig["w_bits"]
        self.weight_signed = True
        self.bias_exp = self.fxp_qconfig["b_exp"]
        self.bias_bits = self.fxp_qconfig["b_bits"]
        self.bias_signed = True
        self.inp_bits = self.fxp_qconfig["inp_bits"]
        self.inp_exp = self.fxp_qconfig["inp_exp"]
        self.out_bits = self.fxp_qconfig["out_bits"]
        self.out_exp = self.fxp_qconfig["out_exp"]

        weight = self.modeldict["kernel"]
        bias = self.modeldict.get("bias", None)
        self.weight = fxp_from_fp(
            weight,
            bits=self.weight_bits,
            exp=self.weight_exp,
            signed=self.weight_signed,
            round_mode=RoundingMode.ROUND,
        )
        if bias is None:
            self.bias = None
        else:
            self.bias = fxp_from_fp(
                bias,
                bits=self.bias_bits,
                exp=self.bias_exp,
                signed=self.bias_signed,
                round_mode=RoundingMode.ROUND,
            )

    def forward(self, x: FxpArray) -> FxpArray:
        """Forward pass of the Dense layer with fxp arithmetic.
        x: shape (L, H) or (B, L, H)
        """
        if (
            (x.bits > self.inp_bits)
            or (x.exp > self.inp_exp)
            or (x.signed and not self.weight_signed)
        ):
            logger.debug(
                f"FxpDense input with bits={x.bits}, exp={x.exp}," f" signed={x.signed}"
            )
            x = x.change_cfg(
                new_bits=self.inp_bits,
                new_exp=self.inp_exp,
                new_signed=True,
            )
            logger.debug(
                f"FxpDense input ---> bits={x.bits}, exp={x.exp}," f" signed={x.signed}"
            )

        wx = fxp_matmul(
            x,
            self.weight,
            result_bits=self.out_bits,
            result_exp=self.out_exp,
        )
        if self.bias is not None:
            wx = fxp_add(
                wx,
                self.bias,
                result_bits=self.out_bits,
                result_exp=self.out_exp,
            )
        self.sow("intermediates", "__call__", wx)
        return wx

    def export(self) -> PyTree:
        assert self.weight.exp == self.weight_exp
        assert self.weight.bits == self.weight_bits
        assert self.weight.signed == self.weight_signed
        assert self.bias.exp == self.bias_exp
        assert self.bias.bits == self.bias_bits
        assert self.bias.signed == self.bias_signed
        return dict(
            params=dict(
                weight=self.weight.data,
                bias=self.bias.data,
            ),
            qconfig=dict(
                weight_exp=self.weight_exp,
                weight_bits=self.weight_bits,
                weight_signed=self.weight_signed,
                bias_exp=self.bias_exp,
                bias_bits=self.bias_bits,
                bias_signed=self.bias_signed,
                inp_bits=self.inp_bits,
                inp_exp=self.inp_exp,
                out_bits=self.out_bits,
                out_exp=self.out_exp,
            ),
            intermediates=self.last_intermediates(),
        )


@dataclass
class FxpSSM(FxpModule):
    H: int
    P: int
    discretization: str
    conj_sym: bool = True
    q_config: QuantizationConfig = QuantizationConfig.none()
    step_rescale: float = 1.0
    # unsupported args
    bidirectional: bool = False
    relufication: bool = True
    associative_scan: bool = False
    ## init args, not used here
    # Lambda_re_init: jax.Array
    # Lambda_im_init: jax.Array
    # V: jax.Array
    # Vinv: jax.Array
    # C_init: str
    # dt_min: float
    # dt_max: float
    clip_eigs: bool = False
    # fused batch norm
    bn_mean: Optional[Array] = None
    bn_var: Optional[Array] = None
    bn_scale: Optional[Array] = None
    bn_bias: Optional[Array] = None
    bn_eps: float = 1e-5
    use_lax_scan: bool = True
    compute_fp32: bool = False

    def setup(
        self,
    ):
        assert self.relufication, "Only relufication=True is supported for now"
        assert (
            not self.associative_scan
        ), "Only associative_scan=False is supported for now"
        assert not self.bidirectional, "Only bidirectional=False is supported for now"
        assert not self.clip_eigs, "Only clip_eigs=False is supported for now"

        # discretize A, B
        B_tilde = self.modeldict["B"][..., 0] + 1j * self.modeldict["B"][..., 1]
        Lambda_re = self.modeldict["Lambda_re"]
        Lambda_im = self.modeldict["Lambda_im"]
        Lambda = Lambda_re + 1j * Lambda_im
        # Lambda = np.clip(Lambda_re, None, -1e-4) + 1j * Lambda_im  # TODO: is this needed?
        log_step = self.modeldict["log_step"]
        step = self.step_rescale * np.exp(log_step[:, 0])
        if self.discretization in ["zoh"]:
            self.Lambda_bar, self.B_bar = discretize_zoh(Lambda, B_tilde, step)
        elif self.discretization in ["bilinear"]:
            self.Lambda_bar, self.B_bar = discretize_bilinear(Lambda, B_tilde, step)
        else:
            raise NotImplementedError(f"Discretization method {self.discretization}")
        self.C_tilde = self.modeldict["C"][..., 0] + 1j * self.modeldict["C"][..., 1]
        self.D = self.modeldict["D"]

        # fused batch norm if possible
        bn_mean_var_none = self.bn_mean is None or self.bn_var is None
        bn_scale_bias_none = self.bn_scale is None or self.bn_bias is None
        if not bn_mean_var_none:
            if not bn_scale_bias_none:
                # scale and bias are present
                if np.isnan(self.bn_scale).any():
                    logger.warning("bn_scale has NaNs, replacing with 1.0")
                    self.bn_scale = np.where(
                        np.isnan(self.bn_scale), 1.0, self.bn_scale
                    )
                scale = self.bn_scale / np.sqrt(self.bn_var + self.bn_eps)
                bias = self.bn_bias - self.bn_mean * scale
            else:
                # scale and bias are not present
                scale = 1.0 / np.sqrt(self.bn_var + self.bn_eps)
                bias = 0.0 - self.bn_mean * scale

            B_bar = self.B_bar * scale
            B_bias = self.B_bar @ bias
            D = self.D * scale
            D_bias = self.D * bias

            # re-compute the exponents for B
            B_real_intbits = int(
                max(0, np.ceil(np.log2(np.abs(B_bar.real).max())).item())
            )
            B_imag_intbits = int(
                max(0, np.ceil(np.log2(np.abs(B_bar.imag).max())).item())
            )
            D_intbits = int(max(0, np.ceil(np.log2(np.abs(D).max())).item()))
            B_real_max_exp = (
                self.fxp_qconfig["weights"]["B_re"]["bits"] - 1 - B_real_intbits
            )
            B_imag_max_exp = (
                self.fxp_qconfig["weights"]["B_im"]["bits"] - 1 - B_imag_intbits
            )
            D_max_exp = self.fxp_qconfig["weights"]["D"]["bits"] - 1 - D_intbits
            B_real_exp = self.fxp_qconfig["weights"]["B_re"]["exp"]
            B_imag_exp = self.fxp_qconfig["weights"]["B_im"]["exp"]
            D_exp = self.fxp_qconfig["weights"]["D"]["exp"]
            logger.debug(
                f"fusing SSM with exp: B_re={B_real_exp}, B_im={B_imag_exp},"
                f" D={D_exp}"
            )
            logger.debug(
                f"            max exp: B_re={B_real_max_exp},"
                f" B_im={B_imag_max_exp}, D={D_max_exp}"
            )

            if D_max_exp < 0 or B_real_max_exp < 0 or B_imag_max_exp < 0:
                err_str = (
                    f"D_exp={D_max_exp}, Bre_exp={B_real_max_exp},"
                    f" Bim_exp={B_imag_max_exp}"
                )
                logger.error(
                    "Fused B/D weights do not fit the given precision:" f" {err_str}"
                )

            if B_real_exp > B_real_max_exp:
                logger.warning(
                    f"Truncating {self.scope}.B_re.exp to"
                    f" {B_real_max_exp} (was {B_real_exp})"
                )
                B_real_exp = B_real_max_exp
            if B_imag_exp > B_imag_max_exp:
                logger.warning(
                    f"Truncating {self.scope}.B_im.exp to"
                    f" {B_imag_max_exp} (was {B_imag_exp})"
                )
                B_imag_exp = B_imag_max_exp
            if D_exp > D_max_exp:
                logger.warning(
                    f"Truncating {self.scope}.D.exp to {D_max_exp} (was" f" {D_exp})"
                )
                D_exp = D_max_exp
        else:
            B_bias = None
            D_bias = None
            B_bar = self.B_bar
            D = self.D
            B_real_exp = self.fxp_qconfig["weights"]["B_re"]["exp"]
            B_imag_exp = self.fxp_qconfig["weights"]["B_im"]["exp"]

        if B_bias is not None:
            self.B_bias = ComplexFxpArray(
                real=partial_fxp_from_fp(
                    B_bias.real,
                    bits=self.fxp_qconfig["activations"]["Bu_re"]["bits"],
                    exp=self.fxp_qconfig["weights"]["Bu_re"]["exp"],
                ),
                imag=partial_fxp_from_fp(
                    B_bias.imag,
                    bits=self.fxp_qconfig["activations"]["Bu_im"]["bits"],
                    exp=self.fxp_qconfig["weights"]["Bu_im"]["exp"],
                ),
            )
        else:
            self.B_bias = None

        if D_bias is not None:
            self.D_bias = partial_fxp_from_fp(
                D_bias,
                bits=self.fxp_qconfig["activations"]["y"]["bits"],
                exp=self.fxp_qconfig["weights"]["y"]["exp"],
            )
        else:
            self.D_bias = None

        # quantize into fxp
        partial_fxp_from_fp = partial(
            fxp_from_fp,
            signed=True,
            round_mode=RoundingMode.ROUND,
        )
        self.Lambda_bar = ComplexFxpArray(
            real=partial_fxp_from_fp(
                self.Lambda_bar.real,
                bits=self.fxp_qconfig["weights"]["A_re"]["bits"],
                exp=self.fxp_qconfig["weights"]["A_re"]["exp"],
            ),
            imag=partial_fxp_from_fp(
                self.Lambda_bar.imag,
                bits=self.fxp_qconfig["weights"]["A_im"]["bits"],
                exp=self.fxp_qconfig["weights"]["A_im"]["exp"],
            ),
        )
        self.B_bar = ComplexFxpArray(
            real=partial_fxp_from_fp(
                B_bar.real,
                bits=self.fxp_qconfig["weights"]["B_re"]["bits"],
                exp=self.fxp_qconfig["weights"]["B_re"]["exp"],
            ),
            imag=partial_fxp_from_fp(
                B_bar.imag,
                bits=self.fxp_qconfig["weights"]["B_im"]["bits"],
                exp=self.fxp_qconfig["weights"]["B_im"]["exp"],
            ),
        )
        self.C_tilde = ComplexFxpArray(
            real=partial_fxp_from_fp(
                self.C_tilde.real,
                bits=self.fxp_qconfig["weights"]["C_re"]["bits"],
                exp=self.fxp_qconfig["weights"]["C_re"]["exp"],
            ),
            imag=partial_fxp_from_fp(
                self.C_tilde.imag,
                bits=self.fxp_qconfig["weights"]["C_im"]["bits"],
                exp=self.fxp_qconfig["weights"]["C_im"]["exp"],
            ),
        )
        self.D = partial_fxp_from_fp(
            D,
            bits=self.fxp_qconfig["weights"]["D"]["bits"],
            exp=self.fxp_qconfig["weights"]["D"]["exp"],
        )

    def forward(
        self,
        input_sequence: FxpArray,  # L x H
    ) -> FxpArray:
        # make sure the input has the correct precision
        # TODO: we should make sure that this is not called
        logger.debug(
            f"{self.scope} input with bits={input_sequence.bits},"
            f" exp={input_sequence.exp}, signed={input_sequence.signed}"
        )
        input_sequence = input_sequence.change_cfg(
            new_bits=self.fxp_qconfig["activations"]["u"]["bits"],
            new_exp=self.fxp_qconfig["activations"]["u"]["exp"],
            new_signed=True,
        )
        logger.debug(
            f"{self.scope} input with bits={input_sequence.bits},"
            f" exp={input_sequence.exp}, signed={input_sequence.signed}"
        )

        # B @ input_sequence + B_bias? -> Bu_elements
        Bu_elements = ComplexFxpArray(
            real=fxp_matmul(
                input_sequence,
                self.B_bar.real.transpose(),
                result_exp=self.fxp_qconfig["activations"]["Bu_re"]["exp"],
                result_bits=self.fxp_qconfig["activations"]["Bu_re"]["bits"],
            ),
            imag=fxp_matmul(
                input_sequence,
                self.B_bar.imag.transpose(),
                result_exp=self.fxp_qconfig["activations"]["Bu_im"]["exp"],
                result_bits=self.fxp_qconfig["activations"]["Bu_im"]["bits"],
            ),
        )
        if self.B_bias is not None:
            Bu_elements = fxp_complex_add(
                Bu_elements,
                self.B_bias,
                result_exp=(Bu_elements.real.exp, Bu_elements.imag.exp),
                result_bits=(Bu_elements.real.bits, Bu_elements.imag.bits),
            )

        self.sow("intermediates", "Bu_elements", Bu_elements)
        if self.compute_fp32:
            Bu_elements_fp32 = input_sequence.to_float() @ self.B_bar.to_float().T
            self.sow("intermediates", "Bu_elements_fp32", Bu_elements_fp32)

        # recurrence
        # A @ xt + But -> xt
        xt_re_bits = self.fxp_qconfig["activations"]["x_re"]["bits"]
        xt_im_bits = self.fxp_qconfig["activations"]["x_im"]["bits"]
        xtshape = (
            (Bu_elements.shape[-1],)
            if Bu_elements.ndim == 2
            else (Bu_elements.shape[0], Bu_elements.shape[-1])
        )
        xt = ComplexFxpArray(
            real=FxpArray(
                data=np.zeros(xtshape, dtype=np.int32),
                bits=xt_re_bits,
                exp=self.fxp_qconfig["activations"]["x_re"]["exp"],
                signed=True,
            ),
            imag=FxpArray(
                data=np.zeros(xtshape, dtype=np.int32),
                bits=xt_im_bits,
                exp=self.fxp_qconfig["activations"]["x_im"]["exp"],
                signed=True,
            ),
        )
        logger.debug(f"Starting recurrent loop for {self.scope}")
        if self.use_lax_scan:
            assert Bu_elements.ndim < 3, "batching not supported"
            recurrent_loop_fn = recurrent_loop
            xs_raw = recurrent_loop_fn(
                Bu_elements.real.data,
                Bu_elements.imag.data,
                jax.numpy.asarray(self.Lambda_bar.real.data),
                jax.numpy.asarray(self.Lambda_bar.imag.data),
                jax.numpy.asarray(xt.real.data),
                jax.numpy.asarray(xt.imag.data),
                Bu_elements.real.exp,
                Bu_elements.imag.exp,
                self.Lambda_bar.real.exp,
                self.Lambda_bar.imag.exp,
                xt.real.exp,
                xt.imag.exp,
            )
            xs = xt
            xs.real.data = xs_raw[..., 0]
            xs.imag.data = xs_raw[..., 1]
            # breakpoint()
        else:
            xs_list = []
            xs_fp_list = []
            for t in range(input_sequence.data.shape[0]):
                if self.compute_fp32:
                    xtfp = (
                        self.Lambda_bar.to_float() * xt.to_float() + Bu_elements_fp32[t]
                    )
                Axt = fxp_complex_mul(
                    self.Lambda_bar,
                    xt,
                    result_exp=(xt.real.exp, xt.imag.exp),
                    result_bits=(xt_re_bits + 4, xt_im_bits + 4),
                )
                xt = fxp_complex_add(
                    Axt,
                    Bu_elements[t],
                    result_exp=(xt.real.exp, xt.imag.exp),
                    result_bits=(xt_re_bits, xt_im_bits),
                    warn_on_overflow=False,
                    warn_on_clip=False,
                )
                if self.compute_fp32:
                    xs_fp_list.append(xtfp)
                xs_list.append(xt)
            xs = xt
            xs.real.data = np.stack([x.real.data for x in xs_list], dtype=np.int32)
            xs.imag.data = np.stack([x.imag.data for x in xs_list], dtype=np.int32)
            if self.compute_fp32:
                xs_fp = np.stack(xs_fp_list)
                self.sow("intermediates", "xs_fp", xs_fp)
        self.sow("intermediates", "xs", xs)
        logger.debug(f"Completed recurrent loop for {self.scope}")

        # relu
        if self.relufication:
            xs = fxp_relu(xs)
            self.sow("intermediates", "xs_relu", xs)

        # output
        logger.debug(f"C projection in {self.scope}")
        ys = fxp_sub(
            fxp_matmul(
                xs.real,
                self.C_tilde.real.transpose(),
                result_exp=self.fxp_qconfig["activations"]["y"]["exp"],
                # result_exp="compute_best",
                result_bits=self.fxp_qconfig["activations"]["y"]["bits"],
            ),
            fxp_matmul(
                xs.imag,
                self.C_tilde.imag.transpose(),
                result_exp=self.fxp_qconfig["activations"]["y"]["exp"],
                # result_exp="compute_best",
                result_bits=self.fxp_qconfig["activations"]["y"]["bits"],
            ),
            result_exp=self.fxp_qconfig["activations"]["y"]["exp"],
            result_bits=self.fxp_qconfig["activations"]["y"]["bits"],
        )
        self.sow("intermediates", "Cxs", ys)
        if self.conj_sym:
            ys.data *= 2
            self.sow("intermediates", "2Cxs", ys)

        # Add feedthrough matrix output Du
        logger.debug(f"D application in {self.scope}")
        Du = fxp_mul(
            self.D,
            input_sequence,
            result_exp=self.fxp_qconfig["activations"]["y"]["exp"],
            result_bits=self.fxp_qconfig["activations"]["y"]["bits"],
        )
        self.sow("intermediates", "Du", Du)
        ysDu = fxp_add(
            ys,
            Du,
            result_exp=self.fxp_qconfig["activations"]["y"]["exp"],
            result_bits=self.fxp_qconfig["activations"]["y"]["bits"],
        )
        if self.D_bias is not None:
            self.sow("intermediates", "ys_pre_Dbias", ysDu)
            logger.debug(f"D_bias application in {self.scope}")
            ysDu = fxp_add(
                ysDu,
                self.D_bias,
                result_exp=self.fxp_qconfig["activations"]["y"]["exp"],
                result_bits=self.fxp_qconfig["activations"]["y"]["bits"],
            )
        self.sow("intermediates", "ys", ysDu)
        return ysDu, xs

    @staticmethod
    def init_fn(
        H: int,
        P: int,
        discretization: str,
        conj_sym: bool = True,
        q_config: QuantizationConfig = QuantizationConfig.none(),
        bidirectional: bool = False,
        relufication: bool = True,
        associative_scan: bool = False,
    ):
        return partial(
            FxpSSM,
            H=H,
            P=P,
            discretization=discretization,
            conj_sym=conj_sym,
            q_config=q_config,
            bidirectional=bidirectional,
            relufication=relufication,
            associative_scan=associative_scan,
        )

    def export(self):
        params = {}
        qconfig = {}
        d = dict(
            A_real=self.Lambda_bar.real,
            A_imag=self.Lambda_bar.imag,
            B_real=self.B_bar.real,
            B_imag=self.B_bar.imag,
            C_real=self.C_tilde.real,
            C_imag=self.C_tilde.imag,
            D=self.D,
            B_bias_real=None if self.B_bias is None else self.B_bias.real,
            B_bias_imag=None if self.B_bias is None else self.B_bias.imag,
            D_bias=self.D_bias,
        )
        for k, x in d.items():
            if x is not None:
                params[k] = x.data
                qconfig[f"{k}_bits"] = x.bits
                qconfig[f"{k}_exp"] = x.exp
                qconfig[f"{k}_signed"] = x.signed
        for k in ["u", "Bu_re", "Bu_im", "x_re", "x_im", "y"]:
            qconfig[f"{k}_bits"] = self.fxp_qconfig["activations"][k]["bits"]
            qconfig[f"{k}_exp"] = self.fxp_qconfig["activations"][k]["exp"]
        return dict(
            params=params,
            qconfig=qconfig,
            intermediates=self.last_intermediates(),
        )


@dataclass
class FxpBatchNorm(FxpModule):
    def setup(self, bn_eps: float = 1e-5):
        self.minus_mean = fxp_from_fp(
            -1 * self.modeldict["mean"],
            bits=self.fxp_qconfig["mean"]["bits"],
            exp=self.fxp_qconfig["mean"]["exp"],
            signed=True,
            round_mode=RoundingMode.ROUND,
        )
        self.invsq_var = fxp_from_fp(
            1.0 / np.sqrt(self.modeldict["var"] + bn_eps),
            bits=self.fxp_qconfig["invsq_var"]["bits"],
            exp=self.fxp_qconfig["invsq_var"]["exp"],
            signed=True,
            round_mode=RoundingMode.ROUND,
        )
        self.bias = (
            fxp_from_fp(
                self.modeldict["bias"],
                bits=self.fxp_qconfig["bias"]["bits"],
                exp=self.fxp_qconfig["bias"]["exp"],
                signed=True,
                round_mode=RoundingMode.ROUND,
            )
            if "bias" in self.modeldict
            else None
        )
        self.scale = (
            fxp_from_fp(
                self.modeldict["scale"],
                bits=self.fxp_qconfig["scale"]["bits"],
                exp=self.fxp_qconfig["scale"]["exp"],
                signed=True,
                round_mode=RoundingMode.ROUND,
            )
            if "scale" in self.modeldict
            else None
        )

    def forward(self, x):
        self.sow("intermediates", "norm_input", x)
        x = fxp_add(
            x,
            self.minus_mean,
            # TODO: set result precision?
            result_exp="compute_best",
        )
        self.sow("intermediates", "norm_input_minus_mean", x)
        logger.debug(
            f"{self.scope}: x - mean: bits={x.bits}, exp={x.exp}," f" signed={x.signed}"
        )
        x = fxp_mul(
            x,
            self.invsq_var,
            # TODO: set result precision?
            result_exp="compute_best",
        )
        self.sow("intermediates", "norm_output_raw", x)
        logger.debug(
            f"{self.scope}: (x - mean) / sqrt(var): bits={x.bits},"
            f" exp={x.exp}, signed={x.signed}"
        )
        # x = (x - self.mean) / np.sqrt(self.var + self.bn_eps)
        if self.scale is not None:
            x = fxp_mul(
                x,
                self.scale,
                # TODO: set result precision?
                result_exp="compute_best",
            )
            self.sow("intermediates", "norm_output_scaled", x)
            logger.debug(
                f"{self.scope}: (x - mean) / sqrt(var) * scale: bits={x.bits},"
                f" exp={x.exp}, signed={x.signed}"
            )
            # x = x * self.scale
        if self.bias is not None:
            x = fxp_add(
                x,
                self.bias,
                # TODO: set result precision?
                result_exp="compute_best",
            )
            self.sow("intermediates", "norm_output_scaled_bias", x)
            logger.debug(
                f"{self.scope}: (x - mean) / sqrt(var) * scale + bias:"
                f" bits={x.bits}, exp={x.exp}, signed={x.signed}"
            )
            # x = x + self.bias
        self.sow("intermediates", "norm_output", x)
        logger.debug(
            f"{self.scope} result: bits={x.bits}, exp={x.exp}," f" signed={x.signed}"
        )
        return x

    def export(self):
        data = dict(
            params=dict(
                mean=-1 * self.minus_mean.data,
                invsq_var=self.invsq_var.data,
            ),
            qconfig=dict(
                mean_bits=self.minus_mean.bits,
                mean_exp=self.minus_mean.exp,
                invsq_var_bits=self.invsq_var.bits,
                invsq_var_exp=self.invsq_var.exp,
            ),
            intermediates=self.last_intermediates(),
        )
        if self.bias is not None:
            data["params"]["bias"] = self.bias.data
            data["qconfig"]["bias_bits"] = self.bias.bits
            data["qconfig"]["bias_exp"] = self.bias.exp
        if self.scale is not None:
            data["params"]["scale"] = self.scale.data
            data["qconfig"]["scale_bits"] = self.scale.bits
            data["qconfig"]["scale_exp"] = self.scale.exp
        return data


@dataclass
class FxpSequenceLayer(FxpModule):
    mixer_cls: FxpModule
    d_model: int
    batchnorm: bool = True
    prenorm: bool = True
    glu_variant: str = "none"
    bn_momentum: float = 0.90
    step_rescale: float = 1.0
    relufication: bool = False
    fuse_batchnorm_linear: bool = False
    q_config: QuantizationConfig = None
    dropout: float = 0.2  # NOTE: n/a in inference-only
    training: bool = True  # NOTE: n/a in inference-only
    layer_idx: Optional[int] = None

    def setup(self):
        keys = [e for e in self.fxp_qconfig.keys() if e.startswith("layers_")]
        if len(keys) > 0:
            print(f"not using shared exponents in layer {self.layer_idx}")
            self.fxp_qconfig = self.fxp_qconfig[f"layers_{self.layer_idx}"]
        else:
            print(f"using shared exponents in layer {self.layer_idx}")

        assert self.batchnorm, "Only batchnorm is supported for now."
        assert self.dropout == 0.0, "Only dropout=0.0 is supported for now."
        assert not self.training, "Only training=False is supported for now."
        assert self.relufication, "Only relufication=True is supported for now."
        assert self.prenorm, "Only prenorm=True is supported for now."
        if self.fuse_batchnorm_linear:
            assert (
                self.batchnorm
            ), "fuse_batchnorm_linear can only be used with batchnorm"
            assert self.prenorm, "fuse_batchnorm_linear can only be used with prenorm"

        if not self.fuse_batchnorm_linear:
            self.norm = FxpBatchNorm(
                modeldict=self.modeldict["norm"],
                fxp_qconfig=self.fxp_qconfig["norm"],
                scope=f"{self.scope}.norm",
                store_intermediates=self.store_intermediates,
            )

        self.mixer = self.mixer_cls(
            modeldict=(
                self.modeldict["seq"]
                if "seq" in self.modeldict
                else self.modeldict["mixer"]
            ),
            fxp_qconfig=self.fxp_qconfig["ssm"],
            scope=f"{self.scope}.mixer",
            store_intermediates=self.store_intermediates,
            step_rescale=self.step_rescale,
            bn_mean=(
                self.modeldict["norm"]["mean"] if self.fuse_batchnorm_linear else None
            ),
            bn_var=(
                self.modeldict["norm"]["var"] if self.fuse_batchnorm_linear else None
            ),
            bn_eps=1e-5,  # NOTE: default value
            bn_scale=(
                self.modeldict["norm"].get("scale")
                if self.fuse_batchnorm_linear
                else None
            ),
            bn_bias=(
                self.modeldict["norm"].get("bias")
                if self.fuse_batchnorm_linear
                else None
            ),
        )
        dense_cls = FxpDense

        assert (
            self.glu_variant in GLU_VARIANTS
        ), f"GLU variant must be one of {GLU_VARIANTS}"
        if self.glu_variant == "full":
            self.out1 = dense_cls(
                modeldict=self.modeldict["out1"],
                fxp_qconfig=self.fxp_qconfig["out1"],
                scope=f"{self.scope}.out1",
                store_intermediates=self.store_intermediates,
            )
            self.out2 = dense_cls(
                modeldict=self.modeldict["out2"],
                fxp_qconfig=self.fxp_qconfig["out2"],
                scope=f"{self.scope}.out2",
                store_intermediates=self.store_intermediates,
            )
        elif self.glu_variant in ["half1", "half2"]:
            self.out2 = dense_cls(
                modeldict=self.modeldict["out2"],
                fxp_qconfig=self.fxp_qconfig["out2"],
                scope=f"{self.scope}.out2",
                store_intermediates=self.store_intermediates,
            )

        self.glu_act_fn = fxp_relu

        # TODO: this is never called in the forward pass!?
        # self.out = dense_cls(modeldict=self.modeldict["out"])

        self.drop = lambda x: x

        def mult_gate(x: FxpArray, y: FxpArray):
            x = x.change_cfg(
                new_bits=self.fxp_qconfig["multgate"]["l_bits"],
                new_exp=self.fxp_qconfig["multgate"]["l_exp"],
                new_signed=True,
            )
            y = y.change_cfg(
                new_bits=self.fxp_qconfig["multgate"]["r_bits"],
                new_exp=self.fxp_qconfig["multgate"]["r_exp"],
                new_signed=True,
            )
            return fxp_mul(
                x,
                y,
                result_exp=self.fxp_qconfig["multgate"]["res_exp"],
                result_bits=self.fxp_qconfig["multgate"]["res_bits"],
                round_mode=RoundingMode.FLOOR,
                warn_on_overflow=True,
            )

        self.mult_gate = mult_gate

        sigmoid_x_exp = self.fxp_qconfig["out2"]["out_exp"]
        sigmoid_y_exp = self.fxp_qconfig["out2"]["out_bits"] - 2
        if sigmoid_x_exp > 6:
            logger.warning(
                f"truncate sigmoid x_exp: {sigmoid_x_exp} -> 6" f" ({sigmoid_x_exp-6})"
            )
            sigmoid_x_exp = 6
        self.sigmoid_cls = FxpSigmoid(x_exp=sigmoid_x_exp, y_exp=sigmoid_y_exp)
        self.sigmoid = partial(
            self.sigmoid_cls.apply,
            output_fxp=True,
        )

    def forward(self, x: FxpArray) -> FxpArray:
        skip = x
        self.sow("intermediates", "ssm_input", x)

        if self.fuse_batchnorm_linear and self.batchnorm and self.prenorm:
            self.sow("intermediates", "pre_s5", x)
            x, x_pre_C = self.mixer(x)
        else:
            if self.prenorm:
                x = self.norm(x)
            self.sow("intermediates", "pre_s5", x)
            x, x_pre_C = self.mixer(x)

        self.sow("intermediates", "pre_C", x_pre_C)

        x1 = self.drop(self.glu_act_fn(x))
        self.sow("intermediates", "pre_GLU", x)

        if self.glu_variant == "full":
            rside = self.sigmoid(self.out2(x1))
            self.sow("intermediates", "out2_sigmoid", rside)
            x = self.mult_gate(self.out1(x1), rside)
            x = self.drop(x)
        elif self.glu_variant == "half1":
            rside = self.sigmoid(self.out2(x1))
            self.sow("intermediates", "out2_sigmoid", rside)
            x = self.mult_gate(x1, rside)
            x = self.drop(x)
        elif self.glu_variant == "half2":
            rside = self.sigmoid(self.out2(x1))
            self.sow("intermediates", "out2_sigmoid", rside)
            x = self.mult_gate(x, rside)
            x = self.drop(x)
        elif self.glu_variant == "none":
            x = x1
        self.sow("intermediates", "post_GLU", x)

        x = fxp_add(
            x,
            skip,
            result_exp="compute_best",
            result_bits=self.fxp_qconfig["multgate"]["res_bits"],
        )
        self.sow("intermediates", "residadd", x)

        if not self.prenorm:
            x = self.norm(x)

        if self.relufication:
            x = fxp_relu(x)
        self.sow("intermediates", "output", x)
        return x

    def export(self):
        norm_data = self.norm.export() if hasattr(self, "norm") else None
        mixer_data = self.mixer.export()
        out1_data = self.out1.export() if hasattr(self, "out1") else None
        out2_data = self.out2.export()
        multgate_qconfig = dict(
            l_bits=self.fxp_qconfig["multgate"]["l_bits"],
            l_exp=self.fxp_qconfig["multgate"]["l_exp"],
            r_bits=self.fxp_qconfig["multgate"]["r_bits"],
            r_exp=self.fxp_qconfig["multgate"]["r_exp"],
            res_bits=self.fxp_qconfig["multgate"]["res_bits"],
            res_exp=self.fxp_qconfig["multgate"]["res_exp"],
        )
        sigmoid_qconfig = dict(
            x_exp=self.sigmoid_cls.x_exp,
            y_exp=self.sigmoid_cls.y_exp,
            x_extra=self.sigmoid_cls.x_extra,
            n_exp=self.sigmoid_cls.n_exp,
        )
        data = dict(
            params=dict(
                mixer=mixer_data["params"],
                out2=out2_data["params"],
            ),
            qconfig=dict(
                mixer=mixer_data["qconfig"],
                out2=out2_data["qconfig"],
                multgate=multgate_qconfig,
                sigmoid=sigmoid_qconfig,
            ),
            intermediates=dict(
                mixer=mixer_data["intermediates"],
                out2=out2_data["intermediates"],
                **self.last_intermediates(),
            ),
        )
        if out1_data is not None:
            data["params"]["out1"] = out1_data["params"]
            data["qconfig"]["out1"] = out1_data["qconfig"]
            data["intermediates"]["out1"] = out1_data["intermediates"]
        if norm_data is not None:
            data["params"]["norm"] = norm_data["params"]
            data["qconfig"]["norm"] = norm_data["qconfig"]
            data["intermediates"]["norm"] = norm_data["intermediates"]
        return data


@dataclass
class FxpStackedEncoderModel(FxpModule):
    mixer_cls: FxpModule  # ssm init function
    n_layers: int
    d_model: int
    batchnorm: bool = True
    prenorm: bool = False
    bn_momentum: float = 0.9
    glu_variant: str = "none"
    step_rescale: float = 1.0
    relufication: bool = False
    fuse_batchnorm_linear: bool = False
    q_config: QuantizationConfig = QuantizationConfig.none()
    dropout: float = 0.2  # NOTE: n/a in inference-only
    training: bool = True  # NOTE: n/a in inference-only

    def setup(self):
        assert self.batchnorm, "Only batchnorm is supported for now."
        assert self.dropout == 0.0, "Only dropout=0.0 is supported for now."
        assert not self.training, "Only training=False is supported for now."
        assert self.relufication, "Only relufication=True is supported for now."

        self.encoder = FxpDense(
            modeldict=self.modeldict["encoder"],
            fxp_qconfig=self.fxp_qconfig["encoder"],
            scope=f"{self.scope}.encoder",
            store_intermediates=self.store_intermediates,
        )
        self.seq_layers = [
            FxpSequenceLayer(
                modeldict=self.modeldict[f"layers_{idx}"],
                fxp_qconfig=self.fxp_qconfig["blocks"],
                scope=f"{self.scope}.layers_{idx}",
                store_intermediates=self.store_intermediates,
                mixer_cls=self.mixer_cls,
                d_model=self.d_model,
                dropout=self.dropout,
                batchnorm=self.batchnorm,
                prenorm=self.prenorm,
                glu_variant=self.glu_variant,
                bn_momentum=self.bn_momentum,
                training=self.training,
                step_rescale=self.step_rescale,
                relufication=self.relufication,
                fuse_batchnorm_linear=self.fuse_batchnorm_linear,
                q_config=self.q_config,
                layer_idx=idx,
            )
            for idx in range(self.n_layers)
        ]

    def forward(self, x: FxpArray, integration_timesteps: int = None) -> FxpArray:
        self.sow("intermediates", "pre_encoder", x)
        x = self.encoder(x)
        self.sow("intermediates", "encoder_output", x)
        if self.relufication:
            x = fxp_relu(x)
        self.sow("intermediates", "encoder_output_relu", x)
        for idx, layer in enumerate(self.seq_layers):
            x = layer(x)
            self.sow("intermediates", f"layer_{idx}_output", x)
        return x

    def export(self):
        encoder_data = self.encoder.export()
        layers_data = [layer.export() for layer in self.seq_layers]
        data = {
            key: {
                f"layers_{idx}": layers_data[idx][key] for idx in range(self.n_layers)
            }
            for key in ["params", "intermediates", "qconfig"]
        }
        data["params"]["encoder"] = encoder_data["params"]
        data["intermediates"]["encoder"] = encoder_data["intermediates"]
        data["qconfig"]["encoder"] = encoder_data["qconfig"]
        data["intermediates"] = {
            **data["intermediates"],
            **self.last_intermediates(),
        }
        return data


@dataclass
class FxpClassificationModel(FxpModule):
    mixer_cls: FxpModule  # ssm init function
    n_layers: int
    d_model: int
    batchnorm: bool = True
    prenorm: bool = False
    bn_momentum: float = 0.9
    glu_variant: str = "none"
    step_rescale: float = 1.0
    relufication: bool = False
    fuse_batchnorm_linear: bool = False
    q_config: QuantizationConfig = QuantizationConfig.none()
    dropout: float = 0.2  # n/a in inference-only
    training: bool = True  # n/a in inference-only
    # args for classification model
    d_output: int = None
    padded: bool = False
    mode: str = "pool"

    def setup(self):
        assert self.batchnorm, "Only batchnorm is supported for now."
        assert self.dropout == 0.0, "Only dropout=0.0 is supported for now."
        assert not self.training, "Only training=False is supported for now."
        assert self.relufication, "Only relufication=True is supported for now."

        self.encoder = FxpStackedEncoderModel(
            modeldict=self.modeldict["encoder"],
            fxp_qconfig=self.fxp_qconfig,
            scope=self.scope + ".encoder",
            store_intermediates=self.store_intermediates,
            mixer_cls=self.mixer_cls,
            n_layers=self.n_layers,
            d_model=self.d_model,
            dropout=self.dropout,
            batchnorm=self.batchnorm,
            prenorm=self.prenorm,
            bn_momentum=self.bn_momentum,
            glu_variant=self.glu_variant,
            training=self.training,
            step_rescale=self.step_rescale,
            relufication=self.relufication,
            fuse_batchnorm_linear=self.fuse_batchnorm_linear,
            q_config=self.q_config,
        )
        self.decoder = FxpDense(
            modeldict=self.modeldict["decoder"],
            fxp_qconfig=self.fxp_qconfig["decoder"],
            scope=self.scope + ".decoder",
            store_intermediates=self.store_intermediates,
        )

    def forward(self, x: FxpArray, integration_timesteps: int) -> FxpArray:
        if self.padded:
            x, length = x

        x = self.encoder(x, integration_timesteps)
        if self.mode in ["pool"]:
            x = masked_meanpool(x, length) if self.padded else fxp_mean(x, axis=0)
        elif self.mode in ["last"] and not self.padded:
            x = x[-1]
        elif self.mode in ["last"] and self.padded:
            raise NotImplementedError("Mode must be in ['pool'] for self.padded=True")
        else:
            raise NotImplementedError("Mode must be in ['pool', 'last]")

        x = self.decoder(x)
        return fxp_log_softmax(x, axis=-1)  # TODO: implement in fxp

    def export(self):
        encoder_data = self.encoder.export()
        decoder_data = self.decoder.export()
        return dict(
            params=dict(
                encoder=encoder_data["params"],
                decoder=decoder_data["params"],
            ),
            qconfig=dict(
                encoder=encoder_data["qconfig"],
                decoder=decoder_data["qconfig"],
            ),
            intermediates=dict(
                encoder=encoder_data["intermediates"],
                decoder=decoder_data["intermediates"],
            ),
        )


@dataclass
class FxpRegressionModel(FxpModule):
    mixer_cls: FxpModule  # ssm init function
    n_layers: int
    d_model: int
    batchnorm: bool = True
    prenorm: bool = False
    bn_momentum: float = 0.9
    glu_variant: str = "none"
    step_rescale: float = 1.0
    relufication: bool = False
    fuse_batchnorm_linear: bool = False
    q_config: QuantizationConfig = QuantizationConfig.none()
    dropout: float = 0.2  # n/a in inference-only
    training: bool = True  # n/a in inference-only
    # args for regression model
    d_output: int = None
    padded: bool = False

    def setup(self):
        assert self.batchnorm, "Only batchnorm is supported for now."
        assert self.dropout == 0.0, "Only dropout=0.0 is supported for now."
        assert not self.training, "Only training=False is supported for now."
        assert self.relufication, "Only relufication=True is supported for now."

        self.encoder = FxpStackedEncoderModel(
            modeldict=self.modeldict["encoder"],
            fxp_qconfig=self.fxp_qconfig,
            scope=self.scope + ".encoder",
            store_intermediates=self.store_intermediates,
            mixer_cls=self.mixer_cls,
            n_layers=self.n_layers,
            d_model=self.d_model,
            dropout=self.dropout,
            batchnorm=self.batchnorm,
            prenorm=self.prenorm,
            bn_momentum=self.bn_momentum,
            glu_variant=self.glu_variant,
            training=self.training,
            step_rescale=self.step_rescale,
            relufication=self.relufication,
            fuse_batchnorm_linear=self.fuse_batchnorm_linear,
            q_config=self.q_config,
        )
        self.decoder = FxpDense(
            modeldict=self.modeldict["decoder"],
            fxp_qconfig=self.fxp_qconfig["decoder"],
            scope=self.scope + ".decoder",
            store_intermediates=self.store_intermediates,
        )

    def forward(self, x: FxpArray, integration_timesteps: int = 10) -> FxpArray:
        if self.padded:
            x, _ = x

        x = self.encoder(x, integration_timesteps)
        self.sow("intermediates", "encoder_output", x)
        x = self.decoder(x)
        self.sow("intermediates", "output", x)
        return x

    def export(self):
        encoder_data = self.encoder.export()
        decoder_data = self.decoder.export()
        return dict(
            params=dict(
                encoder=encoder_data["params"],
                decoder=decoder_data["params"],
            ),
            qconfig=dict(
                encoder=encoder_data["qconfig"],
                decoder=decoder_data["qconfig"],
            ),
            intermediates=dict(
                encoder=encoder_data["intermediates"],
                decoder=decoder_data["intermediates"],
                **self.last_intermediates(),
            ),
        )
