from dataclasses import asdict, dataclass
from enum import Enum
from functools import partial
from typing import Any, Callable, Optional, Tuple, Union

import aqt.jax.v2.config as aqt_config
import jax
import jax.numpy as jnp
import jax.numpy as np
from aqt.jax.v2.aqt_dot_general import CalibrationMode
from flax import linen as nn
from flax.linen.dtypes import canonicalize_dtype
from flax.linen.module import Module, compact
from flax.linen.normalization import _abs_sq, _canonicalize_axes
from flax.typing import Array, Axes, Dtype, PRNGKey, Shape
from jax import lax
from jax.nn import initializers

#####################################################################
# Quantization Configurations
#####################################################################


class QuantScheme(Enum):
    per_tensor_symmetric: int = 1
    per_tensor_affine: int = 2
    per_channel_symmetric: int = 3
    per_channel_affine: int = 4

    @staticmethod
    def DEFAULT():
        return QuantScheme.per_tensor_symmetric


# @dataclass(unsafe_hash=True)
@dataclass(frozen=True)
class QuantizationConfig:
    """Quantization configuration for S5.

    Attributes:
        a_precision: integer precision for A matrix operations.
        b_precision: integer precision for B matrix operations.
        c_precision: integer precision for C matrix operations.
        d_precision: integer precision for D matrix operations.
        non_ssm_precision: integer precision for all layer operations outside of the SSMs (Dense encode/decode layers)
        ssm_act_precision: integer precision for all SSM activations
        non_ssm_act_precision: integer precision for all non-SSM activations
    """

    a_precision: Optional[int]
    b_precision: Optional[int]
    c_precision: Optional[int]
    d_precision: Optional[int]
    non_ssm_precision: Optional[int]
    ssm_act_precision: Optional[int]
    non_ssm_act_precision: Optional[int]
    # quantization modes (for static quantization)
    static_quant: bool = False
    calibrating: bool = False
    q_scheme: QuantScheme = QuantScheme.DEFAULT()

    @staticmethod
    def none():
        return QuantizationConfig(
            a_precision=None,
            b_precision=None,
            c_precision=None,
            d_precision=None,
            non_ssm_precision=None,
            ssm_act_precision=None,
            non_ssm_act_precision=None,
            static_quant=False,
            calibrating=False,
        )

    def __str__(self):
        return (
            f"qConfig(a={self.a_precision}, b={self.b_precision},"
            f" c={self.c_precision}, d={self.d_precision},"
            f" nonssm={self.non_ssm_precision},"
            f" ssm_act={self.ssm_act_precision},"
            f" nonssm_act={self.non_ssm_act_precision},"
            f" static={self.static_quant}, calibrating={self.calibrating})"
        )

    def __repr__(self):
        return str(self)

    def to_dict(self):
        return {
            key: (value.name if isinstance(value, Enum) or key == "q_scheme" else value)
            for key, value in asdict(self).items()
        }


quantization_recipe_map = {
    "none": partial(
        QuantizationConfig,
        a_precision=None,
        b_precision=None,
        c_precision=None,
        d_precision=None,
        non_ssm_precision=None,
        ssm_act_precision=None,
        non_ssm_act_precision=None,
    ),
    "w8a8": partial(
        QuantizationConfig,
        a_precision=16,
        b_precision=8,
        c_precision=8,
        d_precision=8,
        non_ssm_precision=8,
        ssm_act_precision=8,
        non_ssm_act_precision=8,
    ),
    "w8a8A8": partial(
        QuantizationConfig,
        a_precision=8,
        b_precision=8,
        c_precision=8,
        d_precision=8,
        non_ssm_precision=8,
        ssm_act_precision=8,
        non_ssm_act_precision=8,
    ),
    "w8a16": partial(
        QuantizationConfig,
        a_precision=16,
        b_precision=8,
        c_precision=8,
        d_precision=8,
        non_ssm_precision=8,
        ssm_act_precision=16,
        non_ssm_act_precision=16,
    ),
    "w16a16": partial(
        QuantizationConfig,
        a_precision=16,
        b_precision=16,
        c_precision=16,
        d_precision=16,
        non_ssm_precision=16,
        ssm_act_precision=16,
        non_ssm_act_precision=16,
    ),
    "w32a32": partial(
        QuantizationConfig,
        a_precision=32,
        b_precision=32,
        c_precision=32,
        d_precision=32,
        non_ssm_precision=32,
        ssm_act_precision=32,
        non_ssm_act_precision=32,
    ),
    "w4a4": partial(
        QuantizationConfig,
        a_precision=4,
        b_precision=4,
        c_precision=4,
        d_precision=4,
        non_ssm_precision=4,
        ssm_act_precision=4,
        non_ssm_act_precision=4,
    ),
    "w2a2": partial(
        QuantizationConfig,
        a_precision=2,
        b_precision=2,
        c_precision=2,
        d_precision=2,
        non_ssm_precision=2,
        ssm_act_precision=2,
        non_ssm_act_precision=2,
    ),
}

#####################################################################
# AQT Quantization
#####################################################################


@dataclass
class QuantizedOperations:
    """(Possibly quantized) operations for S5.

    Attributes:
        a_had: (possibly quantized) hadamard product operation for A matrix.
            this is actually a tuple of two hadamart product operators.
            the first one is aa_had for A * A operations (WW)
            the second one is abu_had for A * Bu operations (WA)
        b_dot: (possibly quantized) dot product operation for B matrix.
        c_dot: (possibly quantized) dot product operation for C matrix.
        d_had: (possibly quantized) hadamard product operation for D matrix.
    """

    a_had: Tuple[Callable]  # approved
    b_dot: Callable  # approved
    c_dot: Callable  # approved
    d_had: Callable  # approved
    # non_ssm_dot: Callable  # TODO

    def __init__(self, q_config: QuantizationConfig):
        if q_config.static_quant:
            self.a_had = (np.multiply, np.multiply)
            self.b_dot = np.dot
            self.c_dot = np.dot
            self.d_had = np.multiply
        else:
            self.a_had = (
                q_had_maybe(q_config.a_precision, q_config.a_precision),
                q_had_maybe(q_config.a_precision, q_config.ssm_act_precision),
            )
            self.b_dot = q_dot_maybe(q_config.b_precision, q_config.ssm_act_precision)
            self.c_dot = q_dot_maybe(q_config.c_precision, q_config.ssm_act_precision)
            self.d_had = q_had_maybe(q_config.d_precision, q_config.ssm_act_precision)
            # self.non_ssm_dot = q_dot_maybe(q_config.non_ssm_precision, q_config.ssm_act_precision)


fully_quantized = partial(
    aqt_config.fully_quantized,
    calibration_mode=CalibrationMode.ALL_AXES,
    use_stochastic_rounding=False,
)


def q_dot_maybe(lhs_bits: Optional[int], rhs_bits: Optional[int], return_cfg=False):
    if lhs_bits is None and rhs_bits is None:
        return np.dot if not return_cfg else None
    else:
        precision = (lhs_bits, rhs_bits)
        bwd_bits = max([e for e in precision if e is not None])
        dot_general = fully_quantized(fwd_bits=precision, bwd_bits=bwd_bits)
        if return_cfg:
            return dot_general
        else:
            return quant_dot_for_dot(dot_general)


def q_had_maybe(lhs_bits: Optional[int], rhs_bits: Optional[int], return_cfg=False):
    if lhs_bits is None and rhs_bits is None:
        return np.multiply if not return_cfg else None
    else:
        precision = (lhs_bits, rhs_bits)
        bwd_bits = max([e for e in precision if e is not None])
        dot_general = fully_quantized(fwd_bits=precision, bwd_bits=bwd_bits)
        if return_cfg:
            return dot_general
        else:
            return quant_dot_for_hadamard(dot_general)


def quant_dot_for_hadamard(dot_general):
    """Generate a jitted general_dot function to be used for hadamard products.
    Note that this function does not support batch dimensions. All dimensions will
    be used for calibration in the quantization."""

    def _dot(a, b):
        contr_dims = ((), ())  # hadamard has no contracting dims
        batch_dims = (
            tuple(range(a.ndim)),
            tuple(range(b.ndim)),
        )  # use all dims as batch dims
        return dot_general(a, b, (contr_dims, batch_dims))

    return jax.jit(_dot)


def quant_dot_for_dot(general_dot):
    """Generate a jitted general_dot function to be used for dot products.
    Will contract on the last dimension of a, and the first dimension of b.
    This means that there are no batch dimensions, and all dimensions will be used
    for calibration in the quantization."""

    def _dot(a, b):
        # contr_dims = ((a.ndim-1,), (1,))  # batched version (not used)
        # batch_dims = ((0,), (0,))  # batched version (not used)
        contr_dims = ((a.ndim - 1,), (0,))
        batch_dims = ((), ())
        return general_dot(a, b, (contr_dims, batch_dims))

    return jax.jit(_dot)


#####################################################################
# Static quantization without AQT
#####################################################################


class MinMaxObserver(nn.Module):
    qscheme: QuantScheme
    name: str = ""
    axis_name: str = None  # add axis_name to aggregate over 'batch'

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x.shape: (L, H)
        if self.qscheme == QuantScheme.per_channel_affine:
            raise NotImplementedError("affine quantization not implemented yet")

        # define min value and max value
        if self.qscheme in (
            QuantScheme.per_tensor_symmetric,
            QuantScheme.per_tensor_affine,
        ):
            minval = self.variable(
                "batch_stats",
                f"{self.name or ''}_min",
                lambda: jnp.array(jnp.inf),
            )
            maxval = self.variable(
                "batch_stats",
                f"{self.name or ''}_max",
                lambda: jnp.array(-jnp.inf),
            )
        elif self.qscheme == QuantScheme.per_channel_symmetric:
            channel_dim = x.shape[-1]
            minval = self.variable(
                "batch_stats",
                f"{self.name or ''}_min",
                lambda: jnp.full((channel_dim,), jnp.inf),
            )
            maxval = self.variable(
                "batch_stats",
                f"{self.name or ''}_max",
                lambda: jnp.full((channel_dim,), -jnp.inf),
            )

        # update min and max values
        if self.qscheme in (
            QuantScheme.per_tensor_symmetric,
            QuantScheme.per_tensor_affine,
        ):
            local_min = jnp.minimum(minval.value, jnp.min(x))
            local_max = jnp.maximum(maxval.value, jnp.max(x))
        elif self.qscheme == QuantScheme.per_channel_symmetric:
            local_min = jnp.minimum(minval.value, jnp.min(x, axis=0))
            local_max = jnp.maximum(maxval.value, jnp.max(x, axis=0))

        # Aggregate over the 'batch' axis
        global_min = lax.pmin(local_min, axis_name=self.axis_name)
        global_max = lax.pmax(local_max, axis_name=self.axis_name)

        # Update minval and maxval
        minval.value = jnp.minimum(minval.value, global_min)
        maxval.value = jnp.maximum(maxval.value, global_max)

        return x


def _calculate_qparams(
    minval: jnp.ndarray,
    maxval: jnp.ndarray,
    precision: int,
    qscheme: QuantScheme,
    pow2scale: bool = False,
    eps: float = 1e-6,
) -> Tuple[jnp.ndarray, Union[jnp.ndarray, float]]:
    if qscheme in (
        QuantScheme.per_tensor_symmetric,
        QuantScheme.per_channel_symmetric,
    ):
        max_abs = jnp.maximum(jnp.abs(minval), jnp.abs(maxval))
        quant_max = 2 ** (precision - 1) - 1
        scale = max_abs / quant_max
        scale = jnp.maximum(scale, eps)
        if pow2scale:
            scale = 2 ** jnp.round(jnp.log2(scale))
        return scale, jnp.array(0.0)

    elif qscheme == QuantScheme.per_tensor_affine:
        # TODO: test this
        quant_max = 2 ** (precision) - 1
        scale = (maxval - minval) / quant_max
        scale = jnp.maximum(scale, eps)
        if pow2scale:
            scale = 2 ** jnp.round(jnp.log2(scale))
        zero_point = jnp.round(-1 * minval / scale)
        return scale, zero_point

    else:
        raise NotImplementedError("affine quantization not implemented yet")


def _quantdequant(
    x: jnp.ndarray,
    scale: jnp.ndarray,
    zero_point: Union[jnp.ndarray, float],
    precision: int,
) -> Tuple[jnp.ndarray, Callable]:
    quant_min = -(2 ** (precision - 1))
    quant_max = 2 ** (precision - 1) - 1
    # NOTE: round before or after adding zero_point?
    xq = jnp.round(x / scale + zero_point)
    xq = jnp.clip(xq, quant_min, quant_max)
    xdq = (xq - zero_point) * scale
    # STE for backward pass
    xout = x + jax.lax.stop_gradient(xdq - x)
    return xout


class FakeQuant(nn.Module):
    bits: int = 8
    pow2scale: bool = True
    qscheme: QuantScheme = QuantScheme.DEFAULT()
    calibrating: bool = True

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        scale_var = self.variable("params", "scale", lambda: jnp.array(1.0))
        if self.calibrating:
            scale_var = self.variable("batch_stats", "scale", lambda: jnp.array(1.0))

        if self.qscheme not in (
            QuantScheme.per_tensor_symmetric,
            QuantScheme.per_channel_symmetric,
            QuantScheme.per_tensor_affine,
        ):
            raise NotImplementedError("affine quantization not implemented yet")

        xndim = x.ndim
        if x.ndim == 1:
            x = jnp.expand_dims(x, axis=0)
        elif x.ndim != 2:
            raise ValueError("Input shape must be either 1D or 2D")

        if self.calibrating:
            observer = MinMaxObserver(self.qscheme, name="observer", axis_name="batch")
            x = observer(x)
            minval = observer.variables["batch_stats"]["observer_min"]
            maxval = observer.variables["batch_stats"]["observer_max"]
            scale, _ = _calculate_qparams(
                minval, maxval, self.bits, self.qscheme, self.pow2scale
            )
            scale_var.value = scale
        else:
            scale = scale_var.value

        if self.calibrating:
            x_qdq = x
        else:
            x_qdq = _quantdequant(x, scale, jnp.array(0.0), self.bits)
        # diff_sum = jnp.abs(x_qdq - x).sum()
        # jax.debug.print("sum|x-x_qdq| {} scale {}, bits {}", diff_sum, scale, self.bits)
        if xndim == 1:
            x_qdq = jnp.squeeze(x_qdq, axis=0)
        return x_qdq


class FakeQuantComplex(nn.Module):
    bits: int = 8
    pow2scale: bool = True
    qscheme: QuantScheme = QuantScheme.DEFAULT()
    calibrating: bool = True

    def setup(self):
        self.quant_real = FakeQuant(
            bits=self.bits,
            pow2scale=self.pow2scale,
            qscheme=self.qscheme,
            calibrating=self.calibrating,
        )
        self.quant_imag = FakeQuant(
            bits=self.bits,
            pow2scale=self.pow2scale,
            qscheme=self.qscheme,
            calibrating=self.calibrating,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x_real = self.quant_real(x.real)
        x_imag = self.quant_imag(x.imag)
        return make_complex(x_real, x_imag)


def make_complex(real, imag):
    return real + 1j * imag


class QuantizedMultiply(nn.Module):
    left_bits: int = 8
    right_bits: int = 8
    out_bits: Optional[int] = None
    pow2scale: bool = True
    qscheme: QuantScheme = QuantScheme.DEFAULT()
    calibrating: bool = True
    mul_fn: Callable = jnp.multiply

    def setup(self):
        self.quant_left = FakeQuant(
            bits=self.left_bits,
            pow2scale=self.pow2scale,
            qscheme=self.qscheme,
            calibrating=self.calibrating,
        )
        self.quant_right = FakeQuant(
            bits=self.right_bits,
            pow2scale=self.pow2scale,
            qscheme=self.qscheme,
            calibrating=self.calibrating,
        )
        if self.out_bits is not None:
            self.quant_out = FakeQuant(
                bits=self.out_bits,
                pow2scale=self.pow2scale,
                qscheme=self.qscheme,
                calibrating=self.calibrating,
            )

    def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        x1q = self.quant_left(x1)
        x2q = self.quant_right(x2)
        res = self.mul_fn(x1q, x2q)
        if self.out_bits is not None:
            res = self.quant_out(res)
        return res


class QuantizedComplexMultiply(nn.Module):
    left_bits: int = 8
    right_bits: int = 8
    out_bits: Optional[int] = None
    pow2scale: bool = True
    qscheme: QuantScheme = QuantScheme.DEFAULT()
    calibrating: bool = True
    mul_fn: Callable = jnp.multiply

    def setup(self):
        self.quant_x1re = FakeQuant(
            bits=self.left_bits,
            pow2scale=self.pow2scale,
            qscheme=self.qscheme,
            calibrating=self.calibrating,
        )
        self.quant_x1im = FakeQuant(
            bits=self.left_bits,
            pow2scale=self.pow2scale,
            qscheme=self.qscheme,
            calibrating=self.calibrating,
        )
        self.quant_x2re = FakeQuant(
            bits=self.right_bits,
            pow2scale=self.pow2scale,
            qscheme=self.qscheme,
            calibrating=self.calibrating,
        )
        self.quant_x2im = FakeQuant(
            bits=self.right_bits,
            pow2scale=self.pow2scale,
            qscheme=self.qscheme,
            calibrating=self.calibrating,
        )
        if self.out_bits is not None:
            self.quant_yre = FakeQuant(
                bits=self.out_bits,
                pow2scale=self.pow2scale,
                qscheme=self.qscheme,
                calibrating=self.calibrating,
            )
            self.quant_yim = FakeQuant(
                bits=self.out_bits,
                pow2scale=self.pow2scale,
                qscheme=self.qscheme,
                calibrating=self.calibrating,
            )

    def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        x1q = make_complex(self.quant_x1re(x1.real), self.quant_x1im(x1.imag))
        x2q = make_complex(self.quant_x2re(x2.real), self.quant_x2im(x2.imag))
        res = self.mul_fn(x1q, x2q)
        if self.out_bits is not None:
            res = make_complex(self.quant_yre(res.real), self.quant_yim(res.imag))
        return res


class QuantizedDot(nn.Module):
    left_bits: int = 8
    right_bits: int = 8
    out_bits: Optional[int] = None
    pow2scale: bool = True
    qscheme: QuantScheme = QuantScheme.DEFAULT()
    calibrating: bool = True
    dot_fn: Callable = jnp.dot

    def setup(self):
        self.quant_left = FakeQuant(
            bits=self.left_bits,
            pow2scale=self.pow2scale,
            qscheme=self.qscheme,
            calibrating=self.calibrating,
        )
        self.quant_right = FakeQuant(
            bits=self.right_bits,
            pow2scale=self.pow2scale,
            qscheme=self.qscheme,
            calibrating=self.calibrating,
        )
        if self.out_bits is not None:
            self.quant_out = FakeQuant(
                bits=self.out_bits,
                pow2scale=self.pow2scale,
                qscheme=self.qscheme,
                calibrating=self.calibrating,
            )

    def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        x1q = self.quant_left(x1)
        x2q = self.quant_right(x2)
        res = self.dot_fn(x1q, x2q)
        if self.out_bits is not None:
            res = self.quant_out(res)
        return res


class QuantizedComplexDot(nn.Module):
    left_bits: int = 8
    right_bits: int = 8
    out_bits: Optional[int] = None
    pow2scale: bool = True
    qscheme: QuantScheme = QuantScheme.DEFAULT()
    calibrating: bool = True
    dot_fn: Callable = jnp.dot

    def setup(self):
        self.quant_x1re = FakeQuant(
            bits=self.left_bits,
            pow2scale=self.pow2scale,
            qscheme=self.qscheme,
            calibrating=self.calibrating,
        )
        self.quant_x1im = FakeQuant(
            bits=self.left_bits,
            pow2scale=self.pow2scale,
            qscheme=self.qscheme,
            calibrating=self.calibrating,
        )
        self.quant_x2re = FakeQuant(
            bits=self.right_bits,
            pow2scale=self.pow2scale,
            qscheme=self.qscheme,
            calibrating=self.calibrating,
        )
        self.quant_x2im = FakeQuant(
            bits=self.right_bits,
            pow2scale=self.pow2scale,
            qscheme=self.qscheme,
            calibrating=self.calibrating,
        )
        if self.out_bits is not None:
            self.quant_yre = FakeQuant(
                bits=self.out_bits,
                pow2scale=self.pow2scale,
                qscheme=self.qscheme,
                calibrating=self.calibrating,
            )
            self.quant_yim = FakeQuant(
                bits=self.out_bits,
                pow2scale=self.pow2scale,
                qscheme=self.qscheme,
                calibrating=self.calibrating,
            )

    def __call__(self, x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
        x1q = make_complex(self.quant_x1re(x1.real), self.quant_x1im(x1.imag))
        x2q = make_complex(self.quant_x2re(x2.real), self.quant_x2im(x2.imag))
        res = self.dot_fn(x1q, x2q)
        if self.out_bits is not None:
            res = make_complex(self.quant_yre(res.real), self.quant_yim(res.imag))
        return res


class QuantizedDense(nn.Module):
    features: int
    use_bias: bool = True
    a_bits: int = 8  # Number of bits for quantization of activations
    w_bits: int = 8  # Number of bits for quantization of weights
    quantize_out: bool = True  # Quantize output if True
    a_pow2scale: bool = True  # Use power-of-2 quantization for activations
    w_pow2scale: bool = True  # Use power-of-2 quantization for weights
    qscheme: QuantScheme = QuantScheme.DEFAULT()
    calibrating: bool = True

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        # TODO: add option to one-shot compute the weight scales (more efficient)
        # define kernel and bias parameters (similar to flax Dense)
        kernel = self.param(
            "kernel",
            nn.initializers.lecun_normal(),
            (inputs.shape[-1], self.features),
        )
        bias = (
            self.param("bias", nn.initializers.zeros_init(), (self.features,))
            if self.use_bias
            else None
        )

        v_act_scale = self.variable("params", "act_scale", lambda: jnp.array(1.0))
        v_w_scale = self.variable("params", "weight_scale", lambda: jnp.array(1.0))
        if self.quantize_out:
            v_out_scale = self.variable("params", "out_scale", lambda: jnp.array(1.0))

        if self.calibrating:
            v_act_scale = self.variable(
                "batch_stats", "act_scale", lambda: jnp.array(1.0)
            )
            v_w_scale = self.variable(
                "batch_stats", "weight_scale", lambda: jnp.array(1.0)
            )
            if self.quantize_out:
                v_out_scale = self.variable(
                    "batch_stats", "out_scale", lambda: jnp.array(1.0)
                )

        if inputs.ndim == 1:
            inputs = jnp.expand_dims(inputs, axis=0)
        elif inputs.ndim != 2:
            raise ValueError("Input shape must be either 1D or 2D")

        if self.qscheme not in (
            QuantScheme.per_tensor_symmetric,
            QuantScheme.per_channel_symmetric,
            QuantScheme.per_tensor_affine,
        ):
            raise NotImplementedError("affine quantization not implemented yet")
        per_channel = self.qscheme == QuantScheme.per_channel_symmetric

        # quantize inputs
        if self.calibrating:
            observer = MinMaxObserver(
                self.qscheme, name="input_observer", axis_name="batch"
            )
            inputs = observer(inputs)
            minval = observer.variables["batch_stats"]["input_observer_min"]
            maxval = observer.variables["batch_stats"]["input_observer_max"]
            act_scale, _ = _calculate_qparams(
                minval, maxval, self.a_bits, self.qscheme, self.a_pow2scale
            )
            v_act_scale.value = act_scale
        else:
            act_scale = v_act_scale.value

        if self.calibrating:
            qinputs = inputs
        else:
            qinputs = _quantdequant(inputs, act_scale, jnp.array(0.0), self.a_bits)

        # quantize weights
        if self.calibrating:
            axis = 0 if per_channel else None
            weight_min = jnp.min(kernel, axis=axis)
            weight_max = jnp.max(kernel, axis=axis)
            weight_scale, weight_zero = _calculate_qparams(
                weight_min,
                weight_max,
                self.w_bits,
                self.qscheme,
                self.w_pow2scale,
            )
            v_w_scale.value = weight_scale
        else:
            weight_scale = v_w_scale.value
            weight_zero = 0  # TODO: we support zero-point for weights!!!

        if self.calibrating:
            qkernel = kernel
        else:
            qkernel = _quantdequant(kernel, weight_scale, weight_zero, self.w_bits)

        # compute the output using dot product - NOTE:moved to lax.dot_general (like flax.nn.Dense)
        y = lax.dot_general(
            qinputs,
            qkernel,
            (((qinputs.ndim - 1,), (0,)), ((), ())),
        )

        # quantize and add bias, if used
        if self.use_bias:
            # compute bias scales per output channel
            # bias scale = input activation scale * weight scale (per-channel/per-tensor)
            if per_channel and act_scale.ndim == 0:
                # Ensure act_scale is a scalar if per-tensor quantized activations
                act_scale = jnp.broadcast_to(act_scale, (self.features,))
            if self.calibrating:
                qbias = bias
            else:
                qbias = _quantdequant(bias, act_scale, jnp.array(0.0), self.a_bits)
            y += qbias

        # optionally quantize output
        if self.quantize_out:
            if self.calibrating:
                out_observer = MinMaxObserver(
                    self.qscheme, name="output_observer", axis_name="batch"
                )
                y = out_observer(y)
                out_minval = out_observer.variables["batch_stats"][
                    "output_observer_min"
                ]
                out_maxval = out_observer.variables["batch_stats"][
                    "output_observer_max"
                ]
                out_scale, _ = _calculate_qparams(
                    out_minval,
                    out_maxval,
                    self.a_bits,
                    self.qscheme,
                    self.a_pow2scale,
                )
                v_out_scale.value = out_scale
            else:
                out_scale = v_out_scale.value
                y = _quantdequant(y, out_scale, jnp.array(0.0), self.a_bits)

        return y


# Static quantization utility functions
#######################################


def _merge_trained_params_into_calibrated(trained_params, cal_params):
    tp_flat, _ = jax.tree_util.tree_flatten_with_path(trained_params)
    cp_flat, cp_treedef = jax.tree_util.tree_flatten_with_path(cal_params)
    tp_kp_to_val = dict(tp_flat)

    merged_flat = []
    for kp, val in cp_flat:
        leave_key = kp[-1]
        if leave_key not in ("act_scale", "weight_scale", "out_scale"):
            # only consider moving non-scale parameters
            if kp in tp_kp_to_val:
                # key exists in trained params, so use that value
                merged_flat.append((kp, tp_kp_to_val[kp]))
            else:
                # keep original value from calibration state
                merged_flat.append((kp, val))

    values = [v for _, v in merged_flat]
    merged_params = jax.tree_util.tree_unflatten(cp_treedef, values)
    return merged_params


def _move_scales_to_params(params: dict, batch_stats: dict) -> dict:
    """
    Transfers the scales from batch_stats into the params pytree.

    Args:
        batch_stats (dict): The batch_stats dictionary containing the updated scales.
        params (dict): The params dictionary where the scales will be updated.

    Returns:
        dict: The updated params dictionary with the new scales from batch_stats.
    """
    bs_flattened, _ = jax.tree_util.tree_flatten_with_path(batch_stats)
    p_flattened, p_treedef = jax.tree_util.tree_flatten_with_path(params)
    bs_kp_to_val = dict(bs_flattened)

    merged_flattened = []
    for kp, val in p_flattened:
        if kp in bs_kp_to_val:
            merged_flattened.append((kp, bs_kp_to_val[kp]))
        else:
            merged_flattened.append((kp, val))

    values = [v for _, v in merged_flattened]
    merged_params = jax.tree_util.tree_unflatten(p_treedef, values)
    return merged_params


#####################################################################
# Tests for static quantization
#####################################################################


def test_minmax_observer():
    # Test MinMaxObserver with per-tensor symmetric quantization
    x = jnp.array([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]])
    observer = MinMaxObserver(qscheme=QuantScheme.per_tensor_symmetric)
    variables = observer.init(jax.random.PRNGKey(0), x)
    _ = observer.apply(variables, x, mutable=["batch_stats"])
    minval = variables["batch_stats"]["_min"]
    maxval = variables["batch_stats"]["_max"]
    print("Per-tensor Min:", minval)
    print("Per-tensor Max:", maxval)

    # Test MinMaxObserver with per-channel symmetric quantization
    observer_pc = MinMaxObserver(qscheme=QuantScheme.per_channel_symmetric)
    variables_pc = observer_pc.init(jax.random.PRNGKey(0), x)
    _ = observer_pc.apply(variables_pc, x, mutable=["batch_stats"])
    minval_pc = variables_pc["batch_stats"]["_min"]
    maxval_pc = variables_pc["batch_stats"]["_max"]
    print("Per-channel Min:", minval_pc)
    print("Per-channel Max:", maxval_pc)
    assert minval_pc.min() == minval, "Per-channel Min differs from per-tensor Min"
    assert maxval_pc.max() == maxval, "Per-channel Max differs from per-tensor Max"


def test_quantdequant():
    x = jnp.linspace(-1.0, 1.0, num=10)
    scale = 0.1
    zero_point = 0
    precision = 8
    print(f"{scale = }, {zero_point = }, {precision = }")
    x_qdq = _quantdequant(x, scale, zero_point, precision)
    print("Original x:", x)
    print("Quantized and dequantized x:", x_qdq)
    assert jnp.allclose(x, x_qdq, atol=0.1), "x and x_qdq differ by > 0.1"
    assert not jnp.allclose(x, x_qdq, atol=0.01), "x and x_qdq differ by < 0.01"


def test_quantized_dense():
    x = jnp.ones((2, 4))
    model = QuantizedDense(features=3, qscheme=QuantScheme.per_tensor_symmetric)
    variables = model.init(jax.random.PRNGKey(0), x)
    y, newvars = model.apply(variables, x, mutable=["batch_stats"])
    print("Output y:", y)
    # batch stats should not change (init and apply are using the same input)
    assert jax.tree_util.tree_all(
        jax.tree_map(jnp.allclose, variables["batch_stats"], newvars["batch_stats"])
    ), "Batch stats differ"
    # batch stats should change (init and apply are using different input)
    y, newvars = model.apply(variables, x * 0, mutable=["batch_stats"])
    assert not jax.tree_util.tree_all(
        jax.tree_map(jnp.allclose, variables["batch_stats"], newvars["batch_stats"])
    ), "Batch stats did not change"


def test_backward_pass():
    x = jnp.ones((2, 4))
    y_true = jnp.ones((2, 3))

    def loss_fn(params, batch_stats):
        model = QuantizedDense(features=3, qscheme=QuantScheme.per_tensor_symmetric)
        y_pred, newstate = model.apply(
            {"params": params, "batch_stats": batch_stats},
            x,
            mutable=["batch_stats"],
        )
        loss = jnp.mean((y_pred - y_true) ** 2)
        return loss, newstate

    model = QuantizedDense(features=3, qscheme=QuantScheme.per_tensor_symmetric)
    variables = model.init(jax.random.PRNGKey(0), jnp.zeros_like(x))
    params = variables["params"]
    batch_stats = variables["batch_stats"]
    (loss_value, newvars), grads = jax.value_and_grad(loss_fn, has_aux=True)(
        params, batch_stats
    )
    grads_abs_sum = jax.tree_util.tree_map(lambda x: jnp.abs(x).sum(), grads)
    assert sum(jax.tree_util.tree_leaves(grads_abs_sum)) > 0, "Gradients are all zeros"


def test_quantized_dense_calibration_and_inference():
    # Calibration phase
    x = jnp.array([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]])
    model = QuantizedDense(
        features=3, qscheme=QuantScheme.per_tensor_symmetric, calibrating=True
    )
    variables = model.init(jax.random.PRNGKey(0), x)

    # Run the model to calibrate observers
    x = jnp.array([[1.0, 1.5, -3.0], [4.0, -3.0, 4.5]])
    y_cal, newvars = model.apply(variables, x, mutable=["batch_stats"])
    print("Output during calibration:", y_cal)

    assert jnp.allclose(
        variables["params"]["act_scale"], jnp.array(1.0)
    ), "Input act scale is not 1.0 (params)"
    assert jnp.allclose(
        variables["params"]["weight_scale"], jnp.array(1.0)
    ), "Weight scale is not 1.0 (params)"
    assert jnp.allclose(
        variables["params"]["out_scale"], jnp.array(1.0)
    ), "Output act scale is not 1.0 (params)"
    assert not jnp.allclose(
        variables["batch_stats"]["act_scale"], 1.0
    ), "Input act scale is 1.0 (batch stats)"
    assert not jnp.allclose(
        variables["batch_stats"]["weight_scale"], 1.0
    ), "Weight scale is 1.0 (batch stats)"
    assert not jnp.allclose(
        variables["batch_stats"]["out_scale"], 1.0
    ), "Output act scale is 1.0 (batch stats)"
    assert not jnp.allclose(
        newvars["batch_stats"]["act_scale"], 1.0
    ), "Input act scale is 1.0 (new batch stats)"
    assert not jnp.allclose(
        newvars["batch_stats"]["weight_scale"], 1.0
    ), "Weight scale is 1.0 (new batch stats)"
    assert not jnp.allclose(
        newvars["batch_stats"]["out_scale"], 1.0
    ), "Output act scale is 1.0 (new batch stats)"

    # Inference phase
    # Now set inference=True and use the stored parameters
    params = variables["params"]
    inference_model = QuantizedDense(
        features=3, qscheme=QuantScheme.per_tensor_symmetric, calibrating=False
    )
    # Use the same params but without batch_stats
    y_inference = inference_model.apply({"params": params}, x)
    print("Output during inference:", y_inference)
    assert not jnp.allclose(
        y_cal, y_inference, atol=1e-5
    ), "Outputs equal between calibration and non-transferred inference!"

    # move scales from batch_stats to params
    params = _move_scales_to_params(params, newvars["batch_stats"])
    inference_model = QuantizedDense(
        features=3, qscheme=QuantScheme.per_tensor_symmetric, calibrating=False
    )
    # Use the same params but without batch_stats
    y_inference = inference_model.apply({"params": params}, x)

    print("Output during inference:", y_inference)
    assert jnp.allclose(
        y_cal, y_inference, atol=1e-5
    ), "Outputs differ between calibration and inference!"


class _InternalNestedQuantizedModel(nn.Module):
    layer_sizes: Tuple[int]
    qscheme: QuantScheme = QuantScheme.per_tensor_symmetric
    calibrating: bool = True

    @nn.compact
    def __call__(self, inputs: jnp.ndarray) -> jnp.ndarray:
        x = inputs
        # Create a stack of quantized dense layers
        for i, size in enumerate(self.layer_sizes):
            x = QuantizedDense(
                features=size,
                qscheme=self.qscheme,
                calibrating=self.calibrating,
            )(x)
            x = nn.sigmoid(x) if i < len(self.layer_sizes) - 1 else nn.tanh(x)
        return x


def test_nested_model_calibration_and_inference():
    # Create a nested model with 2 layers of different sizes
    layer_sizes = (4, 2)
    x = jnp.array([[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]])

    # Calibration phase - using x/20 as input
    model = _InternalNestedQuantizedModel(
        layer_sizes=layer_sizes,
        qscheme=QuantScheme.per_tensor_symmetric,
        calibrating=True,
    )
    variables = model.init(jax.random.PRNGKey(0), x / 20)

    # Run the model to calibrate observers for each layer
    y_cal, newvars = model.apply(variables, x, mutable=["batch_stats"])
    print("Output during calibration:", y_cal)

    # move scales from batch_stats to params
    params = _move_scales_to_params(variables["params"], newvars["batch_stats"])
    inference_model = _InternalNestedQuantizedModel(
        layer_sizes=layer_sizes,
        qscheme=QuantScheme.per_tensor_symmetric,
        calibrating=False,
    )
    y_inference = inference_model.apply({"params": variables["params"]}, x)
    y_inference_updated = inference_model.apply({"params": params}, x)

    print("Output during inference (uncalibrated scales):", y_inference)
    print("Output during inference (calibrated scales):", y_inference_updated)

    # Ensure calibration and inference outputs are consistent
    assert not jnp.allclose(
        y_cal, y_inference, atol=1e-5
    ), "Outputs equal between calibration and non-transferred inference!"
    assert jnp.allclose(
        y_cal, y_inference_updated, atol=1e-5
    ), "Outputs differ between calibration and inference!"


def run_all_tests():
    print("\nTesting MinMaxObserver...")
    test_minmax_observer()
    print("\nTesting _quantdequant function...")
    test_quantdequant()
    print("\nTesting QuantizedDense layer...")
    test_quantized_dense()
    print("\nTesting backward pass through QuantizedDense...")
    test_backward_pass()
    print("\nTesting QuantizedDense in calibration and inference modes...")
    test_quantized_dense_calibration_and_inference()
    print("\nTesting NestedQuantizedModel in calibration and inference modes...")
    test_nested_model_calibration_and_inference()


#####################################################################
# Old quantization code from Q-S5 paper
#####################################################################


def q_gelu(precision: int):
    """
    Quantized hard squish function to approximate GeLU.
    Operates purely on integers without needing floating points.
    """
    _q_had = q_had_maybe(precision, precision)

    def _hard_sigmoid(x):  # this operates purely on integers!
        return (
            jnp.minimum(jnp.maximum(0, x + 2), 4) / 4
        )  # jnp.right_shift allows for pure integer input/output!

    def _q_gelu(x):
        return _q_had(x, _hard_sigmoid(x))

    return jax.jit(_q_gelu)


def _compute_stats(
    x: Array,
    axes: Axes,
    dtype: Optional[Dtype],
    axis_name: Optional[str] = None,
    axis_index_groups: Any = None,
):
    """Computes mean and variance statistics.

    This implementation takes care of a few important details:
    - Computes in float32 precision for stability in half precision training.
    - mean and variance are computable in a single XLA fusion,
      by using Var = E[|x|^2] - |E[x]|^2 instead of Var = E[|x - E[x]|^2]).
    - Clips negative variances to zero which can happen due to
      roundoff errors. This avoids downstream NaNs.
    - Supports averaging across a parallel axis and subgroups of a parallel axis
      with a single `lax.pmean` call to avoid latency.

    Arguments:
      x: Input array.
      axes: The axes in ``x`` to compute mean and variance statistics for.
      dtype: Optional dtype specifying the minimal precision. Statistics
        are always at least float32 for stability (default: dtype of x).
      axis_name: Optional name for the pmapped axis to compute mean over.
      axis_index_groups: Optional axis indices.

    Returns:
      A pair ``(mean, var)``.
    """
    if dtype is None:
        dtype = jnp.result_type(x)  # TODO is this concerning???
    # promote x to at least float32, this avoids half precision computation
    # but preserves double or complex floating points
    dtype = jnp.promote_types(dtype, jnp.float32)
    x = jnp.asarray(x, dtype)

    mean = jnp.mean(x, axes)
    mean2 = jnp.mean(_abs_sq(x), axes)
    if axis_name is not None:
        concatenated_mean = jnp.concatenate([mean, mean2])
        mean, mean2 = jnp.split(
            lax.pmean(
                concatenated_mean,
                axis_name=axis_name,
                axis_index_groups=axis_index_groups,
            ),
            2,
        )
    # mean2 - _abs_sq(mean) is not guaranteed to be non-negative due
    # to floating point round-off errors.
    var = jnp.maximum(0.0, mean2 - _abs_sq(mean))
    return mean, var


def _q_normalize(
    mdl: Module,
    x: Array,
    mean: Array,
    var: Array,
    reduction_axes: Axes,
    feature_axes: Axes,
    dtype: Dtype,
    param_dtype: Dtype,
    epsilon: float,
    use_bias: bool,
    use_scale: bool,
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array],
    scale_init: Callable[[PRNGKey, Shape, Dtype], Array],
    quantized_hadamard_operator: Callable,
):
    """ "Normalizes the input of a normalization layer and optionally applies a learned scale and bias.
    Arguments:
      mdl: Module to apply the normalization in (normalization params will reside
        in this module).
      x: The input.
      mean: Mean to use for normalization.
      var: Variance to use for normalization.
      reduction_axes: The axes in ``x`` to reduce.
      feature_axes: Axes containing features. A separate bias and scale is learned
        for each specified feature.
      dtype: The dtype of the result (default: infer from input and params).
      param_dtype: The dtype of the parameters.
      epsilon: Normalization epsilon.
      use_bias: If true, add a bias term to the output.
      use_scale: If true, scale the output.
      bias_init: Initialization function for the bias term.
      scale_init: Initialization function for the scaling function.
    Returns:
      The normalized input.
    """
    reduction_axes = _canonicalize_axes(x.ndim, reduction_axes)
    feature_axes = _canonicalize_axes(x.ndim, feature_axes)
    stats_shape = list(x.shape)
    for axis in reduction_axes:
        stats_shape[axis] = 1
    mean = mean.reshape(stats_shape)
    var = var.reshape(stats_shape)
    feature_shape = [1] * x.ndim
    reduced_feature_shape = []
    for ax in feature_axes:
        feature_shape[ax] = x.shape[ax]
        reduced_feature_shape.append(x.shape[ax])
    y = x - mean
    mul = lax.rsqrt(var + epsilon)
    args = [x]
    if use_scale:
        scale = mdl.param(
            "scale", scale_init, reduced_feature_shape, param_dtype
        ).reshape(feature_shape)
        mul *= scale
        args.append(scale)
    y = quantized_hadamard_operator(y, mul)
    if use_bias:
        bias = mdl.param("bias", bias_init, reduced_feature_shape, param_dtype).reshape(
            feature_shape
        )
        y += bias
        args.append(bias)
    dtype = canonicalize_dtype(*args, dtype=dtype)
    return jnp.asarray(y, dtype)


class QLayerNorm(Module):
    """Layer normalization (https://arxiv.org/abs/1607.06450).

    LayerNorm normalizes the activations of the layer for each given example in a
    batch independently, rather than across a batch like Batch Normalization.
    i.e. applies a transformation that maintains the mean activation within
    each example close to 0 and the activation standard deviation close to 1.

    Attributes:
      epsilon: A small float added to variance to avoid dividing by zero.
      dtype: the dtype of the result (default: infer from input and params).
      param_dtype: the dtype passed to parameter initializers (default: float32).
      use_bias:  If True, bias (beta) is added.
      use_scale: If True, multiply by scale (gamma). When the next layer is linear
        (also e.g. nn.relu), this can be disabled since the scaling will be done
        by the next layer.
      bias_init: Initializer for bias, by default, zero.
      scale_init: Initializer for scale, by default, one.
      reduction_axes: Axes for computing normalization statistics.
      feature_axes: Feature axes for learned bias and scaling.
    """

    scaling_quantization: int
    epsilon: float = 1e-6
    dtype: Optional[Dtype] = None
    param_dtype: Dtype = jnp.float32
    use_bias: bool = False
    use_scale: bool = True
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.zeros
    scale_init: Callable[[PRNGKey, Shape, Dtype], Array] = initializers.ones
    reduction_axes: Axes = -1
    feature_axes: Axes = -1

    @compact
    def __call__(self, x):
        """Applies layer normalization on the input.

        Args:
          x: the inputs

        Returns:
          Normalized inputs (the same shape as inputs).
        """
        mean, var = _compute_stats(x, self.reduction_axes, self.dtype, None, None)
        scale_q_hadamard = q_had_maybe(
            self.scaling_quantization, self.scaling_quantization
        )
        return _q_normalize(
            self,
            x,
            mean,
            var,
            self.reduction_axes,
            self.feature_axes,
            self.dtype,
            self.param_dtype,
            self.epsilon,
            self.use_bias,
            self.use_scale,
            self.bias_init,
            self.scale_init,
            scale_q_hadamard,
        )  # TODO is this efficient??


if __name__ == "__main__":
    run_all_tests()
