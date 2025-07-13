import inspect
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Callable, Optional, Tuple, Any

import numpy as np

Array = Any

from sparseRNNs.utils.logging import logger


class RoundingMode(Enum):
    FLOOR = 0
    CEIL = 1
    ROUND = 2
    STOCHASTIC = 3


def round_array(
    x: Array, round_mode: RoundingMode = RoundingMode.FLOOR, dtype=np.int32
) -> Array:
    if round_mode == RoundingMode.ROUND:
        return np.round(x).astype(dtype)
    elif round_mode == RoundingMode.CEIL:
        return np.ceil(x).astype(dtype)
    elif round_mode == RoundingMode.FLOOR:
        return np.floor(x).astype(dtype)
    else:
        raise NotImplementedError(f"rounding mode '{round_mode}' not implemented")


@dataclass
class FxpArray:
    data: Optional[Array] = None
    bits: int = 16
    exp: int = 8
    signed: bool = True

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def dtype(self):
        return self.data.dtype

    def copy(self) -> "FxpArray":
        return FxpArray(
            data=self.data.copy(),
            bits=self.bits,
            exp=self.exp,
            signed=self.signed,
        )

    def minval(self) -> int:
        return fxp_minval(self)

    def maxval(self) -> int:
        return fxp_maxval(self)

    def clip(self, do_warn: bool = False, warn_prefix: str = ""):
        return fxp_clip(self, do_warn=do_warn, warn_prefix=warn_prefix)

    def is_valid(self, do_warn: bool = False) -> bool:
        return fxp_isvalid(self, do_warn=do_warn)

    def to_float(self) -> Array:
        return self.data.astype(np.float32) / (1 << self.exp)

    def change_exp(
        self,
        new_exp: int,
        round_mode: RoundingMode = RoundingMode.FLOOR,
        warn_on_clip: bool = True,
    ) -> "FxpArray":
        return fxp_change_exp(
            self,
            new_exp=new_exp,
            round_mode=round_mode,
            warn_on_clip=warn_on_clip,
        )

    def change_cfg(
        self,
        new_bits: int,
        new_exp: int,
        new_signed: bool,
        round_mode: RoundingMode = RoundingMode.FLOOR,
        warn_on_clip: bool = True,
    ) -> "FxpArray":
        return fxp_change_cfg(
            self, new_bits, new_exp, new_signed, round_mode, warn_on_clip
        )

    def __eq__(self, val):
        if not isinstance(val, FxpArray):
            raise TypeError(
                f"unsupported type for comparison with FxpArray: '{type(val)}'"
            )
        cfg_eq = (
            self.bits == val.bits and self.exp == val.exp and self.signed == val.signed
        )
        data_eq = np.array_equal(self.data, val.data)
        return cfg_eq and data_eq

    def __add__(self, other):
        return fxp_add(self, other)

    def __sub__(self, other):
        minus_other = FxpArray(
            data=-1 * other.data,
            bits=other.bits,
            exp=other.exp,
            signed=other.signed,
        )
        return fxp_add(self, minus_other)

    def __mul__(self, other):
        return fxp_mul(self, other)

    def __matmul__(self, other):
        return fxp_matmul(self, other)

    def __str__(self):
        return (
            f"FxpArray(bits={self.bits}, exp={self.exp}, signed={self.signed},"
            f" data={self.data})"
        )

    def __repr__(self):
        return (
            f"FxpArray(bits={self.bits}, exp={self.exp}, signed={self.signed},"
            f" \n         data={repr(self.data)})"
        )

    def __getitem__(self, key) -> "FxpArray":
        return FxpArray(
            data=self.data[key],
            bits=self.bits,
            exp=self.exp,
            signed=self.signed,
        )

    def transpose(self) -> "FxpArray":
        return FxpArray(
            data=self.data.T, bits=self.bits, exp=self.exp, signed=self.signed
        )

    @staticmethod
    def Zero(shape=(1,), bits=16, exp=8, signed=True):
        return FxpArray(
            data=np.zeros(shape, dtype=np.int32),
            bits=bits,
            exp=exp,
            signed=signed,
        )

    @staticmethod
    def Zero_like(arr: "FxpArray"):
        return FxpArray.Zero(
            shape=arr.data.shape, bits=arr.bits, exp=arr.exp, signed=arr.signed
        )


@dataclass
class ComplexFxpArray:
    real: FxpArray
    imag: FxpArray

    @property
    def shape(self):
        assert self.real.shape == self.imag.shape, "real and imag shapes do not match"
        return self.real.shape

    @property
    def ndim(self):
        assert self.real.ndim == self.imag.ndim, "real and imag ndims do not match"
        return self.real.ndim

    @property
    def dtype(self):
        assert self.real.dtype == self.imag.dtype, "real and imag dtypes do not match"
        return self.real.dtype

    def copy(self) -> "ComplexFxpArray":
        return ComplexFxpArray(real=self.real.copy(), imag=self.imag.copy())

    def clip(self, **kwargs):
        return ComplexFxpArray(
            real=self.real.clip(**kwargs), imag=self.imag.clip(**kwargs)
        )

    def is_valid(self, **kwargs) -> bool:
        return self.real.is_valid(**kwargs) and self.imag.is_valid(**kwargs)

    def to_float(self) -> Array:
        return self.real.to_float() + 1j * self.imag.to_float()

    def transpose(self) -> "ComplexFxpArray":
        return ComplexFxpArray(real=self.real.transpose(), imag=self.imag.transpose())

    def __getitem__(self, key) -> "ComplexFxpArray":
        return ComplexFxpArray(real=self.real[key], imag=self.imag[key])

    def __add__(self, other):
        return ComplexFxpArray(real=self.real + other.real, imag=self.imag + other.imag)

    def __mul__(self, other):
        return ComplexFxpArray(
            real=self.real * other.real - self.imag * other.imag,
            imag=self.real * other.imag + self.imag * other.real,
        )

    def __matmul__(self, other):
        return ComplexFxpArray(
            real=self.real @ other.real - self.imag @ other.imag,
            imag=self.real @ other.imag + self.imag @ other.real,
        )

    def __str__(self):
        return f"ComplexFxpArray(real={self.real}, imag={self.imag})"

    def __repr__(self):
        return f"ComplexFxpArray(real={repr(self.real)}, imag={repr(self.imag)})"


def fxp_change_cfg(
    x: FxpArray,
    new_bits: int,
    new_exp: int,
    new_signed: bool,
    round_mode: RoundingMode = RoundingMode.FLOOR,
    warn_on_clip: bool = True,
    warn_prefix: str = "",
) -> FxpArray:
    if x.bits == new_bits and x.exp == new_exp and x.signed == new_signed:
        return x
    # change exp
    new_array = fxp_change_exp(
        x,
        new_exp=new_exp,
        round_mode=round_mode,
        warn_on_clip=warn_on_clip,
        warn_prefix=f"{warn_prefix}.change_cfg",
    )
    # change bits
    if new_array.bits > new_bits:
        logger.debug(
            f"changing bits from {new_array.bits} to {new_bits} may cause" " overflow"
        )
        new_array.bits = new_bits
        new_array = new_array.clip(
            do_warn=warn_on_clip, warn_prefix=f"{warn_prefix}.change_bits"
        )
    else:
        new_array.bits = new_bits
    # change signed
    if not new_array.signed and new_signed:
        logger.debug(f"changing from unsigned to signed may cause overflow")
        new_array.signed = new_signed
        new_array = new_array.clip(
            do_warn=warn_on_clip, warn_prefix=f"{warn_prefix}.change_sign"
        )
    else:
        new_array.signed = new_signed
    return new_array


def fxp_rshift_round(
    x: Array, rshift: int, round_mode: RoundingMode = RoundingMode.FLOOR
) -> Array:
    if round_mode == RoundingMode.FLOOR:
        return x >> rshift
    elif round_mode == RoundingMode.CEIL:
        return (x + (1 << rshift) - 1) >> rshift  # add 1.0-eps
    elif round_mode == RoundingMode.ROUND:
        return (x + (1 << (rshift - 1))) >> rshift  # add 0.5f
    elif round_mode == RoundingMode.STOCHASTIC:
        raise NotImplementedError("STOCHASTIC rounding mode not implemented")


def fxp_from_fp(
    x: Array,
    bits: int = 16,
    exp: int = 8,
    signed: bool = True,
    round_mode: RoundingMode = RoundingMode.FLOOR,
    warn_on_clip: bool = True,
    warn_prefix: str = "",
    dtype=np.int32,
) -> FxpArray:
    xint = x * (1 << (exp))
    if not signed and np.any(xint < 0):
        logger.warning(
            f"{warn_prefix}.from_fp - negative values detected in unsigned"
            " conversion"
        )
        xint = np.abs(xint)
    xint = round_array(xint, round_mode=round_mode, dtype=dtype)
    xfxp = FxpArray(data=xint, bits=bits, exp=exp, signed=signed)
    xfxp = fxp_clip(xfxp, do_warn=warn_on_clip, warn_prefix=f"{warn_prefix}.from_fp")
    return xfxp


def fxp_change_exp(
    arr: FxpArray,
    new_exp: int,
    round_mode: RoundingMode = RoundingMode.FLOOR,
    warn_on_clip: bool = True,
    warn_prefix: str = "",
) -> FxpArray:
    arr = arr.copy()
    if new_exp == arr.exp:
        return arr
    elif new_exp > arr.exp:
        arr.data = arr.data << (new_exp - arr.exp)
    else:
        arr.data = fxp_rshift_round(arr.data, arr.exp - new_exp, round_mode)
    arr.exp = new_exp
    arr = fxp_clip(arr, do_warn=warn_on_clip, warn_prefix=f"{warn_prefix}_change_exp")
    return arr


def fxp_minval(arr: FxpArray) -> int:
    return -(1 << (arr.bits - 1)) if arr.signed else 0


def fxp_maxval(arr: FxpArray) -> int:
    return (1 << (arr.bits - 1)) - 1 if arr.signed else (1 << arr.bits) - 1


def print_call_stack():
    x = ["--".join(frame.code_context).strip() for frame in inspect.stack()[1:][::-1]]
    x = [e.split(" = ")[-1] for e in x]
    last_idxs = [idx for idx in range(len(x)) if "print_call_stack" in x[idx]]
    x = x[: last_idxs[0]] if len(x) > 0 else x
    x = [e for e in x if "return self.forward(*args, **kwargs)" not in e]
    print(" -> ".join(x))


def fxp_clip(arr: FxpArray, do_warn: bool = False, warn_prefix: str = ""):
    arr = arr.copy()
    minval = fxp_minval(arr)
    maxval = fxp_maxval(arr)
    if np.any(arr.data > maxval) or np.any(arr.data < minval):
        if do_warn:
            logger.warning(
                f"{warn_prefix} - overflow detected when clipping. traceback" " below:"
            )
            print_call_stack()
        arr.data = np.clip(arr.data, minval, maxval)
    return arr


def fxp_sub(
    op1,
    op2,
    result_bits: Optional[int] = None,
    result_bits_fn: Callable[[int, int], int] = max,
    result_bits_add: int = 0,
    result_exp: Optional[int] = None,
    warn_on_overflow: bool = True,
    warn_on_neq_exp: bool = False,
    round_mode: RoundingMode = RoundingMode.FLOOR,
    warn_on_clip: bool = True,
) -> FxpArray:
    return fxp_add(
        op1,
        FxpArray(data=-1 * op2.data, bits=op2.bits, exp=op2.exp, signed=op2.signed),
        result_bits=result_bits,
        result_bits_fn=result_bits_fn,
        result_bits_add=result_bits_add,
        result_exp=result_exp,
        warn_on_overflow=warn_on_overflow,
        warn_on_neq_exp=warn_on_neq_exp,
        round_mode=round_mode,
        warn_on_clip=warn_on_clip,
    )


def fxp_add(
    op1,
    op2,
    result_bits: Optional[int] = None,
    result_bits_fn: Callable[[int, int], int] = max,
    result_bits_add: int = 0,
    result_exp: Optional[int] = None,
    warn_on_overflow: bool = True,
    warn_on_neq_exp: bool = False,
    round_mode: RoundingMode = RoundingMode.FLOOR,
    warn_on_clip: bool = True,
):
    if not (isinstance(op1, FxpArray) and isinstance(op2, FxpArray)):
        return TypeError(
            f"unsupported type(s) for fxp_add: '{type(op1)}' and '{type(op2)}'"
        )
    if warn_on_neq_exp and op1.exp != op2.exp:
        logger.warning(f"unequal exponents: {op1.exp} != {op2.exp}")

    result_signed = op1.signed or op2.signed
    if result_bits is None:
        result_bits = result_bits_fn(op1.bits, op2.bits) + result_bits_add

    # addition + shifting
    if result_exp is None:
        if op1.exp == op2.exp:
            result_exp = op1.exp
            result_data = op1.data + op2.data
        elif op1.exp > op2.exp:
            result_exp = op1.exp
            result_data = op1.data + op2.data << (op1.exp - op2.exp)
        else:
            result_exp = op2.exp
            result_data = op1.data << (op2.exp - op1.exp) + op2.data
    elif result_exp == "compute_best":
        eps = 1e-6
        intbits = max(
            0,
            int(np.ceil(np.log2(np.abs(op1.to_float() + op2.to_float()).max() + eps))),
        )
        result_exp = result_bits - intbits - (1 if result_signed else 0)
        agg_intbits = max(
            max(0, int(np.ceil(np.log2(np.abs(op1.to_float()).max() + 1e-8)))),
            max(0, int(np.ceil(np.log2(np.abs(op2.to_float()).max() + 1e-8)))),
        )
        agg_exp = max(op1.exp, op2.exp)
        agg_bits = agg_intbits + agg_exp + (1 if result_signed else 0)
        op1c = op1.change_cfg(
            new_bits=max(agg_bits, op1.bits),
            new_exp=agg_exp,
            new_signed=result_signed,
        )
        op2c = op2.change_cfg(
            new_bits=max(agg_bits, op2.bits),
            new_exp=agg_exp,
            new_signed=result_signed,
        )
        result_data = op1c.data + op2c.data
        exp_change = result_exp - agg_exp
        if exp_change > 0:
            result_data = result_data << exp_change
        elif exp_change < 0:
            result_data = result_data >> -exp_change
    else:
        change_exp_fn = partial(
            fxp_change_exp, round_mode=round_mode, warn_on_clip=warn_on_clip
        )
        result_data = (
            change_exp_fn(op1, result_exp).data + change_exp_fn(op2, result_exp).data
        )

    # clipping
    result = FxpArray(
        data=result_data,
        bits=result_bits,
        exp=result_exp,
        signed=result_signed,
    )
    result = result.clip(do_warn=warn_on_overflow, warn_prefix="fxp_add")

    return result


def fxp_complex_add(
    op1: ComplexFxpArray,
    op2: ComplexFxpArray,
    result_exp: Optional[Tuple[int, int]] = None,
    result_bits: Optional[Tuple[int, int]] = None,
    result_bits_fn: Callable[[int, int], int] = max,
    result_bits_add: int = 0,
    warn_on_overflow: bool = True,
    round_mode: RoundingMode = RoundingMode.FLOOR,
    warn_on_clip: bool = True,
):
    add_fn = partial(
        fxp_add,
        result_bits_fn=result_bits_fn,
        result_bits_add=result_bits_add,
        warn_on_overflow=warn_on_overflow,
        round_mode=round_mode,
        warn_on_clip=warn_on_clip,
    )
    return ComplexFxpArray(
        real=add_fn(
            op1.real,
            op2.real,
            result_bits=result_bits[0],
            result_exp=result_exp[0],
        ),
        imag=add_fn(
            op1.imag,
            op2.imag,
            result_bits=result_bits[1],
            result_exp=result_exp[1],
        ),
    )


def fxp_complex_mul(
    op1: ComplexFxpArray,
    op2: ComplexFxpArray,
    result_exp: Optional[Tuple[int, int]] = None,
    result_exp_fn: Callable[[int, int], int] = max,
    result_bits: Optional[Tuple[int, int]] = None,
    result_bits_fn: Callable[[int, int], int] = max,
    round_mode: RoundingMode = RoundingMode.FLOOR,
    warn_on_overflow: bool = True,
    warn_on_clip: bool = True,
):
    result_exp = result_exp or (None, None)
    result_bits = result_bits or (None, None)

    mul_fn = partial(
        fxp_mul,
        result_exp_fn=result_exp_fn,
        result_bits_fn=result_bits_fn,
        round_mode=round_mode,
        warn_on_overflow=warn_on_overflow,
    )
    arbr = mul_fn(
        op1.real,
        op2.real,
        result_exp=result_exp[0],
        result_bits=result_bits[0],
    )
    aibr = mul_fn(
        op1.imag,
        op2.real,
        result_exp=result_exp[1],
        result_bits=result_bits[1],
    )
    arbi = mul_fn(
        op1.real,
        op2.imag,
        result_exp=result_exp[1],
        result_bits=result_bits[1],
    )
    aibi = mul_fn(
        op1.imag,
        op2.imag,
        result_exp=result_exp[0],
        result_bits=result_bits[0],
    )

    real_result = fxp_sub(
        arbr,
        aibi,
        result_bits_fn=result_bits_fn,
        result_exp=result_exp[0],
        result_bits=result_bits[0],
        round_mode=round_mode,
        warn_on_overflow=warn_on_overflow,
        warn_on_clip=warn_on_clip,
    )
    imag_result = fxp_add(
        arbi,
        aibr,
        result_bits_fn=result_bits_fn,
        result_exp=result_exp[1],
        result_bits=result_bits[1],
        round_mode=round_mode,
        warn_on_overflow=warn_on_overflow,
        warn_on_clip=warn_on_clip,
    )
    return ComplexFxpArray(real=real_result, imag=imag_result)


def fxp_mul(
    op1,
    op2,
    result_exp: Optional[int] = None,
    result_exp_fn: Callable[[int, int], int] = max,
    result_bits: Optional[int] = None,
    result_bits_fn: Callable[[int, int], int] = max,
    round_mode: RoundingMode = RoundingMode.FLOOR,
    warn_on_overflow: bool = True,
):
    # TODO: implement multiplication with a number?
    if isinstance(op1, ComplexFxpArray) and isinstance(op2, ComplexFxpArray):
        ComplexFxpArray(
            real=op1.real * op2.real - op1.imag * op2.imag,
            imag=op1.real * op2.imag + op1.imag * op2.real,
        )
    if not (isinstance(op1, FxpArray) and isinstance(op2, FxpArray)):
        return TypeError(
            f"unsupported type(s) for fxp_mul: '{type(op1)}' and '{type(op2)}'"
        )

    result_signed = op1.signed or op2.signed
    if result_bits is None:
        result_bits = result_bits_fn(op1.bits, op2.bits)

    # NOTE(sabreu): allow custom result_exp from function argument?
    if result_exp is None:
        result_exp = result_exp_fn(op1.exp, op2.exp)
    elif result_exp == "compute_best":
        logger.debug(f"max result: {np.abs(op1.to_float() * op2.to_float()).max()}")
        eps = 1e-6
        intbits = max(
            0,
            int(np.ceil(np.log2(eps + np.abs(op1.to_float() * op2.to_float()).max()))),
        )
        result_exp = result_bits - intbits - (1 if result_signed else 0)
        logger.debug(f"{intbits=}, {op1.exp=}, {op2.exp=}, {result_exp=}")

    op1bits = np.ceil(np.log2(np.abs(op1.data).max())).astype(int)
    op2bits = np.ceil(np.log2(np.abs(op2.data).max())).astype(int)
    if op1bits + op2bits > 30:
        logger.error("Using 64-bit integers, this should not happen!")
        op1.data = op1.data.astype(np.int64)
        op2.data = op2.data.astype(np.int64)

    # determine right shift for target precision
    rshift = op1.exp + op2.exp - result_exp
    if rshift < 0:
        raise ValueError(f"invalid result_exp: {result_exp}")

    result_data_raw = op1.data * op2.data

    # round and shift to match the exponent
    result_data = fxp_rshift_round(result_data_raw, rshift, round_mode).astype(np.int32)

    # clipping
    result = FxpArray(
        data=result_data,
        bits=result_bits,
        exp=result_exp,
        signed=result_signed,
    )
    result = result.clip(do_warn=warn_on_overflow, warn_prefix="fxp_mul")

    return result


def fxp_matmul(
    op1,
    op2,
    result_bits: Optional[int] = None,
    result_bits_fn: Callable[[int, int], int] = max,
    result_exp: Optional[int] = None,
    result_exp_fn: Callable[[int, int], int] = max,
    round_mode: RoundingMode = RoundingMode.FLOOR,
):
    if not (isinstance(op1, FxpArray) and isinstance(op2, FxpArray)):
        return TypeError(
            f"unsupported type(s) for fxp_matmul: '{type(op1)}' and" f" '{type(op2)}'"
        )

    result_signed = op1.signed or op2.signed
    if result_bits is None:
        result_bits = result_bits_fn(op1.bits, op2.bits)
    if result_exp is None:
        result_exp = result_exp_fn(op1.exp, op2.exp)

    # NOTE: is there already right-shifting in the multiplications?
    # TODO(sabreu): handle overflow in the multiplication vs. addition
    result_data_raw = op1.data @ op2.data
    curr_exp = op1.exp + op2.exp
    rshift = curr_exp - result_exp

    # round and shift to match the exponent
    result_data = fxp_rshift_round(result_data_raw, rshift, round_mode)

    # clipping
    result = FxpArray(
        data=result_data,
        bits=result_bits,
        exp=result_exp,
        signed=result_signed,
    )
    result = result.clip(do_warn=True)

    return result


def fxp_mean(x: FxpArray, axis: int = 0) -> FxpArray:
    logger.warning("fxp_mean function is not tested yet")
    n = x.data.shape[axis]
    recn = np.array(1.0 / n)
    min_exp = int(np.ceil(np.log2(recn)))
    assert x.bits > min_exp, "not enough bits to represent 1/n"
    recn = fxp_from_fp(recn, bits=x.bits, exp=max(x.exp, min_exp), signed=x.signed)
    y = fxp_mul(
        x,
        recn,
        result_exp=x.exp,
        result_bits=x.bits,
        round_mode=RoundingMode.ROUND,
    )
    return y


def fxp_log_softmax(x: FxpArray, axis: int = -1) -> FxpArray:
    # TODO
    logger.warning("log_softmax not implemented, skipping")
    return x


def fxp_isvalid(arr: FxpArray, do_warn: bool = False) -> bool:
    if arr.data is None:
        if do_warn:
            logger.warning(f"invalid FxpArray: data is None")
        return False
    if arr.bits <= 0:
        if do_warn:
            logger.warning(f"invalid FxpArray: bits <= 0")
        return False
    if arr.exp < 0:
        if do_warn:
            logger.warning(f"invalid FxpArray: exp < 0")
        return False
    if np.any(arr.data < arr.minval()) or np.any(arr.data > arr.maxval()):
        if do_warn:
            logger.warning(f"invalid FxpArray: out of bounds")
        return False
    return True
