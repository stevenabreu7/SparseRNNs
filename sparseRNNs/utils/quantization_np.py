from dataclasses import asdict, dataclass
from enum import Enum
from typing import Optional

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
