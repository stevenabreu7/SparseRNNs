from functools import partial

import aqt.jax.v2.flax.aqt_flax as aqt
import jax
import jax.numpy as np
from flax import linen as nn

from sparseRNNs.utils.logging import logger
from sparseRNNs.utils.quantization import (QuantizationConfig, QuantizedDense,
                                           QuantizedMultiply, q_dot_maybe,
                                           q_had_maybe)

GLU_VARIANTS = ["full", "half1", "half2", "none"]


@partial(jax.jit, static_argnames=["k"])
def top_k_sparsity(x, k: int):
    B, _ = x.shape
    top_vals, top_idxs = jax.lax.approx_max_k(x, k)
    sparse_x = (
        np.zeros_like(x)
        .at[np.arange(B)[:, None].repeat(top_vals.shape[-1], axis=1), top_idxs]
        .set(top_vals)
    )
    return sparse_x


@partial(jax.jit, static_argnames=["k"])
def relu_top_k_sparsity(x, k: int):
    x_topk = top_k_sparsity(x, k)
    return jax.nn.relu(x_topk)


@partial(jax.jit, static_argnames=["threshold"])
def jump_relu(x, threshold: float):
    x.at[x <= threshold].set(0.0)
    return x


class QSequenceLayer(nn.Module):
    """Defines a single S5 layer, with S5 SSM, nonlinearity,
        dropout, batch/layer norm, etc.
    Args:
        ssm         (nn.Module): the SSM to be used (i.e. S5 ssm)
        dropout     (float32):  dropout rate
        d_model     (int32):    this is the feature size of the layer inputs and outputs
                                we usually refer to this size as H
        q_bits_aw   (int?, int?): quantization precision for activations and weights
        activation  (string):   Type of activation function to use
        glu_variant  (string):   Type of gated linear unit to use
        training    (bool):     whether in training mode or not
        prenorm     (bool):     apply prenorm if true or postnorm if false
        batchnorm   (bool):     apply batchnorm if true or layernorm if false
        bn_momentum (float32):  the batchnorm momentum if batchnorm is used
        step_rescale  (float32):  allows for uniformly changing the timescale parameter,
                                e.g. after training on a different resolution for
                                the speech commands benchmark
    """

    mixer_cls: nn.Module
    d_model: int
    dropout: float
    batchnorm: bool = True
    prenorm: bool = True
    glu_variant: str = "none"
    bn_momentum: float = 0.90
    training: bool = True
    step_rescale: float = 1.0
    relufication: bool = False
    fuse_batchnorm_linear: bool = False
    q_config: QuantizationConfig = None
    use_batchnorm_scale: bool = True
    use_batchnorm_bias: bool = True
    topk: float = 1.0
    approx_topk: bool = False

    def setup(self):
        """Initializes the ssm, batch/layer norm and dropout"""
        self.mixer = self.mixer_cls(
            step_rescale=self.step_rescale,
            topk=self.topk,
            approx_topk=self.approx_topk,
        )
        q_bits_aw = (
            self.q_config.non_ssm_act_precision,
            self.q_config.non_ssm_precision,
        )
        dot = aqt.AqtDotGeneral(q_dot_maybe(*q_bits_aw, return_cfg=True))
        act_bits, _ = q_bits_aw

        if self.q_config.static_quant:
            logger.debug(
                "Static quantization is enabled. This is an experimental" " feature."
            )
            assert (
                self.q_config.calibrating is not None
            ), "Calibrating must be set if static_quant is True"
            dense_cls = partial(
                QuantizedDense,
                a_bits=q_bits_aw[0],
                w_bits=q_bits_aw[1],
                calibrating=self.q_config.calibrating,
            )
        else:
            dense_cls = partial(nn.Dense, dot_general=dot)

        assert (
            self.glu_variant in GLU_VARIANTS
        ), f"GLU variant must be one of {GLU_VARIANTS}"
        if self.glu_variant == "full":
            self.out1 = dense_cls(self.d_model)
            self.out2 = dense_cls(self.d_model)
        elif self.glu_variant in ["half1", "half2"]:
            self.out2 = dense_cls(self.d_model)

        if self.relufication:
            if self.topk < 1.0 and self.approx_topk:
                self.glu_act_fn = relu_top_k_sparsity
            elif self.topk < 1.0 and not self.approx_topk:
                raise NotImplementedError("Exact top-k sparsity is not yet implemented")
            else:
                self.glu_act_fn = jax.nn.relu
        else:
            self.glu_act_fn = jax.nn.gelu

        if self.topk < 1.0 and self.approx_topk:
            k = int(self.topk * self.d_model)
            self.topk_op = partial(top_k_sparsity, k=k)
        elif self.topk < 1.0 and not self.approx_topk:
            raise NotImplementedError("Exact top-k sparsity is not yet implemented")
        else:
            self.topk_op = lambda x: x

        if self.fuse_batchnorm_linear:
            assert (
                self.batchnorm
            ), "fuse_batchnorm_linear can only be used with batchnorm"
            assert self.prenorm, "fuse_batchnorm_linear can only be used with prenorm"

        if self.batchnorm:
            logger.debug("batchnorm is True in QSequenceLayer, using BatchNorm")
            logger.debug(
                f"batchnorm running_average={not self.training},"
                f" {self.use_batchnorm_bias=}, {self.use_batchnorm_scale=}"
            )
            self.norm = nn.BatchNorm(
                use_running_average=not self.training,
                momentum=self.bn_momentum,
                axis_name="batch",
                use_scale=self.use_batchnorm_scale,
                use_bias=self.use_batchnorm_bias,
            )
        else:
            logger.debug("batchnorm is False in QSequenceLayer, using LayerNorm")
            self.norm = nn.LayerNorm(use_bias=self.use_layernorm_bias)

        self.out = dense_cls(self.d_model)

        dropout_on = self.training
        deterministic = not dropout_on
        logger.debug(f"Dropout: {dropout_on=}, {deterministic=}, rate={self.dropout}")
        self.drop = nn.Dropout(
            self.dropout,
            broadcast_dims=[0],
            deterministic=deterministic,
        )

        if act_bits is not None:
            if self.q_config.static_quant:
                self.mult_gate = QuantizedMultiply(
                    left_bits=act_bits,
                    right_bits=act_bits,
                    calibrating=self.q_config.calibrating,
                )
            else:
                self.mult_gate = q_had_maybe(act_bits, act_bits)
        else:
            self.mult_gate = jax.numpy.multiply

    def __call__(self, x):
        """
        Compute the LxH output of S5 layer given an LxH input.
        Args:
             x (float32): input sequence (L, d_model)
        Returns:
            output sequence (float32): (L, d_model)
        """
        skip = x
        self.sow("intermediates", "input", x)

        vars_exist = ("batch_stats" in self.norm.variables) and (
            "params" in self.norm.variables
        )
        if (
            self.fuse_batchnorm_linear
            and self.batchnorm
            and self.prenorm
            and vars_exist
        ):
            self.sow("intermediates", "pre_s5", x)
            x, x_pre_C = self.mixer(
                x,
                bn_mean=self.norm.variables["batch_stats"]["mean"],
                bn_var=self.norm.variables["batch_stats"]["var"],
                bn_eps=self.norm.epsilon,
                bn_scale=self.norm.variables["params"]["scale"],
                bn_bias=self.norm.variables["params"]["bias"],
            )
        else:
            if self.prenorm:
                x = self.norm(x)
            self.sow("intermediates", "pre_s5", x)
            x, x_pre_C = self.mixer(x)

        self.sow("intermediates", "pre_C", x_pre_C)

        x1 = self.drop(self.glu_act_fn(x))
        self.sow("intermediates", "pre_GLU", x)

        if self.glu_variant == "full":
            x = self.mult_gate(self.out1(x1), jax.nn.sigmoid(self.out2(x1)))
            x = self.drop(x)
        elif self.glu_variant == "half1":
            x = self.mult_gate(x1, jax.nn.sigmoid(self.out2(x1)))
            x = self.drop(x)
        elif self.glu_variant == "half2":
            x = self.mult_gate(x, jax.nn.sigmoid(self.out2(x1)))
            x = self.drop(x)
        elif self.glu_variant == "none":
            x = x1

        x = x + skip

        if not self.prenorm:
            x = self.norm(x)

        if self.relufication:
            x = jax.nn.relu(x)
        x = self.topk_op(x)

        return x
