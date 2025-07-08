from functools import partial
from typing import Optional

import aqt.jax.v2.flax.aqt_flax as aqt
import jax
import jax.numpy as np
from flax import linen as nn

from sparseRNNs.model.layers import QSequenceLayer, relu_top_k_sparsity
from sparseRNNs.utils.quantization import (QuantizationConfig, QuantizedDense,
                                           q_dot_maybe)


@partial(jax.jit, static_argnames=["quant_input_exp"])
def quant_input_fn(x, quant_input_exp: Optional[float] = None):
    if quant_input_exp is None:
        return x
    else:
        xtrunc = np.round(x * 2**quant_input_exp) / 2**quant_input_exp
        return xtrunc


class BaseModel(nn.Module):
    mixer_cls: nn.Module
    n_layers: int
    d_model: int
    dropout: float = 0.2
    batchnorm: bool = True
    prenorm: bool = False
    bn_momentum: float = 0.9
    glu_variant: str = "none"
    training: bool = True
    step_rescale: float = 1.0
    relufication: bool = False
    fuse_batchnorm_linear: bool = False
    q_config: QuantizationConfig = QuantizationConfig.none()
    use_batchnorm_scale: bool = True
    use_batchnorm_bias: bool = True
    topk: float = 1.0
    approx_topk: bool = False
    quant_input: Optional[float] = None


class QStackedEncoderModel(BaseModel):

    def setup(self):
        """
        Initializes a linear encoder and the stack of S5 layers.
        """
        # NOTE: nn.Dense calls dot_general(activation, weights)
        q_bits_aw = (
            self.q_config.non_ssm_act_precision,
            self.q_config.non_ssm_precision,
        )
        if self.q_config.static_quant:
            assert (
                self.q_config.calibrating is not None
            ), "Calibrating must be set if static_quant is True"
            self.encoder = QuantizedDense(
                self.d_model,
                a_bits=q_bits_aw[0],
                w_bits=q_bits_aw[1],
                calibrating=self.q_config.calibrating,
            )
        else:
            dot = aqt.AqtDotGeneral(q_dot_maybe(*q_bits_aw, return_cfg=True))
            self.encoder = nn.Dense(self.d_model, dot_general=dot)

        if self.topk < 1.0 and self.approx_topk:
            self.topk_op = partial(relu_top_k_sparsity, k=int(self.topk * self.d_model))
        elif self.topk < 1.0 and not self.approx_topk:
            raise NotImplementedError("Exact top-k sparsity is not yet implemented")
        elif self.relufication:
            self.topk_op = jax.nn.relu
        else:
            self.topk_op = lambda x: x

        self.layers = [
            QSequenceLayer(
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
                use_batchnorm_scale=self.use_batchnorm_scale,
                use_batchnorm_bias=self.use_batchnorm_bias,
                topk=self.topk,
                approx_topk=self.approx_topk,
            )
            for _ in range(self.n_layers)
        ]

    def __call__(self, x, integration_timesteps):
        """
        Compute the LxH output of the stacked encoder given an Lxd_input
        input sequence.
        Args:
             x (float32): input sequence (L, d_input)
        Returns:
            output sequence (float32): (L, d_model)
        """
        self.sow("intermediates", "pre_encoder", x)
        x = self.encoder(x)
        x = self.topk_op(x)
        self.sow("intermediates", "encoder_output", x)
        for layer in self.layers:
            x = layer(x)
        return x


QBatchEncoderModel = nn.vmap(
    QStackedEncoderModel,
    in_axes=(0, 0),
    out_axes=0,
    variable_axes={
        "params": None,
        "dropout": None,
        "batch_stats": None,
        "cache": 0,
        "prime": None,
        "intermediates": 0,
    },
    split_rngs={"params": False, "dropout": True},
    axis_name="batch",
)


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


# Here we call vmap to parallelize across a batch of input sequences
batch_masked_meanpool = jax.vmap(masked_meanpool)


class QClassificationModel(BaseModel):
    d_output: int = None
    padded: bool = False
    mode: str = "pool"

    def setup(self):
        """
        Initializes the S5 stacked encoder and a linear decoder.
        """
        assert (
            self.d_output is not None
        ), "d_output must be set for classification tasks"
        self.encoder = QStackedEncoderModel(
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
            use_batchnorm_scale=self.use_batchnorm_scale,
            use_batchnorm_bias=self.use_batchnorm_bias,
            topk=self.topk,
            approx_topk=self.approx_topk,
        )
        # NOTE: nn.Dense calls dot_general(activation, weights)
        q_bits_aw = (
            self.q_config.non_ssm_act_precision,
            self.q_config.non_ssm_precision,
        )
        if self.q_config.static_quant:
            assert (
                self.q_config.calibrating is not None
            ), "Calibrating must be set if static_quant is True"
            self.decoder = QuantizedDense(
                self.d_output,
                a_bits=q_bits_aw[0],
                w_bits=q_bits_aw[1],
                calibrating=self.q_config.calibrating,
            )
        else:
            dot = aqt.AqtDotGeneral(q_dot_maybe(*q_bits_aw, return_cfg=True))
            self.decoder = nn.Dense(self.d_output, dot_general=dot)

    def __call__(self, x, integration_timesteps):
        """
        Compute the size d_output log softmax output given a
        Lxd_input input sequence.
        Args:
             x (float32): input sequence (L, d_input)
        Returns:
            output (float32): (d_output)
        """
        if self.padded:
            x, length = x  # input consists of data and prepadded seq lens

        x = self.encoder(x, integration_timesteps)
        if self.mode in ["pool"]:
            # Perform mean pooling across time
            if self.padded:
                x = masked_meanpool(x, length)
            else:
                x = np.mean(x, axis=0)

        elif self.mode in ["last"]:
            # Just take the last state
            if self.padded:
                raise NotImplementedError(
                    "Mode must be in ['pool'] for self.padded=True (for" " now...)"
                )
            else:
                x = x[-1]
        else:
            raise NotImplementedError("Mode must be in ['pool', 'last]")

        x = self.decoder(x)
        return nn.log_softmax(x, axis=-1)


QBatchClassificationModel = nn.vmap(
    QClassificationModel,
    in_axes=(0, 0),
    out_axes=0,
    variable_axes={
        "params": None,
        "dropout": None,
        "batch_stats": None,
        "cache": 0,
        "prime": None,
        "intermediates": 0,
    },
    split_rngs={"params": False, "dropout": True},
    axis_name="batch",
)


class QRetrievalDecoder(nn.Module):
    """
    Defines the decoder to be used for document matching tasks,
    e.g. the AAN task. This is defined as in the S4 paper where we apply
    an MLP to a set of 4 features. The features are computed as described in
    Tay et al 2020 https://arxiv.org/pdf/2011.04006.pdf.
    Args:
        d_output    (int32):    the output dimension, i.e. the number of classes
        d_model     (int32):    this is the feature size of the layer inputs and outputs
                    we usually refer to this size as H
        q_bits_aw   (int?, int?): quantization precision for activations and weights
    """

    d_model: int
    d_output: int
    relufication: bool = False
    q_config: QuantizationConfig = QuantizationConfig.none()

    def setup(self):
        """
        Initializes 2 dense layers to be used for the MLP.
        """
        # NOTE: nn.Dense calls dot_general(activation, weights)
        q_bits_aw = (
            self.q_config.non_ssm_act_precision,
            self.q_config.non_ssm_precision,
        )
        if self.q_config.static_quant:
            assert (
                self.q_config.calibrating is not None
            ), "Calibrating must be set if static_quant is True"
            dense_cls = partial(
                QuantizedDense,
                a_bits=q_bits_aw[0],
                w_bits=q_bits_aw[1],
                calibrating=self.q_config.calibrating,
            )
            self.layer1 = dense_cls(self.d_model)
            self.layer2 = dense_cls(self.d_output)
        else:
            dot = aqt.AqtDotGeneral(q_dot_maybe(*q_bits_aw, return_cfg=True))
            self.layer1 = nn.Dense(self.d_model, dot_general=dot)
            self.layer2 = nn.Dense(self.d_output, dot_general=dot)

    def __call__(self, x):
        """
        Computes the input to be used for the softmax function given a set of
        4 features. Note this function operates directly on the batch size.
        Args:
             x (float32): features (bsz, 4*d_model)
        Returns:
            output (float32): (bsz, d_output)
        """
        x = self.layer1(x)
        if self.relufication:
            x = nn.relu(x)
        else:
            x = nn.gelu(x)
        return self.layer2(x)


BatchRetrievalDecoder = nn.vmap(
    QRetrievalDecoder,
    in_axes=0,
    out_axes=0,
    variable_axes={"params": None, "intermediates": 0},
    split_rngs={"params": False},
)


class QRetrievalModel(BaseModel):
    d_output: int = None
    padded: bool = False

    def setup(self):
        """
        Initializes the S5 stacked encoder and the retrieval decoder. Note that here we
        vmap over the stacked encoder model to work well with the retrieval decoder that
        operates directly on the batch.
        """
        assert self.d_output is not None, "d_output must be set for retrieval tasks"
        self.encoder = QBatchEncoderModel(
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
            use_batchnorm_scale=self.use_batchnorm_scale,
            use_batchnorm_bias=self.use_batchnorm_bias,
            topk=self.topk,
            approx_topk=self.approx_topk,
        )
        self.decoder = BatchRetrievalDecoder(
            d_model=self.d_model,
            d_output=self.d_output,
            q_config=self.q_config,
        )

    def __call__(
        self, input, integration_timesteps
    ):  # input is a tuple of x and lengths
        """
        Compute the size d_output log softmax output given a
        Lxd_input input sequence. The encoded features are constructed as in
        Tay et al 2020 https://arxiv.org/pdf/2011.04006.pdf.
        Args:
             input (float32, int32): tuple of input sequence and prepadded sequence lengths
                input sequence is of shape (2*bsz, L, d_input) (includes both documents) and
                lengths is (2*bsz,)
        Returns:
            output (float32): (d_output)
        """
        x, lengths = input  # x is 2*bsz*seq_len*in_dim, lengths is: (2*bsz,)
        x = self.encoder(
            x, integration_timesteps
        )  # The output is: 2*bszxseq_lenxd_model
        outs = batch_masked_meanpool(x, lengths)  # Avg non-padded values: 2*bszxd_model
        outs0, outs1 = np.split(outs, 2)  # each encoded_i is bszxd_model
        features = np.concatenate(
            [outs0, outs1, outs0 - outs1, outs0 * outs1], axis=-1
        )  # bszx4*d_model
        out = self.decoder(features)
        return nn.log_softmax(out, axis=-1)


class QRegressionModel(BaseModel):
    d_output: int = None
    padded: bool = False

    def setup(self):
        """
        Initializes the S5 stacked encoder and a linear decoder.
        """
        assert self.d_output is not None, "d_output must be set for regression tasks"
        self.encoder = QStackedEncoderModel(
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
            use_batchnorm_scale=self.use_batchnorm_scale,
            use_batchnorm_bias=self.use_batchnorm_bias,
            topk=self.topk,
            approx_topk=self.approx_topk,
        )
        # NOTE: nn.Dense calls dot_general(activation, weights)
        q_bits_aw = (
            self.q_config.non_ssm_act_precision,
            self.q_config.non_ssm_precision,
        )
        if self.q_config.static_quant:
            assert (
                self.q_config.calibrating is not None
            ), "Calibrating must be set if static_quant is True"
            self.decoder = QuantizedDense(
                self.d_output,
                a_bits=q_bits_aw[0],
                w_bits=q_bits_aw[1],
                calibrating=self.q_config.calibrating,
            )
        else:
            dot = aqt.AqtDotGeneral(q_dot_maybe(*q_bits_aw, return_cfg=True))
            self.decoder = nn.Dense(self.d_output, dot_general=dot)

    def __call__(self, x, integration_timesteps):
        """
        Compute the size d_output log softmax output given a
        Lxd_input input sequence.
        Args:
             x (float32): input sequence (L, d_input)
        Returns:
            output (float32): (d_output)
        """
        if self.padded:
            x, _ = x  # input consists of data and prepadded seq lens

        if self.quant_input is not None:
            x = quant_input_fn(x, self.quant_input)
            # self.sow("intermediates", "quant_input", x)
        x = self.encoder(x, integration_timesteps)
        self.sow("intermediates", "pre_decoder", x)
        return self.decoder(x)


QBatchRegressionModel = nn.vmap(
    QRegressionModel,
    in_axes=(0, 0),
    out_axes=0,
    variable_axes={
        "params": None,
        "dropout": None,
        "batch_stats": None,
        "cache": 0,
        "prime": None,
        "intermediates": 0,
    },
    split_rngs={"params": False, "dropout": True},
    axis_name="batch",
)
