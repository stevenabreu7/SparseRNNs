import jax
import os
import pickle
import sys
import json
import time
import jax.numpy as np
from pprint import pprint
from functools import partial

from sparseRNNs.utils.logging import logger
from sparseRNNs.fxparray_np import FxpArray
from sparseRNNs.fxparray_np import fxp_from_fp
from sparseRNNs.fxparray_np import RoundingMode
from sparseRNNs.fxpmodel_np import FxpRegressionModel, FxpSSM, FxpS5Config
from sparseRNNs.fxputils import create_fxp_qconfig, add_target_bits_exp
from sparseRNNs.model.ssm import QuantizationConfig


def load_data(data_folder="data/model_1x"):

    # Create temporary alias for the old module name to fix pickle loading
    import sparseRNNs.fxparray_np as fxparray
    sys.modules['fxparray'] = fxparray

    with open(f"{data_folder}/fxpmodel_fxp_qconfig.pkl", "rb") as f:
        fxp_qconfig = pickle.load(f)

    with open(f"{data_folder}/fxpmodel_io.pkl", "rb") as f:
        io = pickle.load(f)

    with open(f"{data_folder}/fxpmodel.pkl", "rb") as f:
        fxpmodel = pickle.load(f)
        params = fxpmodel['params']
        qconfig_verbose = fxpmodel['qconfig']

    with open(f"{data_folder}/fxpmodel_modeldict.pkl", "rb") as f:
        modeldict = pickle.load(f)

    # Clean up the temporary alias
    del sys.modules['fxparray']

    return io, params, modeldict, fxp_qconfig

def calculate_num_params(params):
    pprint(jax.tree.map(lambda x: np.prod(np.array(x.shape)) if isinstance(x, jax.Array) and x.ndim > 0 else x, params))
    nel = jax.tree.map(lambda x: np.prod(np.array(x.shape)) if isinstance(x, jax.Array) and x.ndim > 0 else x, params)
    return sum(jax.tree.leaves(nel))

def assert_fxp_qconfig_valid(modeldict, fxp_qconfig):
    # create fxp quantization configuration from modeldict
    os.makedirs("./tmp", exist_ok=True)
    _, fxp_qconfig_ = create_fxp_qconfig(modeldict, agg="max", store_json=True, json_store_folder="./tmp")
    fxp_qconfig_ = add_target_bits_exp(modeldict, fxp_qconfig, precisions)
    assert all(jax.tree.leaves(jax.tree.map(lambda x, y: x == y, fxp_qconfig_, fxp_qconfig)))


quantization = "w8a16"
recipe_fname = "recipes/ndns.json"
precisions = dict(
    non_ssm_w = 8,
    non_ssm_b = 16,
    non_ssm_act = 16,
    ssm_w = 8,
    ssm_act = 16,
)
fuse_batchnorm = False
relufication = True
bn_eps = 1e-5
batchsize = 1


io, params, modeldict, fxp_qconfig = load_data()
# assert_fxp_qconfig_valid(modeldict, fxp_qconfig)

# setup configuration
with open(recipe_fname, "r") as f:
    recipe = json.load(f)
n_layers = max([int(e.split("_")[1]) for e in modeldict['encoder'].keys() if e.startswith('layers_')]) + 1
d_model = modeldict['encoder']['encoder']['kernel'].shape[-1]
d_ssm = modeldict['encoder']['layers_0']['mixer']['Lambda_re'].shape[0]
d_input = modeldict['encoder']['encoder']['kernel'].shape[0]
d_output = modeldict['decoder']['kernel'].shape[-1]
assert n_layers == recipe['n_layers'], f"unexpected number of layers for `{recipe['dataset']}`"
padded = recipe['dataset'] in ["imdb-classification", "listops-classification", "aan-classification"]

print(f"{n_layers=}")
print(f"{d_model=}")
print(f"{d_ssm=}")
print(f"{d_input=}")
print(f"{d_output=}")
print(f"{padded=}")
print(f"{recipe=}")

cfg = FxpS5Config(
    # shared configs
    q_config=QuantizationConfig.none(),  # NOTE: default
    relufication=relufication,  # fixed, required for fxp
    # SSM configs
    H=d_model,
    P=d_ssm,
    discretization="zoh",  # NOTE: default
    conj_sym=True,  # NOTE: default
    bidirectional=False,  # fixed
    associative_scan=False,  # fixed
    # model configs
    n_layers=n_layers,
    d_model=d_model,
    d_output=d_output,
    padded=padded,
    glu_variant=recipe['glu_variant'],
    bn_momentum=0.95,  # NOTE: default, never changed
    step_rescale=1.0,  # NOTE: default, never changed
    prenorm=recipe['prenorm'],  # fixed, required for fxp
    batchnorm=recipe['batchnorm'],  # fixed, required for fxp
    dropout=0.0,  # fixed, required for fxp
    training=False,  # fixed, required for fxp
    fuse_batchnorm_linear=fuse_batchnorm,
)

# setup model classes
mixer_cls = FxpSSM.init_fn(
    H=cfg.H,
    P=cfg.P,
    discretization=cfg.discretization,
    conj_sym=cfg.conj_sym,
    q_config=cfg.q_config,
    bidirectional=cfg.bidirectional,
    relufication=cfg.relufication,
    associative_scan=cfg.associative_scan,
)
model_cls = partial(
    FxpRegressionModel,
    modeldict=modeldict,
    fxp_qconfig=fxp_qconfig,
    scope="model",
    mixer_cls=mixer_cls,
    n_layers=cfg.n_layers,
    d_model=cfg.d_model,
    batchnorm=cfg.batchnorm,
    prenorm=cfg.prenorm,
    bn_momentum=cfg.bn_momentum,
    glu_variant=cfg.glu_variant,
    step_rescale=cfg.step_rescale,
    relufication=cfg.relufication,
    fuse_batchnorm_linear=cfg.fuse_batchnorm_linear,
    q_config=cfg.q_config,
    dropout=cfg.dropout,
    training=cfg.training,
    d_output=cfg.d_output,
    padded=cfg.padded,
)

# setup up input
logger.info("loading inputs")
inputs = io["input"][0]
seq_len = inputs.shape[0]
inputs = inputs[:seq_len, :]
# breakpoint()
fxp_inputs = inputs if isinstance(inputs, FxpArray) else fxp_from_fp(
    inputs,
    bits=fxp_qconfig['encoder']['inp_bits'],
    exp=fxp_qconfig['encoder']['inp_exp'],
    signed=True,
    round_mode=RoundingMode.FLOOR,
)

# Set up model
model = model_cls(store_intermediates=True)

# Do forward pass
logger.info("starting forward pass...")
tstart = time.time()
y = model(fxp_inputs, integration_timesteps=10)
tend = time.time()
logger.info(f"forward pass took {tend - tstart:.3f} seconds")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5), dpi=300)
plt.title("Expected vs. actual output (first sample, first dimension)")
plt.plot(y[:, 0].data, label="Actual")
plt.plot(io["output"][0, :, 0].data, label="Expected")
plt.legend()
plt.show()
