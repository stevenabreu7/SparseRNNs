import json
import logging
import os
import pickle
import time
from functools import partial

import pathlib
import jax
import jax.numpy as np
from tqdm import tqdm

from sparseRNNs.dataloaders import Datasets
from sparseRNNs.fxparray import RoundingMode, fxp_from_fp
from sparseRNNs.fxpmodel import FxpRegressionModel, FxpS5Config, FxpSSM
from sparseRNNs.fxpreporter import Reporter
from sparseRNNs.fxputils import (add_target_bits_exp, create_fxp_qconfig,
                                 load_modeldict, load_pytree,
                                 manually_overwrite)
from sparseRNNs.train_helpers import si_snr_jax, stft_mixer, stft_splitter
from sparseRNNs.utils.logging import logger, setup_experiment_logging_fns
from sparseRNNs.utils.quantization import QuantizationConfig


def run_validation(
    args,
    model,
    fxp_qconfig: dict,
    n_batches=None,
):
    """Runs validation on the provided fixed-point model.

    Args:
        model: The fixed-point model instance to validate.
        batchsize: The batch size for processing the validation data.
        fxp_qconfig: The fixed-point quantization configuration dictionary.
        n_batches: Optional number of batches to process for validation.
                   If None, processes the entire validation set.

    Returns:
        A tuple containing:
            - losses (np.ndarray): Array of loss values for each sample.
            - snrs (np.ndarray): Array of SI-SNR scores for each sample.
    """
    init_rng = jax.random.PRNGKey(0)
    create_dataset_fn = Datasets["ndns"]
    init_rng, _ = jax.random.split(init_rng, num=2)
    _, valloader, _, _, n_classes, seq_len, in_dim, _ = create_dataset_fn(
        bsz=args.bsz, 
    )
    losses, snrs = [], []
    batch_idx = 0
    for batch in tqdm(valloader):
        noisy, clean, noise = batch
        noisy_np = noisy.numpy()
        clean_np = clean.numpy()
        noise_np = noise.numpy()
        noisy = np.array(noisy_np)
        clean = np.array(clean_np)
        noise = np.array(noise_np)
        # batch_size = len(noisy)
        # integration_times = np.ones((batch_size, seq_len))
        noisy_mag, noisy_phase = stft_splitter(noisy)
        clean_mag, clean_phase = stft_splitter(clean)
        stft_mag_mean = 0.0007
        noisy_mag_mean_sub = noisy_mag - stft_mag_mean
        noisy_mag_mean_sub = np.transpose(noisy_mag_mean_sub, (0, 2, 1))
        x = noisy_mag_mean_sub
        fxp_x = fxp_from_fp(
            x,
            bits=fxp_qconfig["encoder"]["inp_bits"],
            exp=fxp_qconfig["encoder"]["inp_exp"],
            signed=True,
            round_mode=RoundingMode.FLOOR,
        )
        y = model(fxp_x)
        mask = y.to_float()
        mask = np.transpose(mask, (0, 2, 1))
        cleaned_mag = noisy_mag * (1.0 + mask)
        # cleaned_mag = noisy_mag * (mask)
        cleaned = stft_mixer(cleaned_mag, noisy_phase)
        si_snr_score = si_snr_jax(cleaned, clean)
        lam = 0.001
        losses_i = lam * np.mean((cleaned_mag - clean_mag) ** 2, axis=(1, 2)) + (
            100 - si_snr_score
        )
        losses.append(losses_i)
        snrs.append(si_snr_score)
        batch_idx += 1
        if n_batches is not None and batch_idx > n_batches:
            logger.warning(f"STOPPING AFTER {n_batches} BATCHES")
            break
    losses = np.concatenate(losses)
    snrs = np.concatenate(snrs)
    return losses, snrs


def parse_args():
    """Parses command-line arguments for the FXP inference and verification script."""
    import argparse

    parser = argparse.ArgumentParser(description="Run FXP inference and verification.")
    # Experiment tracking
    parser.add_argument(
        "--wandb_entity", 
        type=str, 
        help="Wandb entity (optional)"
    )
    parser.add_argument(
        "--wandb_project", 
        type=str, 
        help="Wandb project (optional)"
    )
    parser.add_argument(
        "--wandb-name",
        type=str, 
        help="Wandb run name (optional)"
    )
    # Directories and run names
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/export/work/sabreu/squash/checkpoints",
        help="Directory containing checkpoints.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Directory containing data. Defaults to checkpoint_dir if not set.",
    )
    parser.add_argument(
        "--load_run_name",
        type=str,
        default="ndns_sparse_simplebn_relu_inp14_qatw8a16_1.0",
        help="Name of the run to load checkpoints from.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="ndns_sparse_simplebn_relu_inp14_qatw8a16_1.0",
        help="Name of the current run.",
    )
    # File names to be loaded
    parser.add_argument(
        "--params_fname",
        type=str,
        default="sc_calibrated_params.pkl",
        help="Filename for calibrated parameters.",
    )
    parser.add_argument(
        "--stats_fname",
        type=str,
        default="sc_cal_stats.pkl",
        help="Filename for calibration statistics.",
    )
    parser.add_argument(
        "--inputs_fname",
        type=str,
        default="inputs.npy",
        help="Filename for input data.",
    )
    parser.add_argument(
        "--activations_fname",
        type=str,
        default="activations_fp.pkl",
        help="Filename for activation data.",
    )
    # Recipe for the task + fxp quantization
    quant_choices = ["w8a8", "w8a16", "allw8a8", "allw8a16"]
    parser.add_argument(
        "--quantization",
        type=str,
        default="w8a16",
        choices=quant_choices,
        help="Quantization scheme to use.",
    )
    parser.add_argument(
        "--recipe",
        type=str,
        help="Path to the recipe file.",
        default="../recipes/ndns.json",
    )
    parser.add_argument(
        "--fuse_batchnorm",
        action="store_true",
        help="Fuse batch normalization layers.",
    )
    parser.add_argument(
        "--relufication",
        action="store_true",
        help="Apply relufication.",
    )
    parser.add_argument(
        "--bn_eps",
        type=float,
        default=1e-5,
        help="Epsilon value for batch normalization.",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=None,
        help="Sequence length for processing.",
    )
    parser.add_argument(
        "--plot_maxlen",
        type=int,
        default=100,
        help="Maximum length for plotting.",
    )
    parser.add_argument(
        "--plot_toffset",
        type=int,
        default=0,
        help="Time offset for plotting.",
    )
    parser.add_argument(
        "--plot_nlines",
        type=int,
        default=100,
        help="Number of lines for plotting.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode.",
    )
    parser.add_argument(
        "--disable_jit",
        action="store_true",
        help="Disable JIT compilation.",
    )
    parser.add_argument(
        "--bsz",
        type=int,
        default=None,
        help="Batch size for processing.",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Export the model.",
    )
    parser.add_argument(
        "--mlflow_run_id",
        type=str,
        default=None,
        help="MLflow run ID.",
    )
    parser.add_argument(
        "--checkpoint_restore_step",
        type=int,
        default=None,
        help="Specific step to restore checkpoint from.",
    )
    parser.add_argument(
        "--separate_exponents",
        action="store_true",
        help="Use separate exponents for quantization.",
    )
    parser.add_argument(
        "--datafolder_suffix",
        type=str,
        default="",
        help="Suffix for the data folder.",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.checkpoint_restore_step is not None:
        suffix = f"_{args.datafolder_suffix}" if args.datafolder_suffix != "" else ""
        convert_suffix = f"_{args.checkpoint_restore_step}"
        folder = os.path.join(args.checkpoint_dir, args.load_run_name, f"convert{convert_suffix}")
        if not os.path.exists(folder):
            folder = os.path.join(args.checkpoint_dir, args.load_run_name)
        data_dir = args.data_dir if args.data_dir is not None else args.checkpoint_dir
        data_folder = os.path.join(data_dir, args.load_run_name, f"fxp{suffix}")
    else:
        suffix = f"_{args.datafolder_suffix}" if args.datafolder_suffix != "" else ""
        folder = os.path.join(args.checkpoint_dir, args.load_run_name, f"convert")
        if not os.path.exists(folder):
            folder = os.path.join(args.checkpoint_dir, args.load_run_name)
        data_dir = args.data_dir if args.data_dir is not None else args.checkpoint_dir
        data_folder = os.path.join(data_dir, args.load_run_name, f"fxp{suffix}")
    args.data_dir = data_dir
    logger.debug(f"folder: {folder}")
    logger.debug(f"data_folder: {data_folder}")

    run_name = args.run_name
    params_fname = f"{folder}/{args.params_fname}"
    stats_fname = f"{folder}/{args.stats_fname}"
    inputs_fname = f"{folder}/{args.inputs_fname}"
    activations_fname = f"{folder}/{args.activations_fname}"
    current_path = pathlib.Path(__file__).parent.resolve()
    recipe_path = os.path.join(current_path, "..", "recipes", args.recipe)
    with open(recipe_path, "r") as f:
        recipe = json.load(f)
    precisions = dict(
        non_ssm_w=8,
        non_ssm_b=16 if "a16" in args.quantization else 8,
        non_ssm_act=16 if "a16" in args.quantization else 8,
        ssm_w=8,
        ssm_act=16 if "a16" in args.quantization else 8,
    )
    seq_len = args.seq_len
    fuse_batchnorm = args.fuse_batchnorm
    relufication = args.relufication
    bn_eps = args.bn_eps
    export = args.export
    blockwise_shared_exp = not args.separate_exponents
    if args.debug:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
    if args.disable_jit:
        jax.config.update("jax_disable_jit", True)
    
    args.bsz = recipe['bsz'] if args.bsz is None else args.bsz 

    # TODO: implement manual overwrites as command line args
    manual_overwrite = {
        "model_attr": {
            # 'encoder.encoder.out_exp': 14,
        },
        "fxp_qconfig": {
            # "blocks.ssm.activations.x_re.bits": 24,
            # "blocks.ssm.activations.x_re.exp": 24,
            # "blocks.ssm.activations.x_im.bits": 24,
            # "blocks.ssm.activations.x_im.exp": 24,
        },
    }

    if args.bsz > 1 and not args.export:
        # Setup logging for experiment tracking, with wandb
        log_metrics, log_best_metrics, log_artifacts, end_logging = (
            setup_experiment_logging_fns(args)
        )
    else: 
        log_metrics, log_best_metrics, log_artifacts, end_logging = None, None, None, None

    # create modeldict from params and stats
    modeldict = load_modeldict(params_fname, stats_fname)

    # create fxp quantization configuration from modeldict
    os.makedirs("./tmp", exist_ok=True)
    if blockwise_shared_exp:
        _, fxp_qconfig = create_fxp_qconfig(
            modeldict, agg="max", store_json=True, json_store_folder="./tmp"
        )
    else:
        fxp_qconfig = create_fxp_qconfig(
            modeldict, agg=None, store_json=True, json_store_folder="./tmp"
        )
    fxp_qconfig = add_target_bits_exp(
        modeldict, fxp_qconfig, precisions, shared_exp=blockwise_shared_exp
    )

    # manually add precision to different parts of the model
    for key, val in manual_overwrite["fxp_qconfig"].items():
        idx_fxp_qconfig = fxp_qconfig
        key_list = key.split(".")
        for k in key_list[:-1]:
            idx_fxp_qconfig = idx_fxp_qconfig[k]
        prev_val = idx_fxp_qconfig[key_list[-1]]
        idx_fxp_qconfig[key_list[-1]] = val
        logger.info(f"MANUAL OVERWRITE: {key} <= {val} (was: {prev_val})")

    # setup configuration
    #####################
    current_path = pathlib.Path(__file__).parent.resolve()
    recipe_path = os.path.join(current_path, "..", "recipes", args.recipe)
    with open(recipe_path, "r") as f:
        recipe = json.load(f)
    n_layers = (
        max(
            [
                int(e.split("_")[1])
                for e in modeldict["encoder"].keys()
                if e.startswith("layers_")
            ]
        )
        + 1
    )
    d_model = modeldict["encoder"]["encoder"]["kernel"].shape[-1]
    d_ssm = modeldict["encoder"]["layers_0"]["mixer"]["Lambda_re"].shape[0]
    d_input = modeldict["encoder"]["encoder"]["kernel"].shape[0]
    d_output = modeldict["decoder"]["kernel"].shape[-1]
    assert (
        n_layers == recipe["n_layers"]
    ), f"unexpected number of layers for `{recipe['dataset']}`"
    padded = recipe["dataset"] in [
        "imdb-classification",
        "listops-classification",
        "aan-classification",
    ]
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
        glu_variant=recipe["glu_variant"],
        bn_momentum=0.95,  # NOTE: default, never changed
        step_rescale=1.0,  # NOTE: default, never changed
        prenorm=recipe["prenorm"],  # fixed, required for fxp
        batchnorm=recipe["batchnorm"],  # fixed, required for fxp
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


    if args.bsz > 1 or args.export:
        run_inference(args, model_cls, fxp_qconfig, manual_overwrite, end_logging, log_best_metrics)
    else:
        run_verification(args, model_cls, fxp_qconfig, manual_overwrite, inputs_fname, activations_fname, precisions, cfg)
    

def run_inference(args, model_cls, fxp_qconfig, manual_overwrite, end_logging, log_best_metrics):
    model = model_cls(store_intermediates=args.export)
    manually_overwrite(model, manual_overwrite)
    loss, snr = run_validation(
        args, model, fxp_qconfig, n_batches=1 if args.export else None
    )
    logger.info(f"Avg loss: {loss.mean():.5f}")
    logger.info(f"Avg SNR:  {snr.mean():.5f}")
    data_folder = args.data_dir

    if args.export:
        logger.info("exporting fxp model and activations...")
        data = model.export()
        if os.path.exists(data_folder):
            logger.warning("folder already exists, overwriting...")
        os.makedirs(data_folder, exist_ok=True)
        with open(os.path.join(data_folder, "fxpmodel_modeldict.pkl"), "wb") as f:
            pickle.dump(model.modeldict, f)
        with open(os.path.join(data_folder, "fxpmodel_fxp_qconfig.pkl"), "wb") as f:
            pickle.dump(model.fxp_qconfig, f)
        with open(os.path.join(data_folder, "fxpmodel_io.pkl"), "wb") as f:
            io = dict(
                input=data["intermediates"]["encoder"]["pre_encoder"],
                output=data["intermediates"]["decoder"]["__call__"],
            )
            pickle.dump(io, f)
        with open(os.path.join(data_folder, "fxpmodel.pkl"), "wb") as f:
            pickle.dump({k: v for k, v in data.items() if k != "intermediates"}, f)
        with open(os.path.join(data_folder, "fxpmodel_activations.pkl"), "wb") as f:
            pickle.dump({k: v for k, v in data.items() if k == "intermediates"}, f)
    else:
        logger.info("Storing metrics to MLFlow")
        log_best_metrics(
            {
                "Best Val Loss - fxp": loss.mean().item(),
                "Best Val Acc - fxp": snr.mean().item(),
            },
            step=0,
        )
        with open(os.path.join(data_folder, "val_metrics.json"), "w") as f:
            json.dump(
                {
                    "Val Loss - fxp": loss.mean().item(),
                    "Val Acc - fxp": snr.mean().item(),
                },
                f,
            )
        end_logging()
        time.sleep(10)


def run_verification(
        args,
        model_cls,
        fxp_qconfig, manual_overwrite, inputs_fname, activations_fname, precisions, cfg):
    # setup up input
    logger.info("loading inputs")
    inputs = np.load(inputs_fname)
    if args.bsz > inputs.shape[0]:
        n_repeat = int(np.ceil(args.bsz / inputs.shape[0]).item())
        inputs = np.repeat(inputs, n_repeat, axis=0)
        inputs = inputs[:args.bsz, :, :]
    elif 1 < args.bsz < inputs.shape[0]:
        inputs = inputs[:args.bsz, :, :]
    elif args.bsz == 1:
        inputs = inputs[0, :, :]
    else:
        raise ValueError(f"unexpected args.bsz: {args.bsz}")
    args.seq_len = args.seq_len if args.seq_len is not None else inputs.shape[0]
    inputs = inputs[:args.seq_len, :]
    fxp_inputs = fxp_from_fp(
        inputs,
        bits=fxp_qconfig["encoder"]["inp_bits"],
        exp=fxp_qconfig["encoder"]["inp_exp"],
        signed=True,
        round_mode=RoundingMode.FLOOR,
    )

    model = model_cls(store_intermediates=True)
    manually_overwrite(model, manual_overwrite)

    logger.info("starting forward pass...")
    tstart = time.time()
    y = model(fxp_inputs, integration_timesteps=10)
    tend = time.time()
    logger.info(f"forward pass took {tend - tstart:.3f} seconds")

    logger.info("starting verification...")
    verification_results = []
    folder_name = os.path.join(args.checkpoint_dir, args.run_name, "verification")
    os.makedirs(folder_name, exist_ok=True)
    reporter = Reporter(
        precisions,
        fxp_qconfig,
        cfg,
        manual_overwrite,
        folder_name,
        plot_nlines=args.plot_nlines,
        plot_maxlen=args.plot_maxlen,
        plot_toffset=args.plot_toffset,
    )

    logger.info("loading activations")
    activations = load_pytree(activations_fname)

    logger.debug("inputs")
    logger.debug(f"effective exponent: {fxp_inputs.exp}, bits: {fxp_inputs.bits}")
    name = "inputs"
    xrec = fxp_inputs.to_float()
    xhat = inputs
    reporter.add_block_raw(name, xhat=xhat, xrec=xrec)

    for layer_idx in range(cfg.n_layers):
        if layer_idx == 0:
            name = "encoder.encoder (post-relu)"
        else:
            name = f"encoder.layers_{layer_idx}.input"
        xrec = (
            model.encoder.seq_layers[layer_idx].intermediates["ssm_input"][0].to_float()
        )
        xhat = activations["encoder"][f"layers_{layer_idx}"]["input"][0][0, :args.seq_len]
        reporter.add_block_raw(name, xhat=xhat, xrec=xrec)

        if model.encoder.seq_layers[layer_idx].prenorm:
            name = f"encoder.layers_{layer_idx}.norm"
            xrec = (
                model.encoder.seq_layers[layer_idx]
                .intermediates["pre_s5"][0]
                .to_float()
            )
            xhat = activations["encoder"][f"layers_{layer_idx}"]["pre_s5"][0][
                0, :args.seq_len
            ]
            reporter.add_block_raw(name, xhat=xhat, xrec=xrec)

        # NOTE: we don't store B @ ut, so we calculate it from B and ut
        name = f"encoder.layers_{layer_idx}.mixer.Bu.calc_hat"
        xrec = (
            model.encoder.seq_layers[layer_idx]
            .mixer.intermediates["Bu_elements"][0]
            .to_float()
        )
        uhat = activations["encoder"][f"layers_{layer_idx}"]["pre_s5"][0][0, :args.seq_len]
        Bhat = activations["encoder"][f"layers_{layer_idx}"]["mixer"]["B_bar"][0][0]
        xcalc = uhat @ Bhat.T
        reporter.add_block_raw(
            name,
            xhat=xcalc,
            xrec=xrec,
            xhatname="float (calc)",
            xrecname="fxp",
        )

        name = f"encoder.layers_{layer_idx}.mixer.xt"
        xrec = (
            model.encoder.seq_layers[layer_idx]
            .mixer.intermediates["xs_relu"][0]
            .to_float()
        )
        xhat = activations["encoder"][f"layers_{layer_idx}"]["pre_C"][0][0, :args.seq_len]
        reporter.add_block_raw(
            name, xhat=xhat, xrec=xrec, xhatname="float", xrecname="fxp"
        )

        name = f"encoder.layers_{layer_idx}.mixer.yt"
        xrec = (
            model.encoder.seq_layers[layer_idx].mixer.intermediates["ys"][0].to_float()
        )
        xhat = activations["encoder"][f"layers_{layer_idx}"]["mixer"]["__call__"][0][0][
            0, :args.seq_len
        ]
        reporter.add_block_raw(
            name, xhat=xhat, xrec=xrec, xhatname="float", xrecname="fxp"
        )

        name = f"encoder.layers_{layer_idx}.mixer.pre_glu"
        xrec = (
            model.encoder.seq_layers[layer_idx].intermediates["pre_GLU"][0].to_float()
        )
        xhat = activations["encoder"][f"layers_{layer_idx}"]["pre_GLU"][0][0, :args.seq_len]
        reporter.add_block_raw(
            name, xhat=xhat, xrec=xrec, xhatname="float", xrecname="fxp"
        )

        name = f"encoder.layers_{layer_idx}.mixer.out2"
        xrec = (
            model.encoder.seq_layers[layer_idx]
            .out2.intermediates["__call__"][0]
            .to_float()
        )
        xhat = activations["encoder"][f"layers_{layer_idx}"]["out2"]["__call__"][0][
            0, :args.seq_len
        ]
        reporter.add_block_raw(
            name, xhat=xhat, xrec=xrec, xhatname="float", xrecname="fxp"
        )

        name = f"encoder.layers_{layer_idx}.mixer.out2_sigmoid"
        xrec = (
            model.encoder.seq_layers[layer_idx]
            .intermediates["out2_sigmoid"][0]
            .to_float()
        )
        out2hat = activations["encoder"][f"layers_{layer_idx}"]["out2"]["__call__"][0][
            0, :args.seq_len
        ]
        xhat = jax.nn.sigmoid(out2hat)
        reporter.add_block_raw(
            name, xhat=xhat, xrec=xrec, xhatname="float", xrecname="fxp"
        )

        name = f"encoder.layers_{layer_idx}.mixer.post_glu"
        xrec = (
            model.encoder.seq_layers[layer_idx].intermediates["post_GLU"][0].to_float()
        )
        if len(activations["encoder"][f"layers_{layer_idx}"]["drop"]["__call__"]) == 2:
            xhat = activations["encoder"][f"layers_{layer_idx}"]["drop"]["__call__"][1][
                0, :args.seq_len
            ]
        else:
            xhat = activations["encoder"][f"layers_{layer_idx}"]["pre_GLU"][0][
                0, :args.seq_len
            ]
        reporter.add_block_raw(
            name, xhat=xhat, xrec=xrec, xhatname="float", xrecname="fxp"
        )

        name = f"encoder.layers_{layer_idx}.mixer.residadd"
        xrec = (
            model.encoder.seq_layers[layer_idx].intermediates["residadd"][0].to_float()
        )
        post_glu_val = xhat.copy()
        block_input = activations["encoder"][f"layers_{layer_idx}"]["input"][0][
            0, :args.seq_len
        ]
        xhat = post_glu_val + block_input
        reporter.add_block_raw(
            name, xhat=xhat, xrec=xrec, xhatname="float (calc)", xrecname="fxp"
        )

        if not model.encoder.seq_layers[layer_idx].prenorm:
            name = f"encoder.layers_{layer_idx}.norm"
            xrec = (
                model.encoder.seq_layers[layer_idx]
                .intermediates["pre_s5"][0]
                .to_float()
            )
            xhat = activations["encoder"][f"layers_{layer_idx}"]["pre_s5"][0][
                0, :args.seq_len
            ]
            reporter.add_block_raw(name, xhat=xhat, xrec=xrec)

        name = f"encoder.layers_{layer_idx}.mixer.output"
        xrec = model.encoder.seq_layers[layer_idx].intermediates["output"][0].to_float()
        xhat = activations["encoder"][f"layers_{layer_idx}"]["__call__"][0][0, :args.seq_len]
        reporter.add_block_raw(
            name, xhat=xhat, xrec=xrec, xhatname="float", xrecname="fxp"
        )

    name = "decoder"
    xrec = model.intermediates["output"][0].to_float()
    xhat = activations["__call__"][0][0, :args.seq_len]
    reporter.add_block_raw(name, xhat=xhat, xrec=xrec, xhatname="float", xrecname="fxp")

    logger.info("Saving the report...")
    reporter.save()

    # model.export_intermediates()
    # -> activations_fxp.pkl


if __name__ == "__main__":
    main()