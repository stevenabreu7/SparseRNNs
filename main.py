import argparse
import json
import logging
import os
import pathlib

from sparseRNNs.convert import convert
from sparseRNNs.dataloaders.dataloading import Datasets
from sparseRNNs.model.layers import GLU_VARIANTS
from sparseRNNs.train import train
from sparseRNNs.utils.pruning import pruning_recipe_map
from sparseRNNs.utils.quantization import quantization_recipe_map
from sparseRNNs.utils.logging import logger


def get_parser():
    parser = argparse.ArgumentParser()

    # Experiment tracking
    parser.add_argument(
        "--wandb_entity", type=str, help="Wandb entity (optional)"
    )
    parser.add_argument(
        "--wandb_project", type=str, help="Wandb project (optional)"
    )
    parser.add_argument(
        "--wandb-name", type=str, help="Wandb run name (optional)"
    )
    parser.add_argument(
        "--log_grads",
        type=bool,
        default=False,
        help="whether to log gradient norms",
    )
    parser.add_argument(
        "--log_eigenvalues",
        type=bool,
        default=False,
        help="whether to log eigenvalue statistics",
    )
    parser.add_argument(
        "--log_act_sparsity",
        type=str,
        default="none",
        choices=["train", "val", "both", "none"],
        help="whether to log activation sparsity statistics",
    )

    # Model Checkpointing
    parser.add_argument(
        "--load_run_name",
        type=str,
        default=None,
        help="name of the checkpoint to load. if None, use the run_name.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help=(
            "name of this run (for wandb and checkpoint folder). if None, no"
            " checkpoints are made."
        ),
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help=(
            "parent folder where all checkpoints are stored. if None, no"
            " checkpoints are made."
        ),
    )
    parser.add_argument(
        "--checkpoint_interval_steps",
        type=int,
        default=1,
        help="how frequently to store checkpoints.",
    )
    parser.add_argument(
        "--checkpoint_restore_step",
        type=int,
        default=None,
        help="step to restore from checkpoint, default: best checkpoint",
    )

    # Data Parameters
    parser.add_argument(
        "--dir_name",
        type=str,
        default=None,
        help="name of directory where data is cached",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=Datasets.keys(),
        default="mnist-classification",
        help="dataset name",
    )

    # Model Parameters
    parser.add_argument(
        "--n_layers",
        type=int,
        default=6,
        help="Number of layers in the network",
    )
    parser.add_argument(
        "--d_model",
        type=int,
        default=128,
        help="Number of features, i.e. H, dimension of layer inputs/outputs",
    )
    parser.add_argument(
        "--ssm_size_base",
        type=int,
        default=256,
        help="SSM Latent size, i.e. P",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=8,
        help="How many blocks, J, to initialize with",
    )
    parser.add_argument(
        "--dim_scale",
        type=float,
        default=1.0,
        help="scale the dimension of the model",
    )
    parser.add_argument(
        "--batchnorm",
        type=bool,
        default=True,
        help="True=use batchnorm, False=use layernorm",
    )
    parser.add_argument(
        "--prenorm",
        type=bool,
        default=True,
        help="True=apply prenorm, False=apply postnorm",
    )
    parser.add_argument(
        "--glu_variant",
        type=str,
        default="none",
        choices=GLU_VARIANTS,
        help="Type of gated linear unit to use",
    )

    parser.add_argument(
        "--C_init",
        type=str,
        default="trunc_standard_normal",
        choices=["trunc_standard_normal", "lecun_normal", "complex_normal"],
        help=(
            "Options for initialization of C: \\trunc_standard_normal: sample"
            " from trunc. std. normal then multiply by V \\ lecun_normal"
            " sample from lecun normal, then multiply by V\\ complex_normal:"
            " sample directly from complex standard normal"
        ),
    )
    parser.add_argument(
        "--discretization",
        type=str,
        default="zoh",
        choices=["zoh", "bilinear"],
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="pool",
        choices=["pool", "last"],
        help=(
            "options: (for classification tasks) \\"
            " pool: mean pooling \\"
            "last: take last element"
        ),
    )
    parser.add_argument(
        "--conj_sym",
        type=bool,
        default=True,
        help="whether to enforce conjugate symmetry",
    )
    parser.add_argument(
        "--clip_eigs",
        type=bool,
        default=False,
        help="whether to enforce the left-half plane condition",
    )
    parser.add_argument(
        "--bidirectional",
        type=bool,
        default=False,
        help="whether to use bidirectional model",
    )
    parser.add_argument(
        "--dt_min",
        type=float,
        default=0.001,
        help="min value to sample initial timescale params from",
    )
    parser.add_argument(
        "--dt_max",
        type=float,
        default=0.1,
        help="max value to sample initial timescale params from",
    )

    # Optimization Parameters
    parser.add_argument(
        "--bn_momentum", type=float, default=0.95, help="batchnorm momentum"
    )
    parser.add_argument("--bsz", type=int, default=64, help="batch size")
    parser.add_argument(
        "--epochs", type=int, default=100, help="max number of epochs"
    )
    parser.add_argument(
        "--early_stop_patience",
        type=int,
        default=1000,
        help="number of epochs to continue training when val loss plateaus",
    )
    parser.add_argument(
        "--ssm_lr_base",
        type=float,
        default=1e-3,
        help="initial ssm learning rate",
    )
    parser.add_argument(
        "--lr_factor",
        type=float,
        default=1,
        help="global learning rate = lr_factor*ssm_lr_base",
    )
    parser.add_argument(
        "--dt_global",
        type=bool,
        default=False,
        help="Treat timescale parameter as global parameter or SSM parameter",
    )
    parser.add_argument(
        "--lr_min", type=float, default=0, help="minimum learning rate"
    )
    parser.add_argument(
        "--cosine_anneal",
        type=bool,
        default=True,
        help="whether to use cosine annealing schedule",
    )  # always used
    parser.add_argument(
        "--warmup_end", type=int, default=1, help="epoch to end linear warmup"
    )
    parser.add_argument(
        "--lr_patience",
        type=int,
        default=1000000,
        help=(
            "patience before decaying learning rate for"
            " lr_decay_on_val_plateau"
        ),
    )  # never used...
    parser.add_argument(
        "--reduce_factor",
        type=float,
        default=1.0,
        help="factor to decay learning rate for lr_decay_on_val_plateau",
    )  # never used...
    parser.add_argument(
        "--p_dropout", type=float, default=0.0, help="probability of dropout"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay value"
    )
    parser.add_argument(
        "--opt_config",
        type=str,
        default="standard",
        choices=[
            "standard",
            "BandCdecay",
            "BfastandCdecay",
            "noBCdecay",
            "qaft",
            "constant",
        ],
        help=(
            "Opt configurations: \\ standard:       no weight decay on B (ssm"
            " lr), weight decay on C (global lr) \\BandCdecay:     weight"
            " decay on B (ssm lr), weight decay on C (global lr)"
            " \\BfastandCdecay: weight decay on B (global lr), weight decay on"
            " C (global lr) \\noBCdecay:      no weight decay on B (ssm lr),"
            " no weight decay on C (ssm lr) \\qaft:          "
            " quantization-aware fine-tuning (standard, using SGD+momentum) \\"
        ),
    )
    parser.add_argument(
        "--grad_clip_threshold",
        type=float,
        default=None,
        help="max norm for gradient clipping.",
    )
    parser.add_argument(
        "--jax_seed", type=int, default=1919, help="seed randomness"
    )

    # Pruning
    parser.add_argument(
        "--pruning",
        type=str,
        choices=pruning_recipe_map.keys(),
        default="no_prune",
        help="Configuration for JaxPruner.",
    )

    # ReLUfication
    parser.add_argument(
        "--relufication",
        action="store_true",
        help=(
            "Applies ReLU after the linear encoding layer, the s5 hidden"
            " dynamics, and the GLU layer"
        ),
    )

    parser.add_argument(
        "--topk",
        type=float,
        default=1.0,
        help="Top-k sparsity instead of ReLUfication",
    )
    parser.add_argument(
        "--approx_topk",
        type=bool,
        default=False,
        help="Approximate top-k sparsity (if using topk)",
    )

    # Consolidation
    parser.add_argument(
        "--fuse_batchnorm_linear",
        action="store_true",
        help="Fuse batchnorm and linear layers",
    )
    parser.add_argument(
        "--batchnorm_use_scale",
        type=bool,
        default=True,
        help="Whether to use scale in batchnorm",
    )
    parser.add_argument(
        "--batchnorm_use_bias",
        type=bool,
        default=True,
        help="Whether to use bias in batchnorm",
    )

    # Quantization Parameters
    parser.add_argument(
        "--quantization",
        type=str,
        choices=quantization_recipe_map.keys(),
        default="none",
        help="Quantization recipe for QAT",
    )
    parser.add_argument(
        "--convert_quantization",
        type=str,
        choices=quantization_recipe_map.keys(),
        default="w8a16",
        help="Quantization recipe for post-training conversion",
    )
    parser.add_argument(
        "--validate_baseline",
        action="store_true",
        help="Validate baseline model",
    )
    parser.add_argument(
        "--validate_naive_scan",
        action="store_true",
        help="Validate model with naive scan",
    )
    parser.add_argument(
        "--validate_aqt",
        action="store_true",
        help="Validate the model with AQT",
    )
    parser.add_argument(
        "--train_aqt", action="store_true", help="Train the model with AQT"
    )
    parser.add_argument(
        "--aqt_qaft_lr_factor",
        type=float,
        default=0.1,
        help="Learning rate scaling factor for QAFT with AQT",
    )
    parser.add_argument(
        "--validate_static_quant",
        action="store_true",
        help="Validate the model with static quantization (frozen scales)",
    )
    parser.add_argument(
        "--train_static_quant",
        action="store_true",
        help="Train the model with static quantization (frozen scales)",
    )
    parser.add_argument(
        "--store_activations",
        action="store_true",
        help="Get and store intermediate activations",
    )
    parser.add_argument(
        "--quant_input",
        type=int,
        default=None,
        help=(
            "Quantization precision, i.e. spike_exp, for input (default: None,"
            " keep in float). For best results, use --quant_input 16"
        ),
    )

    # Debugging
    parser.add_argument(
        "--debug",
        action="store_true",
        help="whether to run in debug mode",
    )

    # Training Recipe
    parser.add_argument(
        "--recipe",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="whether to train the model",
    )
    parser.add_argument(
        "--dense_to_sparse",
        action="store_true",
        help="whether to sparsify (weights + activations) the model",
    )
    parser.add_argument(
        "--convert", action="store_true", help="whether to convert the model"
    )
    parser.add_argument(
        "--reset_optimizer",
        action="store_true",
        help="whether to reset the optimizer",
    )

    return parser


if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if args.recipe is not None:
        current_path = pathlib.Path(__file__).parent.resolve()
        path = os.path.join(current_path, "recipes", args.recipe)
        with open(path, "r") as f:
            config = json.load(f)
            for key, value in config.items():
                setattr(args, key, value)
        logger.info(f"✅ Loaded recipe from {path}")

    if args.dim_scale is not None:
        d_model_block = args.d_model / args.blocks
        ssm_size_block = args.ssm_size_base / args.blocks
        args.blocks = int(args.blocks * args.dim_scale)
        args.d_model = int(args.blocks * d_model_block)
        args.ssm_size_base = int(args.blocks * ssm_size_block)

    if args.train:
        train(args)

    if args.convert:
        convert(args)
