import json
import os
import pickle
from typing import Union

import jax
import jax.numpy as np

from sparseRNNs.utils.logging import logger


def _move_scales_to_params(params: dict, stats: dict) -> dict:
    """
    Merge stats into params, making sure matching paths have the same data.
    If data does not match, params is taken.
    """

    def merge_pytrees(p, s, path=()):
        if isinstance(p, dict) and isinstance(s, dict):
            merged = {}
            all_keys = set(p.keys()).union(s.keys())
            for key in all_keys:
                sub_path = path + (key,)
                p_value = p.get(key, None)
                s_value = s.get(key, None)
                if p_value is not None and s_value is not None:
                    merged[key] = merge_pytrees(p_value, s_value, sub_path)
                elif p_value is not None:
                    merged[key] = p_value
                else:
                    merged[key] = s_value
        elif isinstance(p, (list, tuple)) and isinstance(s, (list, tuple)):
            max_length = max(len(p), len(s))
            merged_list = []
            for i in range(max_length):
                sub_path = path + (i,)
                p_value = p[i] if i < len(p) else None
                s_value = s[i] if i < len(s) else None
                if p_value is not None and s_value is not None:
                    merged_list.append(merge_pytrees(p_value, s_value, sub_path))
                elif p_value is not None:
                    merged_list.append(p_value)
                else:
                    merged_list.append(s_value)
            if isinstance(p, tuple):
                merged = tuple(merged_list)
            else:
                merged = merged_list
        else:
            # Leaf nodes or different types
            if p == s:
                merged = p
            else:
                if p is not None:
                    print(
                        f"Warning at path {path}: Values differ. Using params" " value."
                    )
                    merged = p
                else:
                    merged = s
        return merged

    merged_params = merge_pytrees(params, stats)
    return merged_params


def compute_integer_bits(pytree, key=""):
    if isinstance(pytree, dict) and key.endswith("observer"):
        prefix = list(pytree.keys())[0].split("_")[0]
        prefix = "" if prefix == "observer" else f"{prefix}_"
        minmax = np.array(
            [pytree[f"{prefix}observer_min"], pytree[f"{prefix}observer_max"]]
        )
        pytree["absmax"] = np.max(np.abs(minmax))
        pytree["intbits"] = (np.ceil(np.log2(pytree["absmax"]))).astype(int)
        return pytree
    elif isinstance(pytree, dict):
        return {k: compute_integer_bits(v, k) for k, v in pytree.items()}
    else:
        return pytree


def remove_norm_from_stats(pytree, key_to_remove="norm", subkeys=["mean", "var"]):
    if isinstance(pytree, dict):
        is_leaf = lambda k: not hasattr(pytree[k], "keys")
        has_norm_leaves = lambda k: set(pytree[k].keys()) == set(subkeys)
        has_norm_key = lambda k: k != key_to_remove
        keep = lambda k: is_leaf(k) or (has_norm_key(k) and not has_norm_leaves(k))
        return {
            k: remove_norm_from_stats(v, key_to_remove)
            for k, v in pytree.items()
            if keep(k)
        }
    else:
        return pytree


def load_pytree(fname):
    with open(fname, "rb") as f:
        return pickle.load(f)


def index_tree(tree: dict, keypath: Union[list[str], str]):
    if isinstance(keypath, str):
        keypath = keypath.split("/")
        for key in keypath:
            tree = tree[key]
    elif isinstance(keypath, list):
        for key in keypath:
            tree = tree[key]
    else:
        raise ValueError("keypath must be a list or a string")
    return tree


########################################################################
# Utils for creating fxp quantization configurations
########################################################################


def load_modeldict(params_fname: str, stats_fname: str):
    params = load_pytree(params_fname)
    stats = load_pytree(stats_fname)

    # merge stats into params (make sure matching paths have the same data, prio: params)
    modeldict = _move_scales_to_params(params, stats)
    # compute log of scale parameters -> # fraction bits
    modeldict = jax.tree_util.tree_map_with_path(
        lambda path, x: np.log2(x) if "scale" in str(path[-1]) else x,
        modeldict,
    )
    # based on minmax, compute integer bits
    modeldict = compute_integer_bits(modeldict)
    return modeldict


def get_intbits(absmax: float):
    log2 = np.log2(absmax)
    bits = int(np.ceil(np.log2(absmax)).item())
    bits = max(0, bits)
    bits += 1 if np.round(log2) == log2 else 0
    return bits


def get_sign(xmin, xmax):
    # 1 = sign bit, 0.25 = no sign bit, always -, 0.75 = no sign bit, always +
    signbits = int((xmin * xmax) < 0)
    if signbits == 0:
        sign = "+" if [xmin, xmax] == [abs(xmin), abs(xmax)] else "-"
        sign = {"+": 0.75, "-": 0.25}[sign]
        return sign
    return signbits


def compute_ssm_fxp_qconfig(modeldict, key):
    ssm_dict = index_tree(modeldict, key)
    data = {"weights": {}, "activations": {}}

    dict_keys = dict(
        A_re="quant_A/quant_real",
        A_im="quant_A/quant_imag",
        B_re="quant_B/quant_real",
        B_im="quant_B/quant_imag",
        C_re="quant_C/quant_real",
        C_im="quant_C/quant_imag",
        D="quant_D",
    )
    for ks, kl in dict_keys.items():
        data["weights"][ks] = {}
        data["weights"][ks]["absmax"] = index_tree(
            ssm_dict, f"{kl}/observer/absmax"
        ).item()
        data["weights"][ks]["intbits"] = get_intbits(data["weights"][ks]["absmax"])
        minval = index_tree(ssm_dict, f"{kl}/observer/observer_min")
        maxval = index_tree(ssm_dict, f"{kl}/observer/observer_max")
        data["weights"][ks]["signbits"] = get_sign(minval, maxval)
        data["weights"][ks]["fracbits"] = (
            -1 * np.ceil(index_tree(ssm_dict, f"{kl}/scale")).astype(int).item()
        )

    dict_keys = dict(
        u="quant_ut",
        Bu_re="quant_But/quant_real",
        Bu_im="quant_But/quant_imag",
        x_re="quant_xt/quant_real",
        x_im="quant_xt/quant_imag",
        y="quant_yt",
    )
    for ks, kl in dict_keys.items():
        data["activations"][ks] = {}
        data["activations"][ks]["absmax"] = index_tree(
            ssm_dict, f"{kl}/observer/absmax"
        ).item()
        data["activations"][ks]["intbits"] = get_intbits(
            data["activations"][ks]["absmax"]
        )
        minval = index_tree(ssm_dict, f"{kl}/observer/observer_min")
        maxval = index_tree(ssm_dict, f"{kl}/observer/observer_max")
        data["activations"][ks]["signbits"] = get_sign(minval, maxval)
        data["activations"][ks]["fracbits"] = (
            -1 * np.ceil(index_tree(ssm_dict, f"{kl}/scale")).astype(int).item()
        )
    return data


def compute_multgate_fxp_qconfig(modeldict, key):
    multgate_dict = index_tree(modeldict, key)

    left_absmax = index_tree(multgate_dict, "quant_left/observer/absmax")
    left_min = index_tree(multgate_dict, "quant_left/observer/observer_min")
    left_max = index_tree(multgate_dict, "quant_left/observer/observer_max")
    left_signbit = get_sign(left_min, left_max)
    left_reqintbits = get_intbits(left_absmax)
    left_reqfracbits = (
        -1 * np.ceil(index_tree(multgate_dict, "quant_left/scale")).astype(int).item()
    )

    right_absmax = index_tree(multgate_dict, "quant_right/observer/absmax")
    right_min = index_tree(multgate_dict, "quant_right/observer/observer_min")
    right_max = index_tree(multgate_dict, "quant_right/observer/observer_max")
    right_signbit = get_sign(right_min, right_max)
    right_reqintbits = get_intbits(right_absmax)
    right_reqfracbits = (
        -1 * np.ceil(index_tree(multgate_dict, "quant_right/scale")).astype(int).item()
    )

    return dict(
        l_absmax=left_absmax.item(),
        l_signbits=left_signbit,
        l_intbits=left_reqintbits,
        l_fracbits=left_reqfracbits,
        r_absmax=right_absmax.item(),
        r_signbits=right_signbit,
        r_intbits=right_reqintbits,
        r_fracbits=right_reqfracbits,
    )


def compute_dense_fxp_qconfig(modeldict, key):
    dense_dict = index_tree(modeldict, key)
    w = dense_dict["kernel"]
    b = dense_dict["bias"]

    inp_absmax = dense_dict["input_observer"]["absmax"].item()
    inp_min = dense_dict["input_observer"]["input_observer_min"]
    inp_max = dense_dict["input_observer"]["input_observer_max"]
    inp_signbit = get_sign(inp_min, inp_max)
    inp_reqintbits = get_intbits(inp_absmax)
    inp_reqfracbits = int(-1 * np.ceil(dense_dict["act_scale"]).item())

    out_absmax = dense_dict["output_observer"]["absmax"].item()
    out_min = dense_dict["output_observer"]["output_observer_min"]
    out_max = dense_dict["output_observer"]["output_observer_max"]
    out_signbit = get_sign(out_min, out_max)
    out_reqintbits = get_intbits(out_absmax)
    out_reqfracbits = int(-1 * np.ceil(dense_dict["out_scale"]).item())

    w_absmax = np.abs(w).max().item()
    w_signbit = get_sign(w.min(), w.max())
    w_reqintbits = get_intbits(w_absmax)
    w_reqfracbits = int(-1 * np.ceil(dense_dict["weight_scale"]).item())

    b_absmax = np.abs(b).max().item()
    b_signbit = get_sign(b.min(), b.max())
    b_reqintbits = get_intbits(b_absmax)
    b_reqfracbits = int(-1 * np.ceil(dense_dict["act_scale"]).item())

    return dict(
        b_absmax=b_absmax,
        b_signbit=b_signbit,
        b_intbits=b_reqintbits,
        b_fracbits=b_reqfracbits,
        w_absmax=w_absmax,
        w_signbit=w_signbit,
        w_intbits=w_reqintbits,
        w_fracbits=w_reqfracbits,
        inp_absmax=inp_absmax,
        inp_signbit=inp_signbit,
        inp_intbits=inp_reqintbits,
        inp_fracbits=inp_reqfracbits,
        out_absmax=out_absmax,
        out_signbit=out_signbit,
        out_intbits=out_reqintbits,
        out_fracbits=out_reqfracbits,
    )


def custom_max(k, v, k_start_idx=0):
    if k[k_start_idx:] in ["signbits"]:
        # if there are different values for sign bit -> yes to sign bit
        return 1 if len(set(v)) >= 2 else max(v)
    else:
        return max(v)


def join_fpx_config_layers(d, agg=None):
    d_joint = {k: [d[l][k] for l in d.keys()] for k in d["layers_0"].keys()}
    if agg in ["max", "set"]:
        d_joint = {k: list(set(v)) for k, v in d_joint.items()}
    if agg == "max":
        d_max = {
            k: v[0] if len(v) == 1 else custom_max(k, v, 2) for k, v in d_joint.items()
        }
        return d_max
    elif agg is not None and agg != "set":
        raise ValueError("agg must be None or 'max'")
    return d_joint


def join_fpx_config_layers_ssm(d, agg=None):
    d_joint = {}
    d_joint["weights"] = {k: {} for k in d["layers_0"]["weights"].keys()}
    d_joint["activations"] = {k: {} for k in d["layers_0"]["activations"].keys()}

    for weight_keys in d_joint["weights"].keys():
        d_joint["weights"][weight_keys] = {
            k: [d[l]["weights"][weight_keys][k] for l in d.keys()]
            for k in d["layers_0"]["weights"][weight_keys].keys()
        }
        if agg in ["max", "set"]:
            d_joint["weights"][weight_keys] = {
                k: list(set(v)) for k, v in d_joint["weights"][weight_keys].items()
            }
        if agg == "max":
            d_max = {
                k: v[0] if len(v) == 1 else custom_max(k, v)
                for k, v in d_joint["weights"][weight_keys].items()
            }
            d_joint["weights"][weight_keys] = d_max

    for activation_keys in d_joint["activations"].keys():
        d_joint["activations"][activation_keys] = {
            k: [d[l]["activations"][activation_keys][k] for l in d.keys()]
            for k in d["layers_0"]["activations"][activation_keys].keys()
        }
        if agg in ["max", "set"]:
            d_joint["activations"][activation_keys] = {
                k: list(set(v))
                for k, v in d_joint["activations"][activation_keys].items()
            }
        if agg == "max":
            d_max = {
                k: v[0] if len(v) == 1 else custom_max(k, v)
                for k, v in d_joint["activations"][activation_keys].items()
            }
            d_joint["activations"][activation_keys] = d_max

    return d_joint


def create_fxp_qconfig(modeldict, agg="max", store_json=False, json_store_folder="."):
    data = {}
    data["encoder"] = compute_dense_fxp_qconfig(modeldict, "encoder/encoder")
    data["blocks"] = dict(ssm={}, multgate={}, out2={})
    for key in modeldict["encoder"].keys():
        if "layers_" in key:
            data["blocks"]["ssm"][key] = compute_ssm_fxp_qconfig(
                modeldict, f"encoder/{key}/mixer"
            )
            data["blocks"]["multgate"][key] = compute_multgate_fxp_qconfig(
                modeldict, f"encoder/{key}/mult_gate"
            )
            data["blocks"]["out2"][key] = compute_dense_fxp_qconfig(
                modeldict, f"encoder/{key}/out2"
            )
    data["decoder"] = compute_dense_fxp_qconfig(modeldict, "decoder")
    if store_json:
        with open(os.path.join(json_store_folder, "fxp_qconfig_full.json"), "w") as f:
            json.dump(data, f, indent=4)
    if agg in ["max", "set"]:
        data_full = data.copy()
        data["blocks"]["out2"] = join_fpx_config_layers(data["blocks"]["out2"], agg=agg)
        data["blocks"]["multgate"] = join_fpx_config_layers(
            data["blocks"]["multgate"], agg=agg
        )
        data["blocks"]["ssm"] = join_fpx_config_layers_ssm(
            data["blocks"]["ssm"], agg=agg
        )
        if store_json:
            with open(
                os.path.join(json_store_folder, f"fxp_qconfig_{agg}.json"), "w"
            ) as f:
                json.dump(data, f, indent=4)
        return data_full, data
    else:
        layer_keys = list(data["blocks"]["ssm"].keys())
        for key in layer_keys:
            data["blocks"][key] = dict(
                ssm=data["blocks"]["ssm"][key],
                multgate=data["blocks"]["multgate"][key],
                out2=data["blocks"]["out2"][key],
            )
        del data["blocks"]["ssm"]
        del data["blocks"]["multgate"]
        del data["blocks"]["out2"]
        if store_json:
            with open(
                os.path.join(json_store_folder, f"fxp_qconfig_{agg}.json"), "w"
            ) as f:
                json.dump(data, f, indent=4)
    return data


def process_dense_exp(fxp_qconfig: dict, precisions: dict[str, int], scope: str = ""):
    # weight
    w_bits = precisions["non_ssm_w"]
    fxp_qconfig["w_bits"] = w_bits
    max_wexp = w_bits - 1 - fxp_qconfig["w_intbits"]
    exp = fxp_qconfig["w_fracbits"]
    if exp > max_wexp:
        diff = exp - max_wexp
        logger.warning(f"truncate {scope}.kernel: {exp:2} -> {max_wexp:2} ({diff:2})")
        exp = max_wexp
    fxp_qconfig["w_exp"] = exp

    # bias
    b_bits = precisions["non_ssm_b"]
    fxp_qconfig["b_bits"] = b_bits
    max_bexp = b_bits - 1 - fxp_qconfig["b_intbits"]
    exp = fxp_qconfig["b_fracbits"]
    if exp > max_bexp:
        diff = exp - max_bexp
        logger.warning(f"truncate {scope}.bias: {exp:2} -> {max_bexp:2} ({diff:2})")
        exp = max_bexp
    fxp_qconfig["b_exp"] = exp

    # input activations
    inp_bits = precisions["non_ssm_act"]
    fxp_qconfig["inp_bits"] = inp_bits
    max_inp_exp = inp_bits - 1 - fxp_qconfig["inp_intbits"]
    exp = fxp_qconfig["inp_fracbits"]
    if exp > max_inp_exp:
        diff = exp - max_inp_exp
        logger.warning(f"truncate {scope}.input: {exp:2} -> {max_inp_exp:2} ({diff:2})")
        exp = max_inp_exp
    fxp_qconfig["inp_exp"] = exp

    # output activations
    out_bits = precisions["non_ssm_act"]
    fxp_qconfig["out_bits"] = out_bits
    max_out_exp = out_bits - 1 - fxp_qconfig["out_intbits"]
    exp = fxp_qconfig["out_fracbits"]
    if exp > max_out_exp:
        diff = exp - max_out_exp
        logger.warning(
            f"truncate {scope}.output: {exp:2} -> {max_out_exp:2} ({diff:2})"
        )
        exp = max_out_exp
    fxp_qconfig["out_exp"] = exp
    return fxp_qconfig


def add_target_bits_exp(
    modeldict: dict,
    fxp_qconfig: dict,
    precisions: dict[str, int],
    shared_exp=True,
):
    # dense blocks: encoder, out2, decoder
    fxp_qconfig["encoder"] = process_dense_exp(
        fxp_qconfig["encoder"], precisions, scope="encoder"
    )
    if shared_exp:
        fxp_qconfig["blocks"]["out2"] = process_dense_exp(
            fxp_qconfig["blocks"]["out2"], precisions, scope="blocks.out2"
        )
    else:
        for key in fxp_qconfig["blocks"].keys():
            fxp_qconfig["blocks"][key]["out2"] = process_dense_exp(
                fxp_qconfig["blocks"][key]["out2"],
                precisions,
                scope=f"blocks.{key}.out2",
            )
    fxp_qconfig["decoder"] = process_dense_exp(
        fxp_qconfig["decoder"], precisions, scope="decoder"
    )

    # multgate
    if shared_exp:
        l_bits = precisions["non_ssm_act"]
        r_bits = precisions["non_ssm_act"]
        fxp_qconfig["blocks"]["multgate"]["l_bits"] = l_bits
        fxp_qconfig["blocks"]["multgate"]["r_bits"] = r_bits
        max_l_exp = l_bits - 1 - fxp_qconfig["blocks"]["multgate"]["l_intbits"]
        max_r_exp = r_bits - 1 - fxp_qconfig["blocks"]["multgate"]["r_intbits"]
        l_exp = fxp_qconfig["blocks"]["multgate"]["l_fracbits"]
        r_exp = fxp_qconfig["blocks"]["multgate"]["r_fracbits"]
        if l_exp > max_l_exp:
            diff = l_exp - max_l_exp
            logger.warning(
                f"truncate multgate left:  {l_exp:2} ->" f" {max_l_exp:2} ({diff:2})"
            )
            l_exp = max_l_exp
        if r_exp > max_r_exp:
            diff = r_exp - max_r_exp
            logger.warning(
                f"truncate multgate right: {r_exp:2} ->" f" {max_r_exp:2} ({diff:2})"
            )
            r_exp = max_r_exp
        fxp_qconfig["blocks"]["multgate"]["l_exp"] = l_exp
        fxp_qconfig["blocks"]["multgate"]["r_exp"] = r_exp
    else:
        for key in fxp_qconfig["blocks"].keys():
            l_bits = precisions["non_ssm_act"]
            r_bits = precisions["non_ssm_act"]
            fxp_qconfig["blocks"][key]["multgate"]["l_bits"] = l_bits
            fxp_qconfig["blocks"][key]["multgate"]["r_bits"] = r_bits
            max_l_exp = l_bits - 1 - fxp_qconfig["blocks"][key]["multgate"]["l_intbits"]
            max_r_exp = r_bits - 1 - fxp_qconfig["blocks"][key]["multgate"]["r_intbits"]
            l_exp = fxp_qconfig["blocks"][key]["multgate"]["l_fracbits"]
            r_exp = fxp_qconfig["blocks"][key]["multgate"]["r_fracbits"]
            if l_exp > max_l_exp:
                diff = l_exp - max_l_exp
                logger.warning(
                    f"truncate {key}.multgate left:  {l_exp:2} ->"
                    f" {max_l_exp:2} ({diff:2})"
                )
                l_exp = max_l_exp
            if r_exp > max_r_exp:
                diff = r_exp - max_r_exp
                logger.warning(
                    f"truncate {key}.multgate right: {r_exp:2} ->"
                    f" {max_r_exp:2} ({diff:2})"
                )
                r_exp = max_r_exp
            fxp_qconfig["blocks"][key]["multgate"]["l_exp"] = l_exp
            fxp_qconfig["blocks"][key]["multgate"]["r_exp"] = r_exp

    # TODO: it would be better to actually collect stats on this..
    if shared_exp:
        fxp_qconfig["blocks"]["multgate"]["res_bits"] = precisions["non_ssm_act"]
        res_intbits = (
            fxp_qconfig["blocks"]["multgate"]["l_intbits"]
            + fxp_qconfig["blocks"]["multgate"]["r_intbits"]
        )
        max_res_exp = fxp_qconfig["blocks"]["multgate"]["res_bits"] - 1 - res_intbits
        fxp_qconfig["blocks"]["multgate"]["res_exp"] = max_res_exp
    else:
        for key in fxp_qconfig["blocks"].keys():
            fxp_qconfig["blocks"][key]["multgate"]["res_bits"] = precisions[
                "non_ssm_act"
            ]
            res_intbits = (
                fxp_qconfig["blocks"][key]["multgate"]["l_intbits"]
                + fxp_qconfig["blocks"][key]["multgate"]["r_intbits"]
            )
            max_res_exp = (
                fxp_qconfig["blocks"][key]["multgate"]["res_bits"] - 1 - res_intbits
            )
            fxp_qconfig["blocks"][key]["multgate"]["res_exp"] = max_res_exp

    # SSM
    if shared_exp:
        keys = list(fxp_qconfig["blocks"]["ssm"]["weights"].keys())
        for key in keys:
            # TODO: is this the right activation precision?
            n_bits = (
                precisions["ssm_act"]
                if key in ("A_re", "A_im")
                else precisions["ssm_w"]
            )
            fxp_qconfig["blocks"]["ssm"]["weights"][key]["bits"] = n_bits
            max_exp = (
                n_bits - 1 - fxp_qconfig["blocks"]["ssm"]["weights"][key]["intbits"]
            )
            exp = fxp_qconfig["blocks"]["ssm"]["weights"][key]["fracbits"]
            if exp > max_exp:
                diff = exp - max_exp
                logger.warning(f"truncate {key:4}: {exp:2} -> {max_exp:2} ({diff:2})")
                exp = max_exp
            fxp_qconfig["blocks"]["ssm"]["weights"][key]["exp"] = exp

        keys = list(fxp_qconfig["blocks"]["ssm"]["activations"].keys())
        for key in keys:
            # TODO: is this the right activation precision?
            n_bits = precisions["ssm_act"]
            fxp_qconfig["blocks"]["ssm"]["activations"][key]["bits"] = n_bits
            max_exp = (
                n_bits - 1 - fxp_qconfig["blocks"]["ssm"]["activations"][key]["intbits"]
            )
            exp = fxp_qconfig["blocks"]["ssm"]["activations"][key]["fracbits"]
            if exp > max_exp:
                diff = exp - max_exp
                logger.warning(f"truncate {key:4}: {exp:2} -> {max_exp:2} ({diff:2})")
                exp = max_exp
            fxp_qconfig["blocks"]["ssm"]["activations"][key]["exp"] = exp
    else:
        for layer in fxp_qconfig["blocks"].keys():
            keys = list(fxp_qconfig["blocks"][layer]["ssm"]["weights"].keys())
            for key in keys:
                # TODO: is this the right activation precision?
                n_bits = (
                    precisions["ssm_act"]
                    if key in ("A_re", "A_im")
                    else precisions["ssm_w"]
                )
                fxp_qconfig["blocks"][layer]["ssm"]["weights"][key]["bits"] = n_bits
                max_exp = (
                    n_bits
                    - 1
                    - fxp_qconfig["blocks"][layer]["ssm"]["weights"][key]["intbits"]
                )
                exp = fxp_qconfig["blocks"][layer]["ssm"]["weights"][key]["fracbits"]
                if exp > max_exp:
                    diff = exp - max_exp
                    logger.warning(
                        f"truncate {layer}.{key:4}: {exp:2} ->"
                        f" {max_exp:2} ({diff:2})"
                    )
                    exp = max_exp
                fxp_qconfig["blocks"][layer]["ssm"]["weights"][key]["exp"] = exp

            keys = list(fxp_qconfig["blocks"][layer]["ssm"]["activations"].keys())
            for key in keys:
                # TODO: is this the right activation precision?
                n_bits = precisions["ssm_act"]
                fxp_qconfig["blocks"][layer]["ssm"]["activations"][key]["bits"] = n_bits
                max_exp = (
                    n_bits
                    - 1
                    - fxp_qconfig["blocks"][layer]["ssm"]["activations"][key]["intbits"]
                )
                exp = fxp_qconfig["blocks"][layer]["ssm"]["activations"][key][
                    "fracbits"
                ]
                if exp > max_exp:
                    diff = exp - max_exp
                    logger.warning(
                        f"truncate {layer}.{key:4}: {exp:2} ->"
                        f" {max_exp:2} ({diff:2})"
                    )
                    exp = max_exp
                fxp_qconfig["blocks"][layer]["ssm"]["activations"][key]["exp"] = exp

    # Norms
    if shared_exp:
        norm_mean = [
            modeldict["encoder"][layer]["norm"]["mean"]
            for layer in modeldict["encoder"].keys()
            if "layers_" in layer
        ]
        norm_mean_absmax = max([np.abs(e).max() for e in norm_mean])
        norm_mean_intbits = max(0, int(np.ceil(np.log2(norm_mean_absmax))))
        norm_mean_fracbits = precisions["non_ssm_act"] - 1 - norm_mean_intbits
        fxp_qconfig["blocks"]["norm"] = {}
        fxp_qconfig["blocks"]["norm"]["mean"] = {
            "intbits": norm_mean_intbits,
            "exp": norm_mean_fracbits,
            "bits": precisions["non_ssm_act"],
        }

        norm_var = [
            modeldict["encoder"][layer]["norm"]["var"]
            for layer in modeldict["encoder"].keys()
            if "layers_" in layer
        ]
        norm_var_absmax = max([np.abs(e).max() for e in norm_var])
        norm_var_intbits = max(0, int(np.ceil(np.log2(norm_var_absmax))))
        norm_var_fracbits = precisions["non_ssm_act"] - 1 - norm_var_intbits
        fxp_qconfig["blocks"]["norm"]["var"] = {
            "intbits": norm_var_intbits,
            "exp": norm_var_fracbits,
            "bits": precisions["non_ssm_act"],
        }

        bn_eps = 1e-5
        invsq_var = [jax.lax.rsqrt(e + bn_eps) for e in norm_var]
        invsq_var_absmax = max([np.abs(e).max() for e in invsq_var])
        invsq_var_intbits = max(0, int(np.ceil(np.log2(invsq_var_absmax))))
        invsq_var_fracbits = precisions["non_ssm_act"] - 1 - invsq_var_intbits
        fxp_qconfig["blocks"]["norm"]["invsq_var"] = {
            "intbits": invsq_var_intbits,
            "exp": invsq_var_fracbits,
            "bits": precisions["non_ssm_act"],
        }
    else:
        for layer in fxp_qconfig["blocks"].keys():
            norm_mean = modeldict["encoder"][layer]["norm"]["mean"]
            norm_mean_absmax = max([np.abs(e).max() for e in norm_mean])
            norm_mean_intbits = max(0, int(np.ceil(np.log2(norm_mean_absmax))))
            norm_mean_fracbits = precisions["non_ssm_act"] - 1 - norm_mean_intbits
            fxp_qconfig["blocks"][layer]["norm"] = {}
            fxp_qconfig["blocks"][layer]["norm"]["mean"] = {
                "intbits": norm_mean_intbits,
                "exp": norm_mean_fracbits,
                "bits": precisions["non_ssm_act"],
            }

            norm_var = modeldict["encoder"][layer]["norm"]["var"]
            norm_var_absmax = max([np.abs(e).max() for e in norm_var])
            norm_var_intbits = max(0, int(np.ceil(np.log2(norm_var_absmax))))
            norm_var_fracbits = precisions["non_ssm_act"] - 1 - norm_var_intbits
            fxp_qconfig["blocks"][layer]["norm"]["var"] = {
                "intbits": norm_var_intbits,
                "exp": norm_var_fracbits,
                "bits": precisions["non_ssm_act"],
            }

            bn_eps = 1e-5
            invsq_var = [jax.lax.rsqrt(e + bn_eps) for e in norm_var]
            invsq_var_absmax = max([np.abs(e).max() for e in invsq_var])
            invsq_var_intbits = max(0, int(np.ceil(np.log2(invsq_var_absmax))))
            invsq_var_fracbits = precisions["non_ssm_act"] - 1 - invsq_var_intbits
            fxp_qconfig["blocks"][layer]["norm"]["invsq_var"] = {
                "intbits": invsq_var_intbits,
                "exp": invsq_var_fracbits,
                "bits": precisions["non_ssm_act"],
            }

    if shared_exp:
        if "scale" in modeldict["encoder"]["layers_0"]["norm"]:
            for layer in modeldict["encoder"].keys():
                if "layers_" in layer:
                    modeldict["encoder"][layer]["norm"]["scale"] = np.where(
                        np.isnan(modeldict["encoder"][layer]["norm"]["scale"]),
                        1.0,
                        modeldict["encoder"][layer]["norm"]["scale"],
                    )
            norm_scale = [
                modeldict["encoder"][layer]["norm"]["scale"]
                for layer in modeldict["encoder"].keys()
                if "layers_" in layer
            ]
            norm_scale_absmax = max([np.abs(e).max() for e in norm_scale])
            norm_scale_intbits = max(0, int(np.ceil(np.log2(norm_scale_absmax))))
            norm_scale_fracbits = precisions["non_ssm_act"] - 1 - norm_scale_intbits
            fxp_qconfig["blocks"]["norm"]["scale"] = {
                "intbits": norm_scale_intbits,
                "exp": norm_scale_fracbits,
                "bits": precisions["non_ssm_act"],
            }
        if "bias" in modeldict["encoder"]["layers_0"]["norm"]:
            for layer in modeldict["encoder"].keys():
                if "layers_" in layer:
                    modeldict["encoder"][layer]["norm"]["bias"] = np.where(
                        np.isnan(modeldict["encoder"][layer]["norm"]["bias"]),
                        1.0,
                        modeldict["encoder"][layer]["norm"]["bias"],
                    )
            norm_bias = [
                modeldict["encoder"][layer]["norm"]["bias"]
                for layer in modeldict["encoder"].keys()
                if "layers_" in layer
            ]
            norm_bias_absmax = max([np.abs(e).max() for e in norm_bias])
            norm_bias_intbits = max(0, int(np.ceil(np.log2(norm_bias_absmax))))
            norm_bias_fracbits = precisions["non_ssm_act"] - 1 - norm_bias_intbits
            fxp_qconfig["blocks"]["norm"]["bias"] = {
                "intbits": norm_bias_intbits,
                "exp": norm_bias_fracbits,
                "bits": precisions["non_ssm_act"],
            }
    else:
        for layer in fxp_qconfig["blocks"].keys():
            if "scale" in modeldict["encoder"][layer]["norm"]:
                modeldict["encoder"][layer]["norm"]["scale"] = np.where(
                    np.isnan(modeldict["encoder"][layer]["norm"]["scale"]),
                    1.0,
                    modeldict["encoder"][layer]["norm"]["scale"],
                )
                norm_scale = modeldict["encoder"][layer]["norm"]["scale"]
                norm_scale_absmax = max([np.abs(e).max() for e in norm_scale])
                norm_scale_intbits = max(0, int(np.ceil(np.log2(norm_scale_absmax))))
                norm_scale_fracbits = precisions["non_ssm_act"] - 1 - norm_scale_intbits
                fxp_qconfig["blocks"][layer]["norm"]["scale"] = {
                    "intbits": norm_scale_intbits,
                    "exp": norm_scale_fracbits,
                    "bits": precisions["non_ssm_act"],
                }
            if "bias" in modeldict["encoder"][layer]["norm"]:
                modeldict["encoder"][layer]["norm"]["bias"] = np.where(
                    np.isnan(modeldict["encoder"][layer]["norm"]["bias"]),
                    1.0,
                    modeldict["encoder"][layer]["norm"]["bias"],
                )
                norm_bias = modeldict["encoder"][layer]["norm"]["bias"]
                norm_bias_absmax = max([np.abs(e).max() for e in norm_bias])
                norm_bias_intbits = max(0, int(np.ceil(np.log2(norm_bias_absmax))))
                norm_bias_fracbits = precisions["non_ssm_act"] - 1 - norm_bias_intbits
                fxp_qconfig["blocks"][layer]["norm"]["bias"] = {
                    "intbits": norm_bias_intbits,
                    "exp": norm_bias_fracbits,
                    "bits": precisions["non_ssm_act"],
                }

    return fxp_qconfig


def manually_overwrite(model, manual_overwrite):
    if len(manual_overwrite["model_attr"]) > 0:
        for key, val in manual_overwrite["model_attr"].items():
            idx_model = model
            key_list = key.split(".")
            for k in key_list[:-1]:
                idx_model = getattr(idx_model, k)
            prev_val = getattr(idx_model, key_list[-1])
            setattr(idx_model, key_list[-1], val)
            now_val = getattr(idx_model, key_list[-1])
            logger.info(f"MANUAL OVERWRITE: {key} <= {now_val} (was: {prev_val})")
