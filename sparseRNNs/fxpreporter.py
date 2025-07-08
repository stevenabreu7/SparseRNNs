import json
import os
from functools import partial

import jax.numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sparseRNNs.fxpmodel import FxpS5Config


def compute_error(xrec, xhat):
    err = np.abs(xrec - xhat)
    rel_err = err / np.abs(xhat)
    err_metrics = dict(
        abs_error_mean=err.mean(),
        abs_error_std=err.std(),
        abs_error_max=err.max(),
        abs_error_med=np.median(err),
        rel_error_mean=rel_err.mean(where=xhat != 0),
        rel_error_max=rel_err.max(where=xhat != 0, initial=0),
        rel_error_med=np.median(rel_err[xhat != 0]),
    )
    return err_metrics


def save_error_plot(
    x1,
    x2,
    name,
    name1,
    name2,
    nlines=100,
    maxlen=100,
    toffset=0,
    folder="verification",
):
    os.makedirs(folder, exist_ok=True)
    title = f"{name} ({name1} vs. {name2})"
    fig, axs = plt.subplots(2, 1, figsize=(10, 4), dpi=300, sharex=True)
    fig.suptitle(title)
    tstart = toffset
    tend = toffset + maxlen if (toffset + maxlen) != 0 else None
    axs[0].plot(x1[tstart:tend, :nlines], label=name1, alpha=0.3)
    axs[0].set_prop_cycle(None)
    axs[0].plot(x2[tstart:tend, :nlines], label=name2, ls="--")
    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].legend(handles[:: len(handles) // 2], labels[:: len(handles) // 2])
    axs[0].set_ylabel("activation value")
    axs[1].plot(np.abs(x1 - x2)[tstart:tend, :nlines], label="error")
    axs[1].set_ylabel("absolute error")
    axs[1].set_xlabel("timesteps")
    fig.savefig(os.path.join(folder, f"{name.replace(' ', '_')}.png"), dpi=300)
    fig.clear()
    plt.close()


class Reporter:
    def __init__(
        self,
        precisions: dict,
        fxp_qconfig: dict,
        cfg: FxpS5Config,
        manual_overwrite: dict,
        folder_name: str,
        plot_nlines: int = 100,
        plot_maxlen: int = 100,
        plot_toffset: int = 0,
    ):
        self.report = ""
        config_section = "## Configuration\n\n"
        precision_str = self._code(json.dumps(precisions, indent=2), "json")
        config_section += f"Precisions config:\n\n{precision_str}\n\n"
        qconfig_str = self._code(json.dumps(fxp_qconfig, indent=2), "json")
        config_section += f"Quant config:\n\n{qconfig_str}\n\n"
        cfg_str = self._code(json.dumps(cfg.to_dict(), indent=2), "json")
        config_section += f"Model config:\n\n{cfg_str}\n\n"
        overwrite_str = self._code(json.dumps(manual_overwrite, indent=2), "json")
        config_section += f"Overwriting:\n\n{overwrite_str}\n\n"
        self.config_section = config_section
        self.folder_name = folder_name
        self.results_data = []
        self.plot_nlines = plot_nlines
        self.plot_maxlen = plot_maxlen
        self.plot_toffset = plot_toffset
        self.error_plot = partial(
            save_error_plot,
            folder=folder_name,
            nlines=plot_nlines,
            maxlen=plot_maxlen,
            toffset=plot_toffset,
        )
        self.data_section = ""

    def _code(self, msg, type=""):
        return f"```{type}\n{msg}\n```"

    def _error_msg(self, err_metrics, xhat):
        err_msg = (
            f"abs. error: mean {err_metrics['abs_error_mean']:8.4f}, median"
            f" {err_metrics['abs_error_med']:8.4f}"
        )
        err_msg += f", max {err_metrics['abs_error_max']:8.4f}\n"
        err_msg += (
            f"rel. error: mean {err_metrics['rel_error_mean']:8.3%}, median"
            f" {err_metrics['rel_error_med']:8.3%}"
        )
        rel_err_max = err_metrics["rel_error_max"]
        rel_err_max = (
            f"{rel_err_max:8.3%}" if rel_err_max < 1.0 else f"{rel_err_max:10.4f}"
        )
        err_msg += f", max {rel_err_max}\n"
        err_msg += (
            f"xhat: mean {xhat.mean():9.4f}, median {np.median(xhat):8.4f},"
            f" max {xhat.max():8.4f}"
        )
        return err_msg

    def add_block_raw(
        self,
        name,
        xhat,
        xrec,
        verbose=True,
        plot=True,
        xhatname="float",
        xrecname="fxp",
    ):
        assert xhat.shape == xrec.shape, "xhat and xrec must have the same shape"
        assert xhat.dtype == xrec.dtype, "xhat and xrec must have the same dtype"
        is_complex = np.issubdtype(xhat.dtype, np.complexfloating)

        if is_complex:
            ## real part
            err_metrics_re = compute_error(xrec=xrec.real, xhat=xhat.real)
            err_msg_re = self._error_msg(err_metrics_re, xhat=xhat.real)
            if verbose:
                print(f"{name:<35}", err_msg_re.replace("\n", " --- "))
            self.results_data.append(
                dict(
                    name=f"{name} (real)",
                    **err_metrics_re,
                )
            )
            if plot:
                self.error_plot(
                    xrec.real, xhat.real, f"{name}_real", xrecname, xhatname
                )
            ## imag part
            err_metrics_im = compute_error(xrec=xrec.imag, xhat=xhat.imag)
            err_msg_im = self._error_msg(err_metrics_im, xhat=xhat.imag)
            if verbose:
                print(f"{name:<35}", err_msg_im.replace("\n", " --- "))
            self.results_data.append(
                dict(
                    name=f"{name} (imag)",
                    **err_metrics_im,
                )
            )
            if plot:
                self.error_plot(
                    xrec.imag, xhat.imag, f"{name}_imag", xrecname, xhatname
                )
            ## add to report
            subnames = ["real", "imag"]
            err_msgs = [err_msg_re, err_msg_im]
            self.add_multi_block(name, subnames, err_msgs)

        else:
            err_metrics = compute_error(xrec=xrec, xhat=xhat)
            err_msg = self._error_msg(err_metrics, xhat=xhat)
            if verbose:
                print(f"{name:<35}", err_msg.replace("\n", " --- "))
            self.results_data.append(
                dict(
                    name=name,
                    **err_metrics,
                )
            )
            if plot:
                self.error_plot(xrec, xhat, name, xrecname, xhatname)
            self.add_block(name, err_msg)

    def add_block(self, name, err_msg):
        self.report += f"## {name}\n\n"
        err_msg = err_msg.replace(" -- ", "\n")
        self.report += f"```\n{err_msg}\n```\n\n"
        self.report += f"![{name} error plot](./{name.replace(' ', '_')}.png)\n\n"

    def add_multi_block(self, name, subnames, err_msgs):
        self.report += f"## {name}\n\n"
        for subname, err_msg in zip(subnames, err_msgs):
            self.report += f"### {subname}\n\n"
            err_msg = err_msg.replace(" -- ", "\n")
            self.report += f"```\n{err_msg}\n```\n\n"
            img_fname = f"./{name.replace(' ', '_')}_{subname.replace(' ', '_')}.png"
            self.report += f"![{name} error plot]({img_fname})\n\n"

    def save_summary_plot(self):
        df = pd.DataFrame(self.results_data)
        fig, all_axs = plt.subplots(2, 2, figsize=(16, 9), dpi=300, sharex=True)
        fig.suptitle("Average error by block for fixed-point S5 model", fontsize=16)
        for i in range(2):
            axs = [all_axs[0][i], all_axs[1][i]]
            metric = "abs"
            axs[0].plot(df[f"{metric}_error_mean"], label="mean", color="tab:blue")
            axs[0].plot(df[f"{metric}_error_med"], label="median", color="tab:green")
            if i == 0:
                lower = df[f"{metric}_error_mean"].astype(float) - df[
                    f"{metric}_error_std"
                ].astype(float)
                label = "$\mu \pm \sigma$"
            else:
                lower = df[f"{metric}_error_mean"].astype(float)
                label = "$[\mu, \mu + \sigma]$"
            upper = df[f"{metric}_error_mean"].astype(float) + df[
                f"{metric}_error_std"
            ].astype(float)
            axs[0].fill_between(
                df.index,
                lower,
                upper,
                alpha=0.3,
                color="tab:blue",
                label=label,
            )
            axs[0].set_xticks(df.index)
            axs[0].legend()
            metric = "rel"
            axs[1].plot(df[f"{metric}_error_mean"], label="mean", color="tab:blue")
            axs[1].plot(df[f"{metric}_error_med"], label="median", color="tab:green")
            axs[1].set_xticks(df.index)
            names = (
                df["name"]
                .str.replace("encoder.", "")
                .str.replace(" (", ".")
                .str.replace(")", "")
            )
            names = names.str.replace(".calc_hat", "_calc").str.replace("layers_", "")
            axs[1].hlines(1.0, 0, len(df), color="red", ls="--", label="100%")
            axs[1].set_xticklabels(names, rotation=90)
            axs[1].legend()
            axs[1].yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
            if i == 0:
                axs[0].set_ylabel("abs. error")
                axs[1].set_ylabel("rel. error")
            if i == 1:
                axs[0].set_yscale("log")
                axs[1].set_yscale("log")
            for ax in axs:
                ax.grid(True, which="both", ls="--", alpha=0.8, axis="x")
        fig.tight_layout()
        fig.savefig(f"{self.folder_name}/summary_plot.png", dpi=300)
        fig.clear()
        plt.close()

    def get_summary_section(self):
        summary_section = "## Summary\n\n"
        df = pd.DataFrame(self.results_data)
        self.save_summary_plot()
        summary_section += "![summary plot](./summary_plot.png)\n\n"
        summary_section += df.to_markdown(index=False)
        return summary_section.strip()

    def save(self, filename: str = "README.md"):
        full_report = "# Verification Report\n\n"
        summary_section = self.get_summary_section()
        full_report = (
            f"{summary_section}\n\n{self.report.strip()}\n\n{self.config_section}"
        )
        with open(f"{self.folder_name}/{filename}", "w") as f:
            f.write(full_report)
