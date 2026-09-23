from __future__ import print_function

import os

import matplotlib.pyplot as plt


def plot_sv_threshold_sensitivity(output_dir="figures/ablation/threshold"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    metrics = {
        "FAR": [0.9510, 0.8116, 0.6654, 0.5171, 0.3128],
        "FRR": [0.0268, 0.1117, 0.2153, 0.3417, 0.5494],
        "ASR-SV": [0.4890, 0.4617, 0.4404, 0.4294, 0.4311],
    }
    colors = {
        "FAR": "#4E79A7",
        "FRR": "#F28E2B",
        "ASR-SV": "#C43C39",
    }
    markers = {
        "FAR": "o",
        "FRR": "s",
        "ASR-SV": "D",
    }

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.linewidth"] = 0.9
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    for name in ["FAR", "FRR", "ASR-SV"]:
        ax.plot(
            thresholds,
            metrics[name],
            marker=markers[name],
            color=colors[name],
            linewidth=2.1,
            markersize=5.8,
            markerfacecolor=colors[name],
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=name,
        )
        for x_value, y_value in zip(thresholds, metrics[name]):
            ax.text(
                x_value,
                y_value + 0.025,
                "{0:.3f}".format(y_value),
                ha="center",
                va="bottom",
                fontsize=8,
                color=colors[name],
            )

    ax.set_xlabel(r"Decision threshold $\tau$", fontsize=11)
    ax.set_ylabel("Metric value", fontsize=11)
    ax.set_title("SV Attack Sensitivity Across Decision Thresholds", fontsize=12, pad=10)
    ax.set_xticks(thresholds)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.35)
    ax.tick_params(axis="both", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    fig.tight_layout()

    png_path = os.path.join(output_dir, "sv_threshold_sensitivity.png")
    pdf_path = os.path.join(output_dir, "sv_threshold_sensitivity.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print("Saved: {0}".format(png_path))
    print("Saved: {0}".format(pdf_path))


if __name__ == "__main__":
    plot_sv_threshold_sensitivity()
