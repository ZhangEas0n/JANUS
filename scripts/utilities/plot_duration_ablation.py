from __future__ import print_function

import os

import matplotlib.pyplot as plt


def plot_duration_ablation(output_dir="figures/ablation/perturbation_length"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    durations = ["0.5s", "1s", "2s", "3s", "4s", "5s"]
    metrics = [
        "Sensitive\nprob.",
        "Sensitive\nrate",
        "Flip\nrate",
        "ASR-SR",
        "EER",
        "FAR",
        "FRR",
        "ASR-SV",
    ]
    data = {
        "0.5s": [0.8495, 0.9800, 0.7887, 0.9763, 0.4632, 0.7952, 0.1531, 0.4742],
        "1s": [0.7994, 0.9917, 0.8004, 0.9907, 0.4211, 0.7267, 0.1691, 0.4480],
        "2s": [0.7664, 0.9812, 0.7925, 0.9809, 0.4206, 0.7865, 0.1134, 0.4500],
        "3s": [0.7994, 0.9879, 0.7958, 0.9850, 0.4283, 0.7295, 0.1686, 0.4491],
        "4s": [0.6493, 0.9425, 0.7567, 0.9366, 0.3420, 0.5846, 0.1454, 0.3650],
        "5s": [0.6867, 0.9162, 0.7333, 0.9077, 0.4305, 0.6333, 0.2494, 0.4414],
    }

    metric_colors = [
        "#4E79A7",
        "#59A14F",
        "#F28E2B",
        "#E15759",
        "#76B7B2",
        "#B07AA1",
        "#9C755F",
        "#EDC948",
    ]
    edge_color = "#333333"

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"

    fig, axes = plt.subplots(2, 3, figsize=(14.2, 7.0), sharey=True)
    axes_flat = [ax for row in axes for ax in row]
    x = list(range(len(metrics)))

    for ax, duration in zip(axes_flat, durations):
        values = data[duration]
        bars = ax.bar(
            x,
            values,
            width=0.58,
            color=metric_colors,
            edgecolor=edge_color,
            linewidth=0.55,
            zorder=3,
        )
        ax.set_title("UAP length = {0}".format(duration), fontsize=12, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0.0, 1.08)
        ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.35, zorder=0)
        ax.tick_params(axis="y", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value + 0.015,
                "{0:.2f}".format(value),
                ha="center",
                va="bottom",
                fontsize=7,
                color="#222222",
            )

    axes[0][0].set_ylabel("Metric value", fontsize=11)
    axes[1][0].set_ylabel("Metric value", fontsize=11)

    legend_labels = [
        "Sensitive prob.",
        "Sensitive rate",
        "Flip rate",
        "ASR-SR",
        "EER",
        "FAR",
        "FRR",
        "ASR-SV",
    ]
    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, color=color)
        for color in metric_colors
    ]
    fig.legend(
        legend_patches,
        legend_labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 1.025),
    )
    fig.suptitle("Effect of UAP Duration on SR/SV Attack Metrics", fontsize=14, y=1.065)
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=1.2, h_pad=1.6)

    png_path = os.path.join(output_dir, "duration_ablation_2x3.png")
    pdf_path = os.path.join(output_dir, "duration_ablation_2x3.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print("Saved: {0}".format(png_path))
    print("Saved: {0}".format(pdf_path))


if __name__ == "__main__":
    plot_duration_ablation()
