from __future__ import print_function

import os

import matplotlib.pyplot as plt


def plot_alpha_gamma_ablation(output_dir="figures/ablation/loss_weights"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    alpha_gamma = [
        (0.0, 1.0),
        (0.1, 0.9),
        (0.25, 0.75),
        (0.5, 0.5),
        (0.75, 0.25),
        (1.0, 0.0),
    ]
    x = list(range(len(alpha_gamma)))
    x_labels = ["({0:g}, {1:g})".format(alpha, gamma) for alpha, gamma in alpha_gamma]

    sr_metrics = [
        ("Sensitive probability", [0.8473, 0.6924, 0.7047, 0.8482, 0.7281, 0.3122]),
        ("Sensitive rate", [0.9900, 0.9550, 0.9688, 0.9933, 0.9700, 0.3583]),
        ("Flip rate", [0.7987, 0.7675, 0.7775, 0.8017, 0.7800, 0.2629]),
        ("ASR-SR", [0.9887, 0.9500, 0.9624, 0.9923, 0.9654, 0.3254]),
    ]
    sv_metrics = [
        ("EER", [0.3137, 0.2879, 0.3023, 0.4501, 0.4475, 0.4329]),
        ("FAR", [0.3659, 0.3198, 0.3488, 0.8763, 0.8608, 0.9470]),
        ("FRR", [0.2675, 0.2616, 0.2597, 0.0777, 0.1000, 0.0422]),
        ("ASR-SV", [0.3167, 0.2907, 0.3043, 0.4771, 0.4585, 0.4947]),
    ]

    colors = {
        "Sensitive probability": "#2E86AB",
        "Sensitive rate": "#00A676",
        "Flip rate": "#F18F01",
        "ASR-SR": "#C73E1D",
        "EER": "#6A4C93",
        "FAR": "#D1495B",
        "FRR": "#00798C",
        "ASR-SV": "#EDA92A",
    }
    markers = ["o", "s", "^", "D"]
    highlight_index = alpha_gamma.index((0.5, 0.5))

    def style_panel(ax, title, metrics):
        for marker, (name, values) in zip(markers, metrics):
            ax.plot(
                x,
                values,
                marker=marker,
                linewidth=2.0,
                markersize=5.0,
                label=name,
                color=colors[name],
                markerfacecolor=colors[name],
                markeredgecolor="white",
                markeredgewidth=0.6,
            )
        ax.axvline(
            highlight_index,
            color="#333333",
            linestyle="--",
            linewidth=1.0,
            alpha=0.55,
        )
        ax.text(
            highlight_index + 0.06,
            0.05,
            r"$\alpha=0.5,\ \gamma=0.5$",
            rotation=90,
            va="bottom",
            ha="left",
            fontsize=8.5,
            color="#333333",
        )
        ax.set_title(title, fontsize=12, pad=9)
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=28, ha="right", fontsize=9)
        ax.set_xlabel(r"($\alpha$, $\gamma$)", fontsize=10)
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.32)
        ax.tick_params(axis="y", labelsize=9)
        ax.tick_params(axis="x", length=2.5)
        ax.legend(frameon=False, fontsize=8.5, loc="lower left")
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#555555")
        ax.set_ylabel("Metric value", fontsize=10)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=True)
    style_panel(axes[0], "SR attack metrics", sr_metrics)
    style_panel(axes[1], "SV attack metrics", sv_metrics)
    axes[1].set_ylabel("")

    fig.suptitle(
        r"Effect of $\alpha$ and $\gamma$ on SR/SV attack performance",
        fontsize=14,
        y=0.99,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), w_pad=2.0)

    png_path = os.path.join(output_dir, "alpha_gamma_ablation_combined.png")
    pdf_path = os.path.join(output_dir, "alpha_gamma_ablation_combined.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print("Saved: {0}".format(png_path))
    print("Saved: {0}".format(pdf_path))


if __name__ == "__main__":
    plot_alpha_gamma_ablation()
