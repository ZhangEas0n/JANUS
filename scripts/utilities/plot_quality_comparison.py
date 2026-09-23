from __future__ import print_function

import os

import matplotlib.pyplot as plt


def plot_quality_comparison(output_dir="figures/main"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    snrs = [10, 20, 30]
    methods = ["BTUAP", "Kenansville", "PGD", "Random Noise"]

    stoi = {
        "BTUAP": [0.8936, 0.9643, 0.9894],
        "Kenansville": [0.9269, 0.9908, 0.9960],
        "PGD": [0.8953, 0.9660, 0.9901],
        "Random Noise": [0.9108, 0.9716, 0.9913],
    }
    pesq = {
        "BTUAP": [1.2530, 1.9050, 2.8439],
        "Kenansville": [1.9466, 2.9197, 3.8156],
        "PGD": [1.2427, 1.8939, 2.8280],
        "Random Noise": [1.2504, 1.9024, 2.8317],
    }

    colors = {
        "BTUAP": "#C43C39",
        "Kenansville": "#4C78A8",
        "PGD": "#59A14F",
        "Random Noise": "#8E6BBE",
    }
    markers = {
        "BTUAP": "o",
        "Kenansville": "s",
        "PGD": "^",
        "Random Noise": "D",
    }

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["axes.linewidth"] = 0.9
    plt.rcParams["xtick.direction"] = "out"
    plt.rcParams["ytick.direction"] = "out"

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))

    for ax, title, ylabel, values, ylim in [
        (axes[0], "STOI under matched SNR", "STOI", stoi, (0.86, 1.01)),
        (axes[1], "PESQ under matched SNR", "PESQ", pesq, (1.0, 4.05)),
    ]:
        for method in methods:
            ax.plot(
                snrs,
                values[method],
                marker=markers[method],
                color=colors[method],
                linewidth=2.0,
                markersize=5.5,
                markerfacecolor=colors[method],
                markeredgecolor="white",
                markeredgewidth=0.7,
                label=method,
            )
            for x_value, y_value in zip(snrs, values[method]):
                ax.text(
                    x_value,
                    y_value + (0.004 if ylabel == "STOI" else 0.055),
                    "{0:.3f}".format(y_value),
                    ha="center",
                    va="bottom",
                    fontsize=7.2,
                    color=colors[method],
                )

        ax.set_title(title, fontsize=12, pad=9)
        ax.set_xlabel("Target SNR (dB)", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(snrs)
        ax.set_ylim(*ylim)
        ax.grid(True, linestyle="--", linewidth=0.55, alpha=0.35)
        ax.tick_params(axis="both", labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, 1.03),
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93), w_pad=2.0)

    png_path = os.path.join(output_dir, "quality_comparison_snr.png")
    pdf_path = os.path.join(output_dir, "quality_comparison_snr.pdf")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print("Saved: {0}".format(png_path))
    print("Saved: {0}".format(pdf_path))


if __name__ == "__main__":
    plot_quality_comparison()
