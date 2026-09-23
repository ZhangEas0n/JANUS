from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_transfer_model_comparison(output_dir="figures/main"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = ["wav2vec2-base", "wav2vec2-large", "HuBERT-base", "WavLM-base"]
    sr_metrics = [
        "Sensitive probability",
        "Sensitive rate",
        "Flip rate",
        "ASR-SR",
    ]
    sv_metrics = [
        "EER",
        "FAR",
        "FRR",
        "ASR-SV",
    ]
    metrics = sr_metrics + sv_metrics
    values = np.array(
        [
            [0.7994, 0.4678, 0.6211, 0.4595],
            [0.9879, 0.7354, 0.9262, 0.7158],
            [0.7958, 0.6121, 0.8112, 0.6662],
            [0.9850, 0.7128, 0.9214, 0.7022],
            [0.4283, 0.3301, 0.32985, 0.3017],
            [0.7295, 0.6895, 0.4474, 0.3786],
            [0.1686, 0.0557, 0.2277, 0.2366],
            [0.4491, 0.3727, 0.3376, 0.3076],
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    image = ax.imshow(values, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")

    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels(metrics)

    ax.set_xticks(np.arange(values.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(values.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.4)
    ax.tick_params(which="minor", bottom=False, left=False)

    ax.axhline(len(sr_metrics) - 0.5, color="#2b2b2b", linewidth=1.2)
    for row in range(values.shape[0]):
        row_best = int(np.argmax(values[row]))
        for col in range(values.shape[1]):
            is_best = col == row_best
            ax.text(
                col,
                row,
                f"{values[row, col]:.4f}",
                ha="center",
                va="center",
                fontsize=9,
                color="white" if values[row, col] > 0.62 else "#1f2933",
                fontweight="bold" if is_best else "normal",
            )

    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.025)
    cbar.set_label("Metric value", rotation=270, labelpad=14)
    cbar.outline.set_visible(False)

    ax.set_title("Transfer Evaluation Across Backbones", fontsize=14, pad=12)
    fig.tight_layout()

    png_path = output_dir / "transfer_model_comparison_heatmap.png"
    pdf_path = output_dir / "transfer_model_comparison_heatmap.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    plot_transfer_model_comparison()
