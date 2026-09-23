from __future__ import print_function

import os
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.linewidth"] = 0.9
plt.rcParams["xtick.direction"] = "out"
plt.rcParams["ytick.direction"] = "out"



weight_labels = [
    r"$\alpha=0,\gamma=1$",
    r"$\alpha=0.1,\gamma=0.9$",
    r"$\alpha=0.25,\gamma=0.75$",
    r"$\alpha=0.5,\gamma=0.5$",
    r"$\alpha=0.75,\gamma=0.25$",
    r"$\alpha=1,\gamma=0$"
]

data = {
    "Sensitive Probability": [0.8473, 0.6924, 0.7047, 0.8482, 0.7281, 0.3122],
    "Sensitive Rate": [0.9900, 0.9550, 0.9688, 0.9933, 0.9700, 0.3583],
    "Flip Rate": [0.7987, 0.7675, 0.7775, 0.8017, 0.7800, 0.2629],
    "ASR-SR": [0.9887, 0.9500, 0.9624, 0.9923, 0.9654, 0.3254],
    "EER": [0.3137, 0.2879, 0.3023, 0.4501, 0.4475, 0.4329],
    "FAR": [0.3659, 0.3198, 0.3488, 0.8763, 0.8608, 0.9470],
    "FRR": [0.2675, 0.2616, 0.2597, 0.0777, 0.1000, 0.0422],
    "ASR-SV": [0.3167, 0.2907, 0.3043, 0.4771, 0.4585, 0.4947],
}



output_dir = "ablation_figures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

bar_color = "#4C78A8"
bar_edge_color = "#2F4B6C"
highlight_color = "#F58518"
highlight_edge_color = "#9C520D"
grid_color = "#D9DEE7"
highlight_index = 3


for metric, values in data.items():
    fig, ax = plt.subplots(figsize=(7.2, 4.4))

    x = range(len(weight_labels))
    colors = [bar_color for _ in values]
    edge_colors = [bar_edge_color for _ in values]
    colors[highlight_index] = highlight_color
    edge_colors[highlight_index] = highlight_edge_color
    bars = ax.bar(
        x,
        values,
        width=0.48,
        color=colors,
        edgecolor=edge_colors,
        linewidth=0.9,
        zorder=3,
    )

    ax.set_xticks(list(x))
    ax.set_xticklabels(weight_labels, rotation=28, ha="right", fontsize=9)
    ax.set_ylabel(metric, fontsize=11)
    ax.set_xlabel(r"Loss Weight Combination $(\alpha,\gamma)$", fontsize=11)
    ax.set_title("Ablation Study of {0}".format(metric), fontsize=12, pad=10)
    ax.set_ylim(0, 1.05)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color=grid_color, zorder=0)
    ax.xaxis.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=9)

    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            v + 0.012,
            "{0:.3f}".format(v),
            ha="center",
            va="bottom",
            fontsize=8,
            color="#222222",
        )

    fig.tight_layout()

    filename = metric.lower().replace(" ", "_").replace("-", "_") + ".png"
    save_path = os.path.join(output_dir, filename)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

print("All figures are saved in: {0}".format(output_dir))
