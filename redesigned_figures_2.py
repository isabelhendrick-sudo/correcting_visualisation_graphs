"""Regenerate Figures 1 and 2 from the figure redesign project."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)

# Okabe-Ito colour-blind-safe palette; each variable keeps one colour in every panel
BLUE, ORANGE, GREEN, VERMILION, SKY, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#8C8C8C"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 10, "legend.frameon": False, "axes.linewidth": 0.8,
    "xtick.direction": "in", "ytick.direction": "in", "savefig.dpi": 300, "savefig.facecolor": "white",
})


def grid(ax, axis="both"):
    """The same light gridlines on every panel."""
    ax.grid(axis=axis, color="0.9", lw=0.6)
    ax.set_axisbelow(True)


def panel_title(ax, letter, text):
    ax.set_title(f"({letter}) {text}", loc="left", fontweight="bold")


# ------------------------------------------------------------------ Figure 1: ablation results (Chen et al., 2026)
def figure_ablation():
    models = ["Baseline (YOLOv8n)", "Exp 1 (+DB&SA)", "Exp 2 (+P2&WIoU)", "Exp 3 (+LSKA)", "Exp 4 (+LAMP)",
              "Ours (DSLW)"]
    metrics = {"mAP@0.5": ([91.1, 90.8, 91.3, 91.6, 91.1, 94.3], BLUE, "o"),
               "Recall": ([89.0, 95.0, 95.0, 92.0, 89.0, 96.0], ORANGE, "s"),
               "F1-score": ([87.0, 88.0, 88.0, 89.0, 87.0, 90.0], GREEN, "D")}
    params = [3.20, 2.47, 2.50, 2.65, 2.20, 3.35]

    fig, axs = plt.subplots(2, 2, figsize=(13, 9.5))

    # (a) Heatmap table: exact values, shaded within each metric, best value in bold
    ax = axs[0, 0]
    names = list(metrics)
    vals = np.array([metrics[m][0] for m in names]).T
    scaled = (vals - vals.min(0)) / (vals.max(0) - vals.min(0))
    ax.imshow(scaled, cmap="Blues", aspect="auto", vmin=-0.15, vmax=1)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            best = vals[i, j] == vals[:, j].max()
            ax.text(j, i, f"{vals[i, j]:.1f}", ha="center", va="center", fontsize=11,
                    fontweight="bold" if best else "normal", color="white" if scaled[i, j] > 0.6 else "black")
    ax.set_xticks(range(len(names)), [f"{m} (%)" for m in names])
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(models)), models)
    ax.get_yticklabels()[-1].set_fontweight("bold")
    ax.tick_params(length=0)
    ax.text(0.5, -0.06, "Darker = better within each metric; best value in bold", transform=ax.transAxes,
            ha="center", va="top", fontsize=10, color="0.35")
    ax.set_title("(a) Detection performance of each model", loc="left", fontweight="bold", pad=24)

    ax = axs[0, 1]
    # Label offsets in points, chosen so no label covers another point
    offsets = [(0, 10, "center"), (0, -16, "center"), (10, -4, "left"), (8, 8, "left"), (0, 10, "center"),
               (0, 10, "center")]
    for name, p, m, (dx, dy, ha) in zip(models, params, metrics["mAP@0.5"][0], offsets):
        ours = name.startswith("Ours")
        ax.scatter(p, m, s=90, color=VERMILION if ours else GREY, zorder=3)
        ax.annotate(name, (p, m), xytext=(dx, dy), textcoords="offset points", ha=ha, fontsize=10,
                    fontweight="bold" if ours else "normal")
    ax.set_xlim(2.0, 3.6)
    ax.set_ylim(90.4, 95)
    ax.set_xlabel("Parameters (millions)")
    ax.set_ylabel("mAP@0.5 (%)")
    grid(ax)
    panel_title(ax, "b", "Accuracy versus model size")

    for ax, letter, title, ylabel, base, ours, change in (
            (axs[1, 0], "c", "Anthracnose fruit rot: average precision", "Average precision (%)", 81.8, 95.8,
             "+14.0 percentage points"),
            (axs[1, 1], "d", "Powdery leaf: false positives", "False positives (count)", 61, 29, "\u221252.4%")):
        ax.bar([0, 1], [base, ours], width=0.55, color=[GREY, VERMILION])
        for x, v in zip((0, 1), (base, ours)):
            ax.text(x, v, f"{v:g}", ha="center", va="bottom", fontsize=11)
        ax.set_xticks([0, 1], ["Baseline (YOLOv8n)", "Ours (DSLW)"])
        ax.set_ylim(0, max(base, ours) * 1.2)
        ax.set_ylabel(ylabel)
        ax.text(0.5, max(base, ours) * 1.1, change, ha="center", fontsize=11, fontweight="bold")
        grid(ax, "y")
        panel_title(ax, letter, title)

    fig.suptitle("Results of the ablation experiments", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.5)
    fig.savefig(OUT / "fig1_ablation_redesigned.png")
    plt.close(fig)


# ------------------------------------------------------------------ Figure 2: channels before and after pruning (Junjie et al., 2025)
def figure_pruning():
    # (larger, smaller) channel count for each of the 71 layers in network order, read from the original figure.
    # The original legend calls the larger bar "Pruned Model"; pruning removes channels, so larger = before pruning.
    pairs = [(32, 16), (64, 32), (64, 32), (64, 32), (32, 16), (32, 16), (128, 64), (128, 64), (128, 64), (64, 32),
             (64, 32), (64, 32), (64, 32), (256, 128), (256, 128), (256, 128), (128, 64), (128, 64), (128, 64),
             (128, 64), (494, 256), (512, 256), (512, 256), (256, 128), (54, 27), (256, 128), (512, 256), (64, 32),
             (256, 128), (256, 128), (128, 64), (128, 64), (64, 32), (128, 64), (126, 64), (64, 32), (64, 32),
             (128, 64), (256, 128), (212, 128), (125, 64), (128, 64), (256, 128), (512, 256), (512, 256), (256, 128),
             (256, 128), (128, 64), (128, 64), (128, 64), (128, 64), (122, 64), (128, 64), (128, 64), (128, 64),
             (128, 64), (128, 64), (128, 64), (128, 64), (116, 64), (128, 64), (128, 64), (4, 2), (102, 64),
             (128, 64), (128, 64), (4, 2), (128, 64), (128, 64), (128, 64), (4, 2)]
    # Network parts by module number (YOLOv8 layout): backbone model.0-9, neck model.10-21, head model.22
    parts = [("Backbone\n(model.0\u20139)", 1, 27), ("Neck\n(model.10\u201321)", 28, 47), ("Head\n(model.22)", 48, 71)]
    before, after = np.array(pairs).T
    x = np.arange(1, len(before) + 1)
    cum_b, cum_a = np.cumsum(before), np.cumsum(after)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.8), gridspec_kw={"width_ratios": [2.2, 1]})
    for k, (label, lo, hi) in enumerate(parts):
        if k % 2 == 0:
            ax1.axvspan(lo - 0.5, hi + 0.5, color="0.96", lw=0, zorder=0)
        ax1.text((lo + hi) / 2, cum_b[-1] * 1.07, label, ha="center", va="top", fontsize=10, color="0.35")
    ax1.fill_between(x, cum_a, cum_b, step="post", color="#cfe3f1", lw=0, label="Channels removed")
    ax1.step(x, cum_b, where="post", color=GREY, lw=2, label="Before pruning")
    ax1.step(x, cum_a, where="post", color=BLUE, lw=2, label="After pruning")
    for v, colour in ((cum_b[-1], GREY), (cum_a[-1], BLUE)):
        ax1.text(len(x) + 1, v, f"{v:,}", color=colour, fontweight="bold", va="center")
    ax1.set_xlim(0.5, len(x) + 8)
    ax1.set_ylim(0, cum_b[-1] * 1.1)
    ax1.set_xlabel("Layer (network order)")
    ax1.set_ylabel("Cumulative channels")
    ax1.legend(loc="lower right", bbox_to_anchor=(0.98, 0.04), frameon=True, facecolor="white", framealpha=0.9)
    grid(ax1, "y")
    panel_title(ax1, "a", "Cumulative channels through the network")

    totals = [before.sum(), after.sum()]
    ax2.bar([0, 1], totals, width=0.55, color=[GREY, BLUE])
    for xb, v in zip((0, 1), totals):
        ax2.text(xb, v, f"{v:,}", ha="center", va="bottom")
    ax2.set_xticks([0, 1], ["Before pruning", "After pruning"])
    ax2.set_ylim(0, totals[0] * 1.35)
    ax2.set_ylabel("Channels (all layers)")
    grid(ax2, "y")
    panel_title(ax2, "b", "Total channels")
    halved = int((np.abs(after / before * 100 - 50) < 2).sum())
    ax2.text(0.5, 0.96, f"{halved} of {len(before)} layers were cut by half", transform=ax2.transAxes,
             ha="center", va="top")

    fig.suptitle("Channels per batch-norm layer before and after pruning", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / "fig2_pruning_redesigned.png")
    plt.close(fig)


if __name__ == "__main__":
    figure_ablation()
    figure_pruning()
    print(f"Saved Figures 1 and 2 in {OUT.resolve()}")
