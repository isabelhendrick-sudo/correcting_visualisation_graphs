"""Regenerate Figure 3 from the figure redesign project.

Values were read from the original figure images, so they are approximate.
Run:  python redesigned_figures3.py
"""
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


# ------------------------------------------------------------------ Figure 3: dataset distribution (Yuan et al., 2026)
def figure_dataset():
    classes = ["Ca", "PS", "FeM", "FeL", "PM"]
    images = {"Train": [1150, 2483, 2414, 1763, 2432], "Validation": [318, 710, 691, 515, 706],
              "Test": [170, 358, 326, 236, 373]}
    boxes = {"Train": [3030, 20746, 10350, 9223, 25560], "Validation": [885, 5861, 3080, 2703, 7141],
             "Test": [443, 3029, 1389, 1281, 3878]}
    sizes = ["[0, 32]", "(32, 96]", "> 96"]
    by_size = np.array([[0, 49, 4309], [1296, 12379, 15961], [0, 4, 14815], [0, 142, 13064], [1552, 18494, 16632]])
    # ColorBrewer qualitative "Dark2" (3 classes) for the unordered splits: colour-blind safe, print friendly,
    # and no blue, so it never clashes with the sequential Blues used for counts in panel (c)
    split_colours = {"Train": "#1b9e77", "Validation": "#d95f02", "Test": "#7570b3"}

    fig = plt.figure(figsize=(14, 9.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05])
    yy = np.arange(len(classes))[::-1]
    for slot, data, letter, title, xlabel in ((gs[0, 0], images, "a", "Images per class", "Number of images"),
                                              (gs[0, 1], boxes, "b", "Bounding boxes per class",
                                               "Number of bounding boxes")):
        ax = fig.add_subplot(slot)
        for (split, vals), off in zip(data.items(), (0.25, 0, -0.25)):
            ax.barh(yy + off, vals, height=0.24, color=split_colours[split], label=split)
            for y, v in zip(yy + off, vals):
                ax.text(v + max(data["Train"]) * 0.01, y, f"{v:,}", va="center", fontsize=8.5, color="0.25")
        ax.set_xlim(0, max(data["Train"]) * 1.13)
        ax.set_yticks(yy, classes)
        ax.set_xlabel(xlabel)
        grid(ax, "x")
        panel_title(ax, letter, title)
    # Legend stacked in the empty top-right corner of panel (b)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in split_colours.values()]
    ax.legend(handles, split_colours.keys(), loc="upper right", ncol=1)

    ax = fig.add_subplot(gs[1, :])
    # ColorBrewer sequential "Blues" for the counts (low = light, high = dark)
    im = ax.imshow(by_size.T, cmap="Blues", aspect="auto", vmin=0)
    for i in range(by_size.shape[0]):
        for j in range(by_size.shape[1]):
            v = by_size[i, j]
            ax.text(i, j, f"{v:,}", ha="center", va="center", fontsize=11,
                    color="white" if v > by_size.max() * 0.55 else "black")
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(sizes)), sizes)
    ax.set_ylabel("Box size interval (pixels)")
    ax.tick_params(length=0)
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label("Number of bounding boxes")
    panel_title(ax, "c", "Bounding boxes by class and size")

    fig.suptitle("Distribution of strawberry leaf samples by class, dataset split and box size",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.5)
    fig.savefig(OUT / "fig3_dataset_redesigned.png")
    plt.close(fig)


if __name__ == "__main__":
    figure_dataset()
    print(f"Saved Figure 3 in {OUT.resolve()}")
