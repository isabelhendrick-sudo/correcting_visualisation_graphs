"""Regenerate Figure 4 from the figure redesign project."""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                     "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "legend.frameon": False,
                     "axes.linewidth": 0.8, "xtick.direction": "in", "ytick.direction": "in", "savefig.dpi": 300})

# ColorBrewer qualitative "Dark2": two unordered groups, colour-blind safe and print friendly
A_COL, B_COL = "#1b9e77", "#7570b3"
cols = {"A": A_COL, "B": B_COL}

years = np.arange(2010, 2020)
refills = {"A": np.array([5.1, 4.8, 5.4, 5.7, 5.0, 5.2, 5.5, 5.7, 4.9, 5.3]),
           "B": np.array([6.5, 4.8, 12.8, 9.1, 7.9, 10.5, 6.4, 7.0, 5.1, 6.9])}
revenue = {"A": np.array([295, 208, 262, 330, 218, 175, 255, 237, 200, 430]),
           "B": np.array([232, 182, 177, 80, 192, 200, 248, 183, 292, 175])}


def panel(ax, data, ylabel, ymax, title, note, note_colour):
    a, b = data["A"], data["B"]
    # Shade the gap in the colour of whichever group is ahead
    ax.fill_between(years, a, b, where=a >= b, color=A_COL, alpha=0.15, lw=0, interpolate=True)
    ax.fill_between(years, a, b, where=a < b, color=B_COL, alpha=0.15, lw=0, interpolate=True)
    for g in "AB":
        ax.plot(years, data[g], "-o", color=cols[g], lw=2.4, ms=5)
        ax.annotate(f"Group {g}", (years[-1], data[g][-1]), xytext=(8, 0), textcoords="offset points",
                    color=cols[g], fontweight="bold", va="center")
    ax.text(2010, ymax * 0.93, note, color=note_colour, fontweight="bold")
    ax.set_xlim(2009.5, 2020.5)
    ax.set_ylim(0, ymax)
    ax.set_xticks(years)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color="0.9", lw=0.6)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontweight="bold")


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
panel(ax1, refills, "Refills per customer", 14.5, "(a) Group B customers refill more often ...",
      "Group B refills more in 9 of 10 years", B_COL)
panel(ax2, revenue, "Revenue per customer (US$)", 480, "(b) ... yet Group A customers bring in more revenue",
      "Group A earns more in 8 of 10 years", A_COL)

for ax in (ax1, ax2):
    ax.set_xlabel("Calendar year")
    ax.tick_params(labelbottom=True)

fig.suptitle("Group B customers refill more prescriptions, but Group A customers bring in more revenue",
             fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2)
out = Path("outputs")
out.mkdir(exist_ok=True)
fig.savefig(out / "fig4_refills_redesigned.png")
plt.close(fig)
print(f"Saved Figure 4 in {out.resolve()}")
