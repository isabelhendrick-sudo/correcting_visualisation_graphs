# Redesigning Four Published Graphs

This project uses Python, Matplotlib and NumPy to redraw four published figures. It focuses on making the comparisons easier to read through consistent colours, readable labels, and chart types suited to each question.

## Figures and Python scripts

| Figure | Python source | PNG output |
| --- | --- | --- |
| 1. Strawberry detector ablation experiments | [`redesigned_figures_2.py`](redesigned_figures_2.py) · `figure_ablation()` | [`fig1_ablation_redesigned.png`](outputs/fig1_ablation_redesigned.png) |
| 2. Batch-normalisation channels before and after pruning | [`redesigned_figures_2.py`](redesigned_figures_2.py) · `figure_pruning()` | [`fig2_pruning_redesigned.png`](outputs/fig2_pruning_redesigned.png) |
| 3. Strawberry leaf dataset distribution | [`redesigned_figures3.py`](redesigned_figures3.py) · `figure_dataset()` | [`fig3_dataset_redesigned.png`](outputs/fig3_dataset_redesigned.png) |
| 4. Prescription refills and revenue per customer | [`refills_1a_1b.py`](refills_1a_1b.py) · two-panel plot | [`fig4_refills_redesigned.png`](outputs/fig4_refills_redesigned.png) |

## Reproduce the figures

Run these commands from the repository root using Python 3:

```bash
python -m pip install -r requirements.txt
python redesigned_figures_2.py
python redesigned_figures3.py
python refills_1a_1b.py
```

The scripts use values stored in Python and save the PNGs to `outputs/`. Each command writes only the figures listed for that script above. Font availability can affect spacing and text rendering.

## Figure gallery

### 1. Strawberry detector ablation experiments

The heatmap gives exact plotted values for mAP@0.5, recall and F1-score, with shading scaled within each metric. The scatter plot compares mAP@0.5 with parameter count. Zero-based bars show average precision for anthracnose fruit rot and false positives for powdery leaf. Grey represents the baseline in the direct comparisons and orange highlights DSLW. The label “Ours (DSLW)” refers to the authors of the [original study by Chen et al. (2026)](https://doi.org/10.3390/s26154831).

![Heatmap of detection metrics, scatter plot of mAP versus model size, and two baseline-to-DSLW bar charts](outputs/fig1_ablation_redesigned.png)

### 2. Batch-normalisation channels before and after pruning

Cumulative step lines show how channel counts change across the backbone, neck and head. A pair of bars summarises the reduction from 11,379 to 5,745 channels. The redesign interprets the larger series in the source chart as the before-pruning counts. [Original YOLOv8-DDS study (2025)](https://doi.org/10.1016/j.inpa.2025.07.004)

![Cumulative channel counts across 71 layers and total channels before and after pruning](outputs/fig2_pruning_redesigned.png)

### 3. Strawberry leaf dataset distribution

Panel (a) shows image counts and panel (b) shows bounding-box counts, both by class and dataset split. Panel (c) shows bounding-box counts by class and size interval. The two bar charts share a categorical palette for train, validation and test. The heatmap uses sequential blue for counts. [Original study by Yuan et al. (2026)](https://doi.org/10.1016/j.atech.2026.102102)

![Grouped horizontal bars for images and bounding boxes, plus a heatmap of bounding-box sizes](outputs/fig3_dataset_redesigned.png)

### 4. Prescription refills and revenue per customer

Two line charts align the 2010–2019 time series and give each measure its own axis. Direct labels identify the groups, and shading shows which group leads. Group B had more refills per customer in nine years, with a tie in 2011. Group A had higher revenue per customer in eight years. [Original example by Bala Gupta (2019)](https://www.linkedin.com/pulse/misleading-graphs-bala-gupta/)

![Time series showing more refills for Group B in most years and more revenue per customer for Group A in most years](outputs/fig4_refills_redesigned.png)

## Data and attribution

The scripts plot values transcribed from the source figures. Treat values read from axes as approximate. Figure 3 preserves the shown counts, including two discrepancies between the split totals in panel (b) and the size-bin totals in panel (c). FeL totals 13,207 and 13,206 respectively. PM totals 36,579 and 36,678 respectively. The source links identify the underlying studies and chart example.
