#!/usr/bin/env python3
"""Recall trajectory for free deletion vs. fixed vs. governed edge protection.

Reproduces exp.tex, section "Preserving Routes During Specialization" /
"Governed Edge Protection", Figure "edge_protection_10m_ablation", generated
entirely from run_edge_protection.sh's control_diskann output CSVs (format:
method,q,n,recall). The only prior figure for this ablation came from a
different, external, in-RAM harness (see ../../VIOLATIONS.md item 1); that
harness is not part of this repro, so numbers from this script will not
match that prior figure.

Usage: plot_edge_protection.py [run_logs_dir]
"""
import csv
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
RUN_LOGS = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "run_logs" / "edge_protection"
DATASETS = ["sift10m", "deep10m", "t2i10m"]
BLUE, GREEN, RED, INK, MUTED, GRID = "#2a78d6", "#3a9e52", "#c8384f", "#202124", "#777777", "#dddddd"


def load_recall(path):
    with path.open() as fh:
        rows = [r for r in csv.DictReader(fh) if r["method"] != "SUMMARY"]
    rows.sort(key=lambda r: int(r["q"]))
    return np.array([int(r["q"]) for r in rows]), np.array([float(r["recall"]) for r in rows])


def main():
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(7.15, 2.12), sharex=True)
    for ax, ds in zip(axes, DATASETS):
        q_free, free = load_recall(RUN_LOGS / f"{ds}_free_deletion.csv")
        q_fix, fixed = load_recall(RUN_LOGS / f"{ds}_fixed_protection.csv")
        q_gov, gov = load_recall(RUN_LOGS / f"{ds}_governed.csv")
        assert np.array_equal(q_free, q_gov) and np.array_equal(q_free, q_fix), \
            f"{ds}: checkpoint mismatch between arms"
        ax.plot(q_free, free, color=BLUE, marker="o", ms=2.5, mfc="white", mew=.75, label="Free deletion", zorder=3)
        ax.plot(q_fix, fixed, color=GREEN, marker="^", ms=2.5, mfc="white", mew=.75, label="Fixed protection", zorder=3)
        ax.plot(q_gov, gov, color=RED, marker="s", ms=2.35, mfc=RED, mew=0, label="Governor", zorder=4)
        ax.grid(axis="y", color=GRID, lw=.45)
        ax.set_axisbelow(True)
        ax.set_xlabel("Training queries")
        ax.set_title(f"{ds}\nfinal gain vs. free: {gov[-1]-free[-1]:+.4f}", loc="left", pad=3, color=INK)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("Recall@100")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, 1.03))
    fig.subplots_adjust(left=.07, right=.98, bottom=.20, top=.76, wspace=.30)
    out = HERE / "figures" / "edge_protection_10m_ablation_reproduced.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight", pad_inches=.025)
    print(out)


if __name__ == "__main__":
    main()
