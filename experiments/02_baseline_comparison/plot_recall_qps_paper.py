#!/usr/bin/env python3
"""Publication-quality recall-QPS frontier plot.

One panel per dataset/workload; one curve per system.  Points are annotated with
the search-list parameter (L for Starling/MARGO, ef for the DiskANN-family
systems) so a reader can see the accuracy/throughput operating points directly.

Usage:
  plot_recall_qps_paper.py --out FIG.pdf \
      --panel "SIFT-10M hard100 ($\\sigma$=8)" \
        --curve "DiskANN"  pristine_s1.csv \
        --curve "MARGO"    margo_s1.csv \
        --curve "Starling" starling_s1.csv \
        --curve "PageWire" pagewire_s1.csv
"""
import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Colourblind-safe categorical palette (Okabe-Ito derived), stable per system so
# the same system keeps its colour across every figure in the paper.
SYSTEM_STYLE = {
    "DiskANN":  {"color": "#7f7f7f", "marker": "s", "ls": "--", "zorder": 2},
    "MARGO":    {"color": "#0072B2", "marker": "^", "ls": "-",  "zorder": 3},
    "Starling": {"color": "#D55E00", "marker": "v", "ls": "-",  "zorder": 3},
    "PageWire": {"color": "#009E73", "marker": "o", "ls": "-",  "zorder": 4},
}
FALLBACK = ["#CC79A7", "#E69F00", "#56B4E9", "#F0E442"]


def load(path):
    rows = []
    with open(path) as handle:
        for row in csv.DictReader(handle):
            rows.append({key: float(value) for key, value in row.items()})
    for row in rows:
        if row["recall"] <= 1.0:          # some csvs store a fraction
            row["recall"] *= 100.0
        if "L" not in row and "ef" in row:
            row["L"] = row["ef"]
    rows.sort(key=lambda r: r["recall"])
    return rows


class PanelAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        panels = getattr(namespace, "panels", None) or []
        panels.append({"title": values, "curves": []})
        namespace.panels = panels


class CurveAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        panels = getattr(namespace, "panels", None)
        if not panels:
            parser.error("--curve must follow a --panel")
        panels[-1]["curves"].append(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--panel", action=PanelAction, metavar="TITLE")
    parser.add_argument("--curve", nargs=2, action=CurveAction, metavar=("LABEL", "CSV"))
    parser.add_argument("--annotate", default="PageWire",
                        help="system whose points carry L/ef labels; 'none' to disable")
    parser.add_argument("--xlim", nargs=2, type=float, default=None)
    parser.add_argument("--width", type=float, default=3.5, help="panel width in inches")
    parser.add_argument("--height", type=float, default=3.0)
    args = parser.parse_args()

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.7,
        "grid.linewidth": 0.4,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.2,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })

    panels = args.panels
    count = len(panels)
    figure, axes = plt.subplots(1, count, figsize=(args.width * count, args.height),
                                sharey=True, squeeze=False)
    axes = axes[0]

    for axis, panel in zip(axes, panels):
        recalls = []
        for index, (label, csv_path) in enumerate(panel["curves"]):
            rows = load(csv_path)
            style = SYSTEM_STYLE.get(label, {
                "color": FALLBACK[index % len(FALLBACK)], "marker": "D",
                "ls": "-", "zorder": 3})
            axis.plot([r["recall"] for r in rows], [r["qps"] for r in rows],
                      marker=style["marker"], linestyle=style["ls"],
                      color=style["color"], label=label, zorder=style["zorder"],
                      markerfacecolor="white", markeredgewidth=1.1)
            recalls.extend(r["recall"] for r in rows)
            if label == args.annotate:
                for row in rows:
                    axis.annotate(f"{int(row['L'])}",
                                  (row["recall"], row["qps"]),
                                  textcoords="offset points", xytext=(4, 4),
                                  fontsize=6, color=style["color"])
        axis.set_yscale("log")
        axis.set_xlabel("Recall@100 (\\%)" if plt.rcParams["text.usetex"] else "Recall@100 (%)")
        axis.set_title(panel["title"])
        axis.grid(True, which="major", alpha=0.25, linestyle="-")
        axis.grid(True, which="minor", alpha=0.12, linestyle=":")
        axis.set_axisbelow(True)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        if args.xlim:
            axis.set_xlim(*args.xlim)
        else:
            pad = 0.04 * (max(recalls) - min(recalls))
            axis.set_xlim(min(recalls) - pad, max(recalls) + pad)

    axes[0].set_ylabel("Sustained QPS")
    axes[0].legend(loc="lower left", frameon=True, framealpha=0.95,
                   edgecolor="0.8", handlelength=1.8, borderpad=0.4)

    figure.tight_layout()
    figure.savefig(args.out)
    stem = args.out.rsplit(".", 1)[0]
    for extension in ("pdf", "png"):
        figure.savefig(f"{stem}.{extension}")
    print(f"wrote {stem}.pdf and {stem}.png")


if __name__ == "__main__":
    main()
