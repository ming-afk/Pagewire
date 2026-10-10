#!/usr/bin/env python3
"""Recall@100 at a fixed search budget as the query distribution concentrates.

x-axis: sigma, the expected L2 norm of the jitter applied to a reference query.
Small sigma = queries repeatedly land near the same 100 reference points (a
highly concentrated workload); large sigma = a diffuse workload.  Train and test
are always disjoint draws, so nothing here is memorisation of a seen query.

The point of the figure: the un-adapted graph and the static layouts are flat in
sigma, while the query-adaptive system's recall rises as the workload
concentrates -- adaptation has more structure to exploit.

Usage:
  plot_sigma_ladder_paper.py --out FIG.pdf --ef 100 \
      --series "DiskANN"  8:56.35 4:56.40 35.8:56.49 \
      --series "PageWire" 8:68.07 4:70.10 35.8:66.00
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SYSTEM_STYLE = {
    "DiskANN":  {"color": "#7f7f7f", "marker": "s", "ls": "--"},
    "MARGO":    {"color": "#0072B2", "marker": "^", "ls": "-"},
    "Starling": {"color": "#D55E00", "marker": "v", "ls": "-"},
    "PageWire": {"color": "#009E73", "marker": "o", "ls": "-"},
}


class SeriesAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        series = getattr(namespace, "series", None) or []
        label, points = values[0], values[1:]
        parsed = sorted((float(p.split(":")[0]), float(p.split(":")[1])) for p in points)
        series.append((label, parsed))
        namespace.series = series


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--ef", default="100")
    parser.add_argument("--series", nargs="+", action=SeriesAction, metavar="LABEL X:Y")
    parser.add_argument("--width", type=float, default=3.5)
    parser.add_argument("--height", type=float, default=2.9)
    args = parser.parse_args()

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.linewidth": 0.7, "grid.linewidth": 0.4,
        "lines.linewidth": 1.4, "lines.markersize": 4.2,
        "figure.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })

    figure, axis = plt.subplots(figsize=(args.width, args.height))
    for label, points in args.series:
        style = SYSTEM_STYLE.get(label, {"color": "#CC79A7", "marker": "D", "ls": "-"})
        axis.plot([x for x, _ in points], [y for _, y in points],
                  marker=style["marker"], linestyle=style["ls"], color=style["color"],
                  label=label, markerfacecolor="white", markeredgewidth=1.1)

    axis.set_xscale("log")
    # Label only the sigma levels actually measured; matplotlib's default log
    # minor ticks collide at these values and render unreadable.
    ticks = sorted({x for _, points in args.series for x, _ in points})
    axis.set_xticks(ticks)
    axis.set_xticklabels([f"{t:g}" for t in ticks])
    axis.minorticks_off()
    axis.set_xlabel(r"query-distribution spread $\sigma$  (smaller = more concentrated)")
    axis.set_ylabel(f"Recall@100 (%) at ef={args.ef}")
    axis.grid(True, which="major", alpha=0.25)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    axis.legend(loc="center left", frameon=True, framealpha=0.95, edgecolor="0.8")
    axis.invert_xaxis()

    figure.tight_layout()
    stem = args.out.rsplit(".", 1)[0]
    for extension in ("pdf", "png"):
        figure.savefig(f"{stem}.{extension}")
    print(f"wrote {stem}.pdf and {stem}.png")


if __name__ == "__main__":
    main()
