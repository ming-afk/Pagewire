#!/usr/bin/env python3
"""Figure 4 (write-control test plan), combined across datasets.

Aggregates the three per-dataset write-control figures
(plot_dcu_fig4_writecontrol_sifthard100.py, _unihalf.py, _t2i100m.py) into a
single 2-row x 3-column grid: columns are datasets, rows are panels (a)
write activity, (b) and (b) quality-write tradeoff.
Data loading is identical to
those three scripts -- see their docstrings for the write/recall
definitions and caveats (train-query recall, IODEV sector counts, T2I-100M's
harder absolute recall regime).

Method / arms are identical across all three datasets: lambda commit gate
(--lambda-noise-floor-quantile) at {none, q0.30, q0.50, q0.70} vs. the
--no-rewire control, same write-delay (0.5 ms/write) and layout settings.
"""
import argparse
import csv
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
plt.rcParams["pdf.fonttype"] = 42

ARMS = ["none", "q0.30", "q0.50", "q0.70"]
LAMBDA_ARMS = ARMS[1:]
ARM_LABELS = {"none": "no gate", "q0.30": "$q$=0.30", "q0.50": "$q$=0.50", "q0.70": "$q$=0.70"}
ARM_COLORS = {"none": "#c0392b", "q0.30": "#e08214", "q0.50": "#2a78d6", "q0.70": "#2a9d5c"}
CHECKPOINTS = list(range(500, 5001, 500))
TEXT_PRIMARY, GRID = "#1a1a1a", "#e3e2df"
LINE_ALPHA = 0.78
LINE_WIDTH = 1.2
MARKER_SIZE = 3.0
FONT_SIZE = 8

DATASETS = [
    {
        "key": "sift",
        "label": "SIFT-10M",
        "run_dir": pathlib.Path(__file__).resolve().parent / "run_logs" / "dcu_sifthard100_wd05",
        "tag": lambda arm: f"sh_s42_{arm}",
        "baseline_tag": "sh_s42_norewire",
    },
    {
        "key": "deep",
        "label": "DEEP-10M",
        "run_dir": pathlib.Path(__file__).resolve().parent / "run_logs" / "dcu_unihalf_lam_raw",
        "tag": lambda arm: f"uh_s42_{arm}",
        "baseline_tag": "norewire/nr_s42",
    },
    {
        "key": "t2i",
        "label": "T2I-100M",
        "run_dir": pathlib.Path(__file__).resolve().parent / "run_logs" / "dcu_t2i100m_hard100_wd05",
        "tag": lambda arm: f"t2i_s42_{arm}",
        "baseline_tag": "t2i_s42_norewire",
    },
]

ISTAR_ARMS = {
    "sift": [
        ("dcu_sift10m_hot40_c64", "sift_hot40_istar40_g20x_c64", "#1baf7a", "I*=40"),
        ("dcu_sift10m_hot40_c64", "sift_hot40_istar20_g20x_c64", "#2a78d6", "I*=20"),
        ("dcu_sift10m_hot40_c64", "sift_hot40_istar10_g20x_c64", "#d6336c", "I*=10"),
    ],
    "deep": [
        ("dcu_deep10m_c9_hot40_c32", "hot40_istar40_g20x_c32", "#1baf7a", "I*=40"),
        ("dcu_deep10m_c9_hot40_c64", "hot40_istar20_g20x_c64", "#2a78d6", "I*=20"),
        ("dcu_deep10m_c9_hot40_c64", "hot40_istar10_g20x_c64", "#d6336c", "I*=10"),
    ],
    "t2i": [
        ("dcu_t2i100m_hot40_c64", "t2i_hot40_istar40_g20x_c32", "#1baf7a", "I*=40"),
        ("dcu_t2i100m_hot40_c64", "t2i_hot40_istar20_g20x_c64", "#2a78d6", "I*=20"),
        ("dcu_t2i100m_hot40_c64", "t2i_hot40_istar10_g20x_c128", "#d6336c", "I*=10"),
    ],
}


def parse_out_table(path):
    text = path.read_text()
    rows = re.findall(r"QL-Beam-UCB-Recall,(\d+),(\d+),\d+,\d+,\d+,(\d+),", text)
    writes = {}
    for q, _edit, write in rows:
        writes[int(q)] = int(write)
    return writes


def per_query_recall(path):
    per_query = []
    with open(path, newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            res = {row[f"res{i}"] for i in range(100)}
            gt = {row[f"gt{i}"] for i in range(100)}
            per_query.append(len(res & gt) / 100.0)
    return per_query


def recall_trajectory(path):
    per_query = per_query_recall(path)
    cumulative = {}
    running = 0.0
    for i, value in enumerate(per_query, start=1):
        running += value
        if i in CHECKPOINTS:
            cumulative[i] = running / i
    return cumulative


def applied_lambda_trajectory(path):
    values = [float(match.group(1)) for match in
              re.finditer(r"LAMBDAGATE: lambda=([\d.eE+-]+)", path.read_text())]
    return [100 * (batch + 2) for batch in range(len(values))], values


def istar_cumulative_reads(log_path):
    values = []
    for line in open(log_path):
        match = re.search(r"IOPRICE: q=(\d+) io_per_query=([\d.]+)", line)
        if match:
            values.append((int(match.group(1)), float(match.group(2))))
    queries, reads, total = [0], [0.0], 0.0
    for query, per_query in values:
        total += 100 * per_query
        queries.append(query)
        reads.append(total)
    return queries, reads


def load_dataset(spec):
    run_dir = spec["run_dir"]
    baseline_cumulative = recall_trajectory(run_dir / f"{spec['baseline_tag']}.train.csv")
    baseline_cumulative = [baseline_cumulative[q] for q in CHECKPOINTS]

    arms = {}
    for arm in ARMS:
        tag = spec["tag"](arm)
        writes = parse_out_table(run_dir / f"{tag}.out")
        recall = recall_trajectory(run_dir / f"{tag}.train.csv")
        arms[arm] = {
            "writes": [writes[q] / 1000.0 for q in CHECKPOINTS],
            "gain": [recall[q] - b for q, b in zip(CHECKPOINTS, baseline_cumulative)],
        }
        if arm in LAMBDA_ARMS:
            arms[arm]["lambda_queries"], arms[arm]["lambda"] = applied_lambda_trajectory(
                run_dir / f"{tag}.err")
    return arms


def style_axes(axes):
    axes.grid(True, color=GRID, linewidth=0.7, zorder=0)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color("#8a8a86")
    axes.tick_params(labelsize=7, color="#8a8a86", length=2, pad=1)
    axes.tick_params(which="minor", left=False, bottom=False)


def main(include_istar=False, paper_layout=False, output_dir=None):
    data = {spec["key"]: load_dataset(spec) for spec in DATASETS}

    figure, axes_grid = plt.subplots(
        3 if include_istar else 2, 3, figsize=(3.4, 4.5 if include_istar else 2.6), sharey="row",
        gridspec_kw={"wspace": 0.12, "hspace": 0.80 if include_istar else 0.62},
    )

    for col, spec in enumerate(DATASETS):
        arms = data[spec["key"]]
        write_axes = axes_grid[0, col]
        quality_axes = axes_grid[1, col]

        write_axes.set_title(spec["label"], fontsize=FONT_SIZE, color=TEXT_PRIMARY, pad=2)
        for arm in ARMS:
            color = ARM_COLORS[arm]
            write_axes.plot(CHECKPOINTS, arms[arm]["writes"], color=color,
                             linewidth=LINE_WIDTH, alpha=LINE_ALPHA, label=ARM_LABELS[arm])
        style_axes(write_axes)

        for arm in ARMS:
            color = ARM_COLORS[arm]
            quality_axes.plot(arms[arm]["writes"], arms[arm]["gain"], color=color,
                               linewidth=LINE_WIDTH, alpha=LINE_ALPHA)
            quality_axes.plot(arms[arm]["writes"][-1], arms[arm]["gain"][-1], color=color,
                               marker="*", markersize=6, alpha=LINE_ALPHA,
                               markeredgecolor="white", markeredgewidth=0.6, zorder=6)
        quality_axes.axhline(0.0, color="#8a8a86", linewidth=0.7, linestyle="--", zorder=0)
        style_axes(quality_axes)
        write_axes.set_xlim(0, 5300)
        write_axes.set_xticks([0, 5000])
        write_axes.set_xticklabels(["0", "5k"])
        quality_axes.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=2, integer=True))


    axes_grid[0, 0].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=3))
    axes_grid[1, 0].set_yticks([0.0, 0.1])
    axes_grid[0, 0].set_ylabel("writes (k)", fontsize=FONT_SIZE, color=TEXT_PRIMARY)
    axes_grid[1, 0].set_ylabel("recall gain", fontsize=FONT_SIZE, color=TEXT_PRIMARY)
    # One x-label per row, centred under the middle column.
    axes_grid[0, 1].set_xlabel("served queries", fontsize=FONT_SIZE, color=TEXT_PRIMARY, labelpad=1)
    axes_grid[1, 1].set_xlabel("cumulative writes (k sectors)", fontsize=FONT_SIZE, color=TEXT_PRIMARY, labelpad=1)
    for col in range(1, 3):
        plt.setp(axes_grid[0, col].get_yticklabels(), visible=False)
        plt.setp(axes_grid[1, col].get_yticklabels(), visible=False)

    axes_grid[0, 0].text(-0.62, 1.12, "(a)", transform=axes_grid[0, 0].transAxes,
                          fontsize=FONT_SIZE, fontweight="bold", color=TEXT_PRIMARY, ha="left")
    axes_grid[1, 0].text(-0.62, 1.12, "(b)", transform=axes_grid[1, 0].transAxes,
                          fontsize=FONT_SIZE, fontweight="bold", color=TEXT_PRIMARY, ha="left")

    if include_istar:
        for col, spec in enumerate(DATASETS):
            axis = axes_grid[2, col]
            for directory, arm, color, label in ISTAR_ARMS[spec["key"]]:
                queries, reads = istar_cumulative_reads(
                    pathlib.Path(__file__).resolve().parent / "run_logs" / directory / f"{arm}.err")
                axis.plot(queries, reads, color=color, linewidth=LINE_WIDTH, label=label)
            style_axes(axis)
            axis.set_xlim(0, 5300)
            axis.set_xticks([0, 5000])
            axis.set_xticklabels(["0", "5k"])
            axis.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
                lambda value, _: f"{value / 1000:.0f}k"))
        axes_grid[2, 0].set_ylim(bottom=0)
        axes_grid[2, 0].set_ylabel("cum. disk reads", fontsize=FONT_SIZE, color=TEXT_PRIMARY)
        axes_grid[2, 1].set_xlabel("# train queries", fontsize=FONT_SIZE, color=TEXT_PRIMARY, labelpad=1)
        for col in range(1, 3):
            plt.setp(axes_grid[2, col].get_yticklabels(), visible=False)
        axes_grid[2, 0].text(-0.62, 1.12, "(c)", transform=axes_grid[2, 0].transAxes,
                              fontsize=FONT_SIZE, fontweight="bold", color=TEXT_PRIMARY, ha="left")
        istar_handles, istar_labels = axes_grid[2, 0].get_legend_handles_labels()
        figure.legend(istar_handles, istar_labels, loc="upper center", ncol=3, frameon=False,
                      fontsize=7.5, bbox_to_anchor=(0.55, 0.34), handlelength=1.2,
                      columnspacing=0.8, handletextpad=0.4)

    handles, labels = axes_grid[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=7.5,
                  bbox_to_anchor=(0.55, 0.995), handlelength=1.2, columnspacing=0.8, handletextpad=0.4)

    figure.subplots_adjust(left=0.17, right=0.98, top=0.88 if include_istar else 0.83,
                           bottom=0.09 if include_istar else 0.13)

    if paper_layout:
        figure.set_size_inches(3.4, 2.95 if include_istar else 2.0)
        for row, axes in enumerate(axes_grid):
            for col, axis in enumerate(axes):
                axis.set_position([0.15 + 0.28 * col, [0.69, 0.40, 0.065][row], 0.25, 0.19])
                axis.tick_params(labelsize=4.8)
                axis.title.set_fontsize(5.5)
                axis.xaxis.label.set_fontsize(5.5)
                axis.yaxis.label.set_fontsize(5.5)
                for label in axis.texts:
                    label.set_fontsize(5.5)
                    label.set_position((-0.50, 1.10))
                for line in axis.lines:
                    line.set_linewidth(0.75)
        for legend in figure.legends:
            for label in legend.get_texts():
                label.set_fontsize(5.2)
        if include_istar:
            figure.legends[0].set_bbox_to_anchor((0.55, 0.30))

    out_dir = pathlib.Path(output_dir) if output_dir else (pathlib.Path(__file__).resolve().parent / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        stem = "dcu_fig4_writecontrol_istar_combined" if include_istar else "dcu_fig4_writecontrol_combined"
        figure.savefig(out_dir / f"{stem}.{suffix}", dpi=220,
                        facecolor="white")
    print(f"wrote {out_dir}/{stem}.pdf and .png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-istar", action="store_true",
                        help="append the I* ablation as panel (c)")
    parser.add_argument("--paper-layout", action="store_true")
    parser.add_argument("--output-dir")
    main(**vars(parser.parse_args()))
