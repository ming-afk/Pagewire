#!/usr/bin/env python3
"""Figure 4 (write-control test plan): T2I-100M hard100, 0.5 ms/write, seed 42.

Method: lambda commit gate (--lambda-noise-floor-quantile), arms
    {none, q0.30, q0.50, q0.70}. Control: --no-rewire (0 writes throughout).
    --rewire-unidirectional, write-delay held fixed. Same tuple as the
    SIFT-10M / DEEP-10M hard100 counterparts
    (plot_dcu_fig4_writecontrol_sifthard100.py,
    plot_dcu_fig4_writecontrol_unihalf.py) except the dataset/index/cache.

Binary: control_diskann_fixed, not control_diskann_wdelay/_unidir -- those
    link an outdated npio.hpp that reinterpret_casts this cache's uint32
    proxy_gt as int64, corrupting every id and segfaulting do_rewire on
    every rewiring arm. control_diskann_fixed is built from the current
    control_diskann.cpp + a dtype-checked npio.hpp. See
    [[t2i100m-hard100-fig4-blocked]].

Caveat carried from that investigation: brute-forced exact MIPS ground
truth confirms the cache's proxy_gt is 100% correct for all 100 refs, so
this is NOT a data bug -- but T2I-100M's hard100 baseline recall (~0.048)
is far below SIFT/DEEP's (~0.57-0.58) at the same construction, because
hard100 selects literally the worst-100 queries and at 100M points (10x
SIFT/DEEP) that tail is apparently much harder at ef=100/beam=12. The two
workloads are not on the same footing, so treat this figure's absolute
recall level as a different regime, not a controlled cross-dataset
comparison -- the write-vs-quality SHAPE within T2I-100M is still valid.

Data: writes = cumulative `writes_rewire` from the per-500-query checkpoint
    table in each arm's .out log (device sectors, post-coalescing, IODEV).
    recall = recomputed offline from each arm's per-query dump (.train.csv,
    5000 rows) as a running mean -- this is TRAIN-QUERY recall, not held-out.

Panel (a): cumulative writes vs. served queries.
Panel (b): recall GAINED OVER --no-rewire vs. cumulative writes. All arms
    share the identical query stream in the identical order, and
    --no-rewire never touches the graph, so subtracting its per-query
    recall before averaging cancels the query-difficulty confound exactly,
    as in the SIFT/DEEP hard100 figures.
"""
import csv
import pathlib
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN_DIR = pathlib.Path(__file__).resolve().parent / "run_logs" / "dcu_t2i100m_hard100_wd05"
ARMS = ["none", "q0.30", "q0.50", "q0.70"]
ARM_LABELS = {"none": "no gate", "q0.30": "lambda q=0.30", "q0.50": "lambda q=0.50", "q0.70": "lambda q=0.70"}
ARM_COLORS = {"none": "#c0392b", "q0.30": "#e08214", "q0.50": "#2a78d6", "q0.70": "#2a9d5c"}
CHECKPOINTS = list(range(500, 5001, 500))
TEXT_PRIMARY, TEXT_SECONDARY, GRID, SURFACE = "#0b0b0b", "#52514e", "#dedddA", "#ffffff"


def parse_out_table(path):
    text = path.read_text()
    rows = re.findall(r"QL-Beam-UCB-Recall,(\d+),(\d+),\d+,\d+,\d+,(\d+),", text)
    writes, edits = {}, {}
    for q, edit, write in rows:
        writes[int(q)] = int(write)
        edits[int(q)] = int(edit)
    return writes, edits


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


def windowed_recall_trajectory(path):
    per_query = per_query_recall(path)
    windowed = {}
    prev = 0
    for q in CHECKPOINTS:
        window = per_query[prev:q]
        windowed[q] = sum(window) / len(window)
        prev = q
    return windowed


def load_arm(arm):
    tag = f"t2i_s42_{arm}"
    writes, edits = parse_out_table(RUN_DIR / f"{tag}.out")
    recall = recall_trajectory(RUN_DIR / f"{tag}.train.csv")
    windowed_recall = windowed_recall_trajectory(RUN_DIR / f"{tag}.train.csv")
    return {
        "writes": [writes[q] for q in CHECKPOINTS],
        "edits": [edits[q] for q in CHECKPOINTS],
        "recall": [recall[q] for q in CHECKPOINTS],
        "windowed_recall": [windowed_recall[q] for q in CHECKPOINTS],
    }


def main():
    data = {arm: load_arm(arm) for arm in ARMS}
    baseline_windowed = windowed_recall_trajectory(RUN_DIR / "t2i_s42_norewire.train.csv")
    baseline_windowed = [baseline_windowed[q] for q in CHECKPOINTS]
    baseline_cumulative = recall_trajectory(RUN_DIR / "t2i_s42_norewire.train.csv")
    baseline_cumulative = [baseline_cumulative[q] for q in CHECKPOINTS]

    figure, (write_axes, quality_axes) = plt.subplots(1, 2, figsize=(14.5, 6.0))
    figure.subplots_adjust(wspace=0.35)

    for arm in ARMS:
        color = ARM_COLORS[arm]
        writes_k = [w / 1000.0 for w in data[arm]["writes"]]
        write_axes.plot(CHECKPOINTS, writes_k, color=color, linewidth=2.0,
                        marker="o", markersize=5, label=ARM_LABELS[arm])

    write_axes.set_xlabel("served queries", fontsize=10, color=TEXT_PRIMARY)
    write_axes.set_ylabel("cumulative writes (k sectors)", fontsize=10, color=TEXT_PRIMARY)
    write_axes.set_title("(a) Write activity", fontsize=11.5, loc="left", color=TEXT_PRIMARY)
    write_axes.grid(True, color=GRID, linewidth=0.8, zorder=0)
    write_axes.set_axisbelow(True)
    write_axes.legend(frameon=False, fontsize=8.5, loc="upper left")
    for side in ("top", "right"):
        write_axes.spines[side].set_visible(False)

    all_gains = []
    for arm in ARMS:
        color = ARM_COLORS[arm]
        writes_k = [w / 1000.0 for w in data[arm]["writes"]]
        gain = [r - b for r, b in zip(data[arm]["recall"], baseline_cumulative)]
        all_gains.extend(gain)
        quality_axes.plot(writes_k, gain, color=color, linewidth=2.0, marker="o",
                          markersize=5, label=ARM_LABELS[arm])
        quality_axes.plot(writes_k[-1], gain[-1], color=color, marker="*",
                          markersize=16, markeredgecolor=SURFACE, markeredgewidth=1.0,
                          zorder=6)

    span = max(all_gains) - min(all_gains)
    quality_axes.set_ylim(min(all_gains) - span * 0.12, max(all_gains) + span * 0.12)
    quality_axes.set_xlabel("cumulative writes (k sectors)", fontsize=10, color=TEXT_PRIMARY)
    quality_axes.set_ylabel("recall gain over --no-rewire", fontsize=10, color=TEXT_PRIMARY)
    quality_axes.set_title("(b) Quality-write tradeoff", fontsize=11.5, loc="left", color=TEXT_PRIMARY)
    quality_axes.grid(True, color=GRID, linewidth=0.8, zorder=0)
    quality_axes.set_axisbelow(True)
    quality_axes.legend(frameon=False, fontsize=7.5, loc="lower right")
    for side in ("top", "right"):
        quality_axes.spines[side].set_visible(False)

    figure.suptitle("T2I-100M hard100, 0.5 ms/write: lambda quantile trades writes for recall",
                    fontsize=12.5, y=1.02)

    out_dir = pathlib.Path(__file__).resolve().parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        figure.savefig(out_dir / f"dcu_fig4_writecontrol_t2i100m.{suffix}", dpi=200,
                       bbox_inches="tight", facecolor="white")
    print(f"wrote {out_dir}/dcu_fig4_writecontrol_t2i100m.pdf and .png")

    print(f"{'arm':14} {'writes@5000':>12} {'recall@5000':>12} {'cum_gain@5000':>14}")
    for arm in ARMS:
        gain5000 = data[arm]["recall"][-1] - baseline_cumulative[-1]
        print(f"{arm:14} {data[arm]['writes'][-1]:12,d} {data[arm]['recall'][-1]:12.4f} "
              f"{gain5000:14.4f}")

    print(f"\nwrites per 500-query window, by arm (checkpoints {CHECKPOINTS}):")
    for arm in ARMS:
        writes = data[arm]["writes"]
        window_writes = [writes[0]] + [b - a for a, b in zip(writes, writes[1:])]
        print(f"{arm:14} " + " ".join(f"{w:8,d}" for w in window_writes))

    print("\nmarginal recall-per-write, by window (window recall gain over --no-rewire, "
          "divided by that window's own writes, x1e6):")
    for arm in ARMS:
        writes = data[arm]["writes"]
        window_writes = [writes[0]] + [b - a for a, b in zip(writes, writes[1:])]
        window_gain = [r - b for r, b in zip(data[arm]["windowed_recall"], baseline_windowed)]
        cells = [f"{g / w * 1e6:8.2f}" if w else "     nan" for g, w in zip(window_gain, window_writes)]
        print(f"{arm:14} " + " ".join(cells))


if __name__ == "__main__":
    main()
