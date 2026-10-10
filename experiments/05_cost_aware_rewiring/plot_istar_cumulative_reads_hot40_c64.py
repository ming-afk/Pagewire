"""Three-arm hot40 I* read-control plots for DEEP-10M and SIFT-10M.

(a) cumulative training traversal device reads vs #train queries, one line per arm.
(b) per-batch traversal device reads per query at each batch midpoint, I* dashed.

Series parsed from the IOPRICE lines in each arm's .err (100 queries/batch),
checked against the IODEV traversal[r=] checkpoints every 500 queries.
"""
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt

run_logs_directory = Path(__file__).parent / "run_logs"
figure_directory = Path(__file__).parent / "figures"
queries_per_batch = 100

plot_configs = {
    "deep": (
        "DEEP-10M c9, hot40 uncacheable, gamma 20x: 16 / 32 / 64 MB C_blk",
        "istar_cumulative_reads_hot40_c64",
        [
            ("dcu_deep10m_c9_hot40_c32", "hot40_istar40_g20x_c32", 40, "#1baf7a", "I*=40", "-"),
            ("dcu_deep10m_c9_hot40_c64", "hot40_istar20_g20x_c64", 20, "#2a78d6", "I*=20", "-"),
            ("dcu_deep10m_c9_hot40_c64", "hot40_istar10_g20x_c64", 10, "#d6336c", "I*=10", "-"),
        ],
    ),
    "sift": (
        "SIFT-10M, hot40 uncacheable, gamma 20x: 64 MB C_blk",
        "istar_cumulative_reads_sifthard100_c64",
        [
            ("dcu_sift10m_hot40_c64", "sift_hot40_istar40_g20x_c64", 40, "#1baf7a", "I*=40", "-"),
            ("dcu_sift10m_hot40_c64", "sift_hot40_istar20_g20x_c64", 20, "#2a78d6", "I*=20", "-"),
            ("dcu_sift10m_hot40_c64", "sift_hot40_istar10_g20x_c64", 10, "#d6336c", "I*=10", "-"),
        ],
    ),
    "t2i": (
        "T2I-100M, hot40 uncacheable, gamma 20x: 32 / 64 / 128 MB C_blk",
        "istar_cumulative_reads_t2i100m_hot40_c64",
        [
            ("dcu_t2i100m_hot40_c64", "t2i_hot40_istar40_g20x_c32", 40, "#1baf7a", "I*=40", "-"),
            ("dcu_t2i100m_hot40_c64", "t2i_hot40_istar20_g20x_c64", 20, "#2a78d6", "I*=20", "-"),
            ("dcu_t2i100m_hot40_c64", "t2i_hot40_istar10_g20x_c128", 10, "#d6336c", "I*=10", "-"),
        ],
    ),
}
dataset_key = os.environ.get("PLOT_DATASET", "deep")


def load_ioprice_batch_series(log_path):
    rows = []
    for line in open(log_path):
        match = re.search(r"IOPRICE: q=(\d+) io_per_query=([\d.]+).*theta=[\d.e+-]+->([\d.e+-]+)", line)
        if match:
            rows.append((int(match.group(1)), float(match.group(2)), float(match.group(3))))
    return rows


def load_iodev_traversal_checkpoints(log_path):
    checkpoints = {}
    for line in open(log_path):
        match = re.search(r"IODEV: checkpoint q=(\d+) traversal\[r=(\d+)", line)
        if match:
            checkpoints[int(match.group(1))] = int(match.group(2))
    return checkpoints


def centered_moving_mean(values, window):
    radius = window // 2
    return [sum(values[max(0, index - radius):index + radius + 1]) /
            len(values[max(0, index - radius):index + radius + 1])
            for index in range(len(values))]


def draw_series(reads_axis, per_batch_axis, arm_specs, draw_targets=True, smooth_window=1):
    drawn_targets = set()
    for log_directory, arm_name, io_target, color, label, linestyle in arm_specs:
        log_path = run_logs_directory / log_directory / f"{arm_name}.err"
        batch_series = load_ioprice_batch_series(log_path)
        query_counts, cumulative_reads, running_total = [0], [0.0], 0.0
        for query_count, io_per_query, _ in batch_series:
            running_total += io_per_query * queries_per_batch
            query_counts.append(query_count)
            cumulative_reads.append(running_total)
        iodev_checkpoints = load_iodev_traversal_checkpoints(log_path)
        for query_count, reads in zip(query_counts, cumulative_reads):
            if query_count in iodev_checkpoints and abs(reads - iodev_checkpoints[query_count]) > 0.5:
                print(f"WARNING {arm_name} q={query_count} series={reads:.0f} IODEV={iodev_checkpoints[query_count]}")
        reads_axis.plot(query_counts, cumulative_reads, color=color, linewidth=2, linestyle=linestyle, label=label)
        if per_batch_axis is None:
            continue
        per_batch_reads = [r for _, r, _ in batch_series]
        per_batch_axis.plot([q - queries_per_batch / 2 for q, _, _ in batch_series],
                            centered_moving_mean(per_batch_reads, smooth_window),
                            color=color, linewidth=1.5, label=label)
        if draw_targets and io_target not in drawn_targets:
            drawn_targets.add(io_target)
            per_batch_axis.axhline(io_target, color="#1baf7a" if io_target == 40 else color,
                                   linewidth=1, linestyle="--", alpha=0.7)
    for axis in [axis for axis in (reads_axis, per_batch_axis) if axis is not None]:
        axis.set_xlabel("# train queries")
        axis.grid(True, color="#e6e6e3", linewidth=0.6)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
        axis.set_xlim(0, 5300)
        axis.set_ylim(bottom=0)
    reads_axis.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value / 1000:.0f}k"))


if dataset_key == "combined":
    # Single-column (~3.4 in) version: cumulative reads only.
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    figure, axes_row = plt.subplots(1, 3, figsize=(3.4, 1.75))
    labels = {"sift": "SIFT-10M", "deep": "DEEP-10M", "t2i": "T2I-100M"}
    for axis, key in zip(axes_row, ("sift", "deep", "t2i")):
        _, _, arm_specs = plot_configs[key]
        draw_series(axis, None, arm_specs, draw_targets=False)
        for line in axis.get_lines():
            line.set_linewidth(1.4)
        axis.set_title(labels[key], fontsize=9, pad=2)
        axis.set_xlabel("")
        axis.set_xticks([0, 5000])
        axis.set_xticklabels(["0", "5k"])
        axis.locator_params(axis="y", nbins=3)
        axis.tick_params(labelsize=8, length=2, pad=1)
    axes_row[0].set_ylabel("cum. disk reads", fontsize=9)
    figure.supxlabel("# train queries", fontsize=9, y=0.02)
    handles, legend_labels = axes_row[0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=False, fontsize=8.5,
                  bbox_to_anchor=(0.55, 1.03), handlelength=1.2, columnspacing=0.8)
    figure.subplots_adjust(left=0.15, right=0.99, top=0.76, bottom=0.2, wspace=0.55)
    figure_stem = "istar_cumulative_reads_hot40_combined"
else:
    figure_title, figure_stem, arm_specs = plot_configs[dataset_key]
    figure, (reads_axis, per_batch_axis) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    draw_series(reads_axis, per_batch_axis, arm_specs)
    reads_axis.set_ylabel("cumulative traversal disk reads (sectors)")
    reads_axis.set_title("(a) cumulative traversal disk reads", fontsize=10)
    per_batch_axis.set_ylabel("traversal disk reads per query (sectors)")
    per_batch_axis.set_title("(b) per-batch traversal disk reads per query\n(100 queries/batch, at batch midpoint; dashed: I*)", fontsize=10)
    reads_axis.legend(frameon=False, loc="upper left", fontsize=8.5)
    figure.suptitle(figure_title, fontsize=11)
    figure.tight_layout()

for extension in ("png", "pdf"):
    figure.savefig(figure_directory / f"{figure_stem}.{extension}", dpi=160)

if dataset_key != "combined":
    figure.suptitle("")
    reads_axis.set_title("")
    per_batch_axis.set_title("")
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(figure_directory / f"{figure_stem}_nocaptions.{extension}", dpi=160)
