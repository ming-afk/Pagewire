"""ESCI A1->B->A2 (30k queries/phase): PageWire vs DiskANN, both with C_blk capped
at 96 MB (24576 sectors, freeze, lifetime 50).

One figure, shared x-axis:
  top    training-query recall@100 (batch means) vs. # queries;
  middle cumulative-time gap PageWire - DiskANN since query 1;
  bottom per-query cost (ms/query) per 500-query window.
Time comes from TIMEPROF checkpoint
train_s with side-eval C wall time and audit time removed (same accounting as
esci_aba30k_goal_check.py).
Phase transitions and detector adaptation start/stop are marked.
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.patches
import matplotlib.lines

here = Path(__file__).parent
log_directory = here / "run_logs" / "esci_aba30k"
pagewire_log_path = log_directory / "coloc_cap96mb_short6k_run.log"   # job 63721885
diskann_log_path = log_directory / "diskann_cap96mb_run.log"          # job 63721530
phase_queries = 30000


def read_train_recall_series(text):
    cumulative_queries, batch_recalls, query_count = [], [], 0
    for recall, batch_size in re.findall(r"TRAINRECALL: batch_mean=([0-9.]+) n=(\d+)", text):
        query_count += int(batch_size)
        cumulative_queries.append(query_count)
        batch_recalls.append(float(recall))
    return cumulative_queries, batch_recalls


def read_cumulative_accounted_seconds(text):
    checkpoint_queries, accounted_seconds = [0], [0.0]
    side_eval_seconds_so_far = 0.0
    for line in text.splitlines():
        side_eval_match = re.search(r"SIDEEVAL: C .*?wall_s=([0-9.]+)", line)
        if side_eval_match:
            side_eval_seconds_so_far += float(side_eval_match.group(1))
            continue
        checkpoint_match = re.search(r"TIMEPROF: checkpoint q=(\d+) train_s=([0-9.]+).*?audit=([0-9.]+)", line)
        if checkpoint_match:
            query, train_seconds, audit_seconds = checkpoint_match.groups()
            checkpoint_queries.append(int(query))
            accounted_seconds.append(float(train_seconds) - side_eval_seconds_so_far - float(audit_seconds))
    return checkpoint_queries, accounted_seconds


def read_adaptation_intervals(text):
    events = re.findall(r"DETECTOR_EVENT,event=(DRIFT_START|ADAPT_STOP),q=(\d+)", text)
    intervals, start_query = [], 0  # adaptation is on from query 0
    for event_name, query in events:
        if event_name == "ADAPT_STOP" and start_query is not None:
            intervals.append((start_query, int(query)))
            start_query = None
        elif event_name == "DRIFT_START":
            start_query = int(query)
    return intervals


pagewire_text = pagewire_log_path.read_text()
diskann_text = diskann_log_path.read_text()
adaptation_intervals = read_adaptation_intervals(pagewire_text)


plt.rcParams.update({"font.size": 7.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "xtick.major.size": 2, "ytick.major.size": 2})
bin_queries = 1000  # coarser than the 250/500-query log batches, to fit one column


def mark_phases_and_adaptation(axis, with_phase_labels=False):
    for phase_index, phase_label in enumerate(["A1 (cluster9)", "B (cluster3)", "A2 (cluster9)"]):
        if with_phase_labels:
            axis.text((phase_index + 0.5) * phase_queries, 1.02, phase_label, ha="center", va="bottom",
                      fontsize=6.5, color="0.25", transform=axis.get_xaxis_transform())
        if phase_index:
            axis.axvline(phase_index * phase_queries, color="black", linestyle="--", linewidth=0.7)
    for start_query, stop_query in adaptation_intervals:
        axis.axvspan(start_query, stop_query, color="tab:orange", alpha=0.45, linewidth=0)
    axis.set_xlim(0, 3 * phase_queries)


def bin_recall_series(queries, recalls):
    binned_sums, binned_counts, previous_query = {}, {}, 0
    for query, recall in zip(queries, recalls):
        bin_index = (query - 1) // bin_queries
        binned_sums[bin_index] = binned_sums.get(bin_index, 0.0) + recall * (query - previous_query)
        binned_counts[bin_index] = binned_counts.get(bin_index, 0) + (query - previous_query)
        previous_query = query
    bin_indices = sorted(binned_sums)
    return [(i + 1) * bin_queries for i in bin_indices], [binned_sums[i] / binned_counts[i] for i in bin_indices]


pagewire_checkpoints, pagewire_seconds = read_cumulative_accounted_seconds(pagewire_text)
diskann_checkpoints, diskann_seconds = read_cumulative_accounted_seconds(diskann_text)
pagewire_seconds_at = dict(zip(pagewire_checkpoints, pagewire_seconds))
diskann_seconds_at = dict(zip(diskann_checkpoints, diskann_seconds))
grid_queries = list(range(0, 3 * phase_queries + 1, bin_queries))
gap_seconds = [pagewire_seconds_at[q] - diskann_seconds_at[q] for q in grid_queries]
pagewire_ms_per_query = [(pagewire_seconds_at[b] - pagewire_seconds_at[a]) / (b - a) * 1000
                         for a, b in zip(grid_queries, grid_queries[1:])]
diskann_ms_per_query = [(diskann_seconds_at[b] - diskann_seconds_at[a]) / (b - a) * 1000
                        for a, b in zip(grid_queries, grid_queries[1:])]

figure, (recall_axis, gap_axis, cost_axis) = plt.subplots(
    3, 1, figsize=(3.35, 2.48), sharex=True, gridspec_kw={"hspace": 0.10})

pagewire_line, = recall_axis.plot(*bin_recall_series(*read_train_recall_series(pagewire_text)),
                                  color="tab:blue", linewidth=0.9)
diskann_line, = recall_axis.plot(*bin_recall_series(*read_train_recall_series(diskann_text)),
                                 color="0.45", linewidth=0.9)
recall_axis.set_ylim(0.78, 0.90)
recall_axis.set_yticks([0.80, 0.85, 0.90])
recall_axis.set_ylabel("Recall@100")
mark_phases_and_adaptation(recall_axis, with_phase_labels=True)

gap_axis.axhline(0, color="black", linewidth=0.5)
gap_axis.plot(grid_queries, gap_seconds, color="tab:purple", linewidth=0.9)
gap_axis.fill_between(grid_queries, gap_seconds, 0, where=[g > 0 for g in gap_seconds],
                      color="tab:red", alpha=0.15, interpolate=True, linewidth=0)
gap_axis.fill_between(grid_queries, gap_seconds, 0, where=[g <= 0 for g in gap_seconds],
                      color="tab:green", alpha=0.25, interpolate=True, linewidth=0)
gap_axis.set_ylim(-6, 21)
gap_axis.set_yticks([0, 10, 20])
gap_axis.set_ylabel("Time gap (s)")
mark_phases_and_adaptation(gap_axis)

cost_axis.step(grid_queries, pagewire_ms_per_query + pagewire_ms_per_query[-1:], where="post",
               color="tab:blue", linewidth=0.9)
cost_axis.step(grid_queries, diskann_ms_per_query + diskann_ms_per_query[-1:], where="post",
               color="0.45", linewidth=0.9)
cost_axis.set_yscale("log")
cost_axis.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
cost_axis.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
cost_axis.set_yticks([2, 5, 10, 20])
cost_axis.set_ylim(1.4, 30)
cost_axis.set_ylabel("ms/query")
mark_phases_and_adaptation(cost_axis)
cost_axis.set_xticks([0, 30000, 60000, 90000])
cost_axis.set_xticklabels(["0", "30k", "60k", "90k"])
cost_axis.set_xlabel("")

for axis in (recall_axis, gap_axis, cost_axis):
    axis.yaxis.set_label_coords(-0.11, 0.5)
figure.legend([pagewire_line, diskann_line, matplotlib.patches.Patch(color="tab:orange", alpha=0.45),
               matplotlib.lines.Line2D([], [], color="black", linestyle="--", linewidth=0.7)],
              ["PageWire", "DiskANN", "adapting", "cluster switch"],
              loc="lower center", ncol=4, fontsize=6, frameon=False, handlelength=1.4,
              columnspacing=0.8, bbox_to_anchor=(0.55, 0.0))
figure.subplots_adjust(left=0.17, right=0.97, top=0.90, bottom=0.17)
figure_stem = here / "figures" / "esci_aba30k_cap96mb_recall_gap_cost"
figure.savefig(figure_stem.with_suffix(".png"), dpi=300)
figure.savefig(figure_stem.with_suffix(".pdf"))
print("adaptation intervals:", adaptation_intervals)
print(figure_stem.with_suffix(".png"))
