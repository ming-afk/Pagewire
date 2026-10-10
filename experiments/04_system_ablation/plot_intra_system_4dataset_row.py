#!/usr/bin/env python3
"""Intra-system 4-arm recall-QPS ablation, four datasets in one row.

Panels (left to right), titled with the dataset name only:
  SIFT10M   run_logs/intra_system_sift10m_efsweep_sigma1/   (sigma1 held-out,
            exact GT, 2 shards -- per-shard QPS averaged over w0/w1)
  DEEP10M   run_logs/intra_system_deep10m_efsweep_v1/
  T2I100M   run_logs/t2i100m_normal_efsweep_ef100/          (normal cluster)
  SPACEV1B  run_logs/spacev1b_hard100_4arm_efsweep/         (refs=1/1, sigma=2.0
            -- not the 100-reference spread the "hard100" name implies)

Arms are the same four everywhere; the CSV stems differ per dataset because the
runs were scripted separately. Canonical arm names, and what changes between
them (everything else held fixed within a dataset):

  static / baseline_norewire   rewiring off, layout off
  layout                       rewiring off, --layout-adapt --lambda-pi 0.3
  rewire_audit                 rewiring on (beam_ucb_recall, --edge-protection
                               --protect-audit-rate 0.1 --protect-audit-ef 1),
                               layout off
  rewire+layout / rewire_full  rewiring on + layout on

Every panel gets its own x AND y scale (recall ranges and QPS ranges differ by
dataset); the y axis is log in all three. House style (colors, dotted lines,
markers, grid, margins) follows the standing 8_cost_utility convention recorded
in memory/dcu-plot-style-roargraph-tone.md -- RoarGraph Figure 11 tone, with a
single alpha of 0.9 applied uniformly so no series looks more washed out than
another.

CAVEATS carried from the source runs, unchanged here:
  - SIFT10M: use the `_v2` rewire_audit files; the non-v2 ones are superseded
    (see that run's README.md, correction #3).
  - DEEP10M: this is the v1 sweep, which still has the query-repetition
    artifact (post-test sigma=0 draws with replacement from 100 refs), so its
    absolute QPS is biased high relative to SIFT10M's sigma1 panel. The PNG
    deep10m_efsweep_all4_final_recall_qps.png was drawn from CSVs that are not
    in this repo, so this panel's QPS values do not match that figure.
  - T2I100M: training-query recall (sigma=0); no held-out set for that cluster.
  - SPACEV1B: refs=1/1, sigma=2.0 (single hot reference), not yet rerun at
    --n-refs 100 for a true 100-reference "hard100" workload. ef list differs
    by arm: static/layout swept ef 100-2500, rewire_audit/rewire_full swept
    ef 10-2500 (widened after the original 100-2500 sweep landed entirely on
    the ~0.9205 recall ceiling those two arms hit by ef=150, stacking their
    points in a near-vertical line).
"""
import csv
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
RUN_LOGS = HERE / "run_logs"
FIGURES = HERE / "figures"

# House style, order-based palette (red-orange, orange, muted blue, green) and
# per-series marker shapes, from memory/dcu-plot-style-roargraph-tone.md.
ARM_STYLE = {
    "static": ("#ff5719", "o"),
    "layout": ("#ff9719", "s"),
    "rewire_audit": ("#588ebb", "^"),
    "rewire+layout": ("#3fb53f", "D"),
}
LINESTYLE = (0, (2, 1.6))
ALPHA = 0.9
TEXT_PRIMARY, GRID, SPINE = "#1a1a1a", "#e3e2df", "#8a8a86"
ARM_ORDER = ["static", "layout", "rewire_audit", "rewire+layout"]

PANELS = [
    {
        "title": "SIFT10M",
        "directory": "intra_system_sift10m_efsweep_sigma1",
        # two shards per arm; averaged
        "shards": ["_w0.ef.csv", "_w1.ef.csv"],
        "stems": {
            "static": "static_s1",
            "layout": "layout_s1",
            "rewire_audit": "rewire_audit_s1_v2",
            "rewire+layout": "rewire_layout_s1",
        },
    },
    {
        "title": "DEEP10M",
        "directory": "intra_system_deep10m_efsweep_v1",
        "shards": [".ef.csv"],
        "stems": {
            "static": "static",
            "layout": "layout",
            "rewire_audit": "rewire_audit",
            "rewire+layout": "full",
        },
        # last 8 ef points only, per user request
        "tail": 8,
    },
    {
        "title": "T2I100M",
        "directory": "t2i100m_normal_efsweep_ef100",
        "shards": ["_efsweep.csv"],
        "stems": {
            "static": "baseline_norewire",
            "layout": "layout",
            "rewire_audit": "rewire_audit",
            "rewire+layout": "rewire_full",
        },
    },
    {
        "title": "SPACEV1B",
        "directory": "spacev1b_hard100_4arm_efsweep",
        "shards": ["_efsweep.csv"],
        "stems": {
            "static": "static",
            "layout": "layout",
            "rewire_audit": "rewire_audit",
            "rewire+layout": "rewire_full",
        },
        # Matches the published intra_system_3dataset_row panel: rewire arms from the
        # diverse-query rerun, and each arm's sweep clipped to its lowest shown ef.
        "stems_override": {
            "rewire_audit": "diverse_queries_20260927_raw/rewire_audit",
            "rewire+layout": "diverse_queries_20260927_raw/rewire_full",
        },
        "min_ef": {"static": 300, "layout": 300, "rewire_audit": 75, "rewire+layout": 75},
        "max_ef": {"rewire_audit": 500, "rewire+layout": 500},
    },
]


def read_sweep(path):
    """ef -> (recall, qps) for one CSV."""
    with path.open() as handle:
        return {
            int(row["ef"]): (float(row["recall"]), float(row["qps"]))
            for row in csv.DictReader(handle)
        }


def arm_curve(panel, arm_name):
    """Recall and QPS per ef, averaged over shards when a run was sharded."""
    directory = RUN_LOGS / panel["directory"]
    stem = panel.get("stems_override", {}).get(arm_name, panel["stems"][arm_name])
    per_shard = [read_sweep(directory / f"{stem}{suffix}")
                 for suffix in panel["shards"]]
    minimum_ef = panel.get("min_ef", {}).get(arm_name, 0)
    maximum_ef = panel.get("max_ef", {}).get(arm_name, float("inf"))
    ef_values = sorted(ef for ef in per_shard[0] if minimum_ef <= ef <= maximum_ef)
    recall_values, qps_values = [], []
    for ef in ef_values:
        recalls = [shard[ef][0] for shard in per_shard]
        qpss = [shard[ef][1] for shard in per_shard]
        recall_values.append(sum(recalls) / len(recalls))
        qps_values.append(sum(qpss) / len(qpss))
    tail = panel.get("tail")
    if tail is not None:
        recall_values, qps_values = recall_values[-tail:], qps_values[-tail:]
    return recall_values, qps_values


def main():
    figure, axes = plt.subplots(1, len(PANELS), figsize=(11.0, 2.7))

    for axis, panel in zip(axes, PANELS):
        for arm_name in ARM_ORDER:
            color, marker = ARM_STYLE[arm_name]
            recall_values, qps_values = arm_curve(panel, arm_name)
            axis.plot(recall_values, qps_values, marker=marker, color=color,
                      linestyle=LINESTYLE, linewidth=2.2, alpha=ALPHA,
                      markersize=5.5, markeredgewidth=0, label=arm_name)
        axis.set_yscale("log")
        # Per-panel y scales: a narrow decade range makes matplotlib label the
        # minor log ticks (2x10^1, 3x10^1, ...), which overflows into the
        # neighbouring panel. Keep the minor gridlines, drop their labels.
        axis.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        axis.set_title(panel["title"], fontsize=15, color=TEXT_PRIMARY, pad=2)
        axis.set_axisbelow(True)
        axis.grid(True, which="major", color=GRID, linewidth=0.7)
        axis.tick_params(labelsize=13, colors=TEXT_PRIMARY, pad=1.5)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(SPINE)

    axes[0].set_ylabel("QPS (log)", fontsize=14, color=TEXT_PRIMARY, labelpad=2)
    figure.supxlabel("Recall@100", fontsize=14, color=TEXT_PRIMARY, y=0.14)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(ARM_ORDER),
                  frameon=False, fontsize=14, handlelength=2.6,
                  columnspacing=1.8, borderaxespad=0.0,
                  bbox_to_anchor=(0.5, -0.01))

    figure.subplots_adjust(left=0.065, right=0.995, top=0.90, bottom=0.32,
                           wspace=0.14)
    FIGURES.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(FIGURES / f"intra_system_4dataset_row.{suffix}", dpi=300)
        figure.savefig(FIGURES / f"intra_system_3dataset_row.{suffix}", dpi=300)
    print("wrote", FIGURES / "intra_system_4dataset_row.png")


if __name__ == "__main__":
    main()
