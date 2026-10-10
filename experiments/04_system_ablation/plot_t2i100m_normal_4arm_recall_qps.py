#!/usr/bin/env python3
"""Recall-QPS, 4 arms, T2I-100M normal cluster, post-test ef 100-2500.

Arms (all else fixed: 4000 queries, sigma=0, b_rewire=50, ef=100/K=100 during
training, C_blk 24576 write-through lifetime 50q, --query-block-buffer):
  baseline_norewire  beam, --no-rewire
  layout             beam, --no-rewire --layout-adapt --lambda-pi 0.3
  rewire_audit       beam_ucb_recall --edge-protection --protect-audit-rate 0.1
                     --protect-audit-ef 1
  rewire_full        rewire_audit + --layout-adapt --lambda-pi 0.3
Recall is training-query recall (sigma=0 repeats the 100 training refs).
"""
import csv
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
RUN_LOGS = HERE / "run_logs" / "t2i100m_normal_efsweep_ef100"
FIGS = HERE / "figures"

ARMS = [
    ("baseline_norewire", "#52514e", "o", "-"),
    ("layout", "#7ab3ec", "s", "-"),
    ("rewire_audit", "#c8384f", "^", "--"),
    ("rewire_full", "#e0a03a", "D", "-"),
]


def sweep(arm_name):
    path = RUN_LOGS / f"{arm_name}_efsweep.csv"
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    return ([float(row["recall"]) for row in rows],
            [float(row["qps"]) for row in rows])


def main():
    figure, axes = plt.subplots(figsize=(6.0, 4.2))
    for arm_name, color, marker, linestyle in ARMS:
        recall_values, qps_values = sweep(arm_name)
        axes.plot(recall_values, qps_values, marker=marker, color=color,
                  linestyle=linestyle, markersize=5, linewidth=1.6,
                  label=arm_name)
    axes.set_yscale("log")
    axes.set_xlabel("recall@100 (training queries, post-test ef 100-2500)")
    axes.set_ylabel("QPS (log scale)")
    axes.set_title("T2I-100M, normal cluster: recall vs QPS, 4 arms")
    axes.grid(True, which="both", alpha=0.25, linewidth=0.5)
    axes.legend(frameon=False, fontsize=9)
    figure.tight_layout()
    FIGS.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(FIGS / f"t2i100m_normal_4arm_recall_qps.{suffix}", dpi=200)


if __name__ == "__main__":
    main()
