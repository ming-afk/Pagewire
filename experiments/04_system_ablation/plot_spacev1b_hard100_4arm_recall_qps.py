#!/usr/bin/env python3
"""Recall-QPS, 4 arms, SPACEV1B-1B hard100 workload.

post-test ef list: static/layout use 100-2500 (original sweep); rewire_audit/
rewire_full use 10-2500 (widened with 10,25,50,75 ahead of 100, added to
resolve the recall-vs-ef curve below ef=100 -- the original 100-2500 sweep
landed entirely on/after the ~0.9205 recall ceiling those two arms hit by
ef=150, so their points stacked in a near-vertical line at high recall; the
widened sweep shows the actual rising curve from recall 0.14 (ef=10) up to
the same ceiling).

Arms (all else fixed: 2000 training queries, refs=1/1, sigma=2, hot-cache-cap
24576 write-through lifetime 50q, --query-block-buffer; post-test n=500,
true GT):
  static        --no-rewire
  layout        --no-rewire --layout-adapt --lambda-pi 0.3
  rewire_audit  beam_ucb_recall --edge-protection --protect-audit-rate 0.1
                --protect-audit-ef 1
  rewire_full   rewire_audit + --layout-adapt --lambda-pi 0.3
Recall is held-out test-query recall (500 queries, true GT, disjoint from
training).
"""
import csv
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
RUN_LOGS = HERE / "run_logs" / "spacev1b_hard100_4arm_efsweep"
FIGS = HERE / "figures"

ARMS = [
    ("static", "#52514e", "o", "-"),
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
    axes.set_xlabel("recall@100 (held-out, true GT; ef 100-2500 static/layout, "
                     "10-2500 rewire arms)")
    axes.set_ylabel("QPS (log scale)")
    axes.set_title("SPACEV1B-1B, hard100: recall vs QPS, 4 arms")
    axes.grid(True, which="both", alpha=0.25, linewidth=0.5)
    axes.legend(frameon=False, fontsize=9)
    figure.tight_layout()
    FIGS.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        figure.savefig(FIGS / f"spacev1b_hard100_4arm_efsweep.{suffix}", dpi=200)


if __name__ == "__main__":
    main()
