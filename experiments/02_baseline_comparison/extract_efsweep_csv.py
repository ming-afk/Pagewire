#!/usr/bin/env python3
"""Turn a control_diskann run.err ef-sweep into the recall-QPS csv the plotter reads.

Each `EFSWEEP: ef=<L>` line is followed by the `TESTRECALL:` line it produced, so
the two are paired in order.  Emits the same columns the Starling/MARGO sweeps
use, so every system's csv is interchangeable in the plot script.

Usage: extract_efsweep_csv.py RUN.err OUT.csv [BEAMWIDTH]
"""
import re
import sys

EFSWEEP = re.compile(r"EFSWEEP:\s*ef=(\d+)")
TESTRECALL = re.compile(r"TESTRECALL:.*recall@100=([0-9.]+).*qps=([0-9.]+)")


def main():
    err_path, out_path = sys.argv[1:3]
    beamwidth = sys.argv[3] if len(sys.argv) > 3 else "8"

    rows = []
    pending_ef = None
    with open(err_path, errors="replace") as handle:
        for line in handle:
            ef_match = EFSWEEP.search(line)
            if ef_match:
                pending_ef = int(ef_match.group(1))
                continue
            if pending_ef is None:
                continue
            recall_match = TESTRECALL.search(line)
            if recall_match:
                recall = float(recall_match.group(1)) * 100.0
                qps = float(recall_match.group(2))
                rows.append((pending_ef, qps, recall))
                pending_ef = None

    if not rows:
        sys.exit(f"no EFSWEEP/TESTRECALL pairs found in {err_path}")

    with open(out_path, "w") as handle:
        handle.write("L,beamwidth,qps,mean_latency_us,recall\n")
        for ef, qps, recall in rows:
            handle.write(f"{ef},{beamwidth},{qps:.2f},{1e6 / qps:.2f},{recall:.4f}\n")
    print(f"wrote {out_path} ({len(rows)} points)")


if __name__ == "__main__":
    main()
