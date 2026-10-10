#!/bin/bash
#SBATCH --account=def-koudas_cpu
#SBATCH --job-name=esci_aba30k_adaptive
#SBATCH --time=2-00:00:00
#SBATCH --tmp=20G
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/scratch/liming76/esci_full/results/esci_aba30k_adaptive_s42_%j.out
#SBATCH --error=/scratch/liming76/esci_full/results/esci_aba30k_adaptive_s42_%j.err

set -e
module load StdEnv/2023 gcc/12.3 cmake/3.31.0 boost/1.85.0 imkl/2024.2.0
module load python/3.11 scipy-stack/2023b
BASE=/scratch/liming76/esci_full
RUNDIR=$BASE/repro_aba30k_s42
mkdir -p "$RUNDIR" "$BASE/results"
cd "$BASE"

# ABA / 30000-queries-per-phase stream from full cluster-9 (A) / cluster-3
# (B), plus 1000-query monitor set for --side-eval-c-query/-gt.
python3 make_esci_cluster_stream.py \
  --base-dir "$BASE" \
  --out-prefix esci_cluster_aba30k \
  --phases ABA --phase-queries 30000 \
  --seed 42 \
  --monitor-queries 1000 --monitor-seed 1000

set +e
# Cache capped at 24,576 sectors (96 MiB), lifetime=50q, no edge-protection
# audits. Protection is a deterministic anneal, not exp_tr.tex's governed
# setup (see ../../VIOLATIONS.md item 8).
/project/def-koudas/liming76/DiskANN/build/apps/control_diskann \
  --index-prefix index_fp/esci_R32to16approx \
  --cache-npz dummy_cache.npz \
  --methods beam_ucb_recall --k 100 --ef 100 --sigma-index 0 --seed 42 \
  --n-queries 90000 --eval-every 250 --b-rewire 500 --beam-width 4 \
  --self-label \
  --hot-node-cache-unbounded --hot-cache-writethrough --hot-cache-cap 24576 \
  --hot-cache-lifetime 50 \
  --edge-protection --protect-k-p-init 8 --protect-k-p-max 16 \
  --protect-anneal-linear 2000 --protect-anneal-step 2 \
  --detector-control \
  --detector-calibration-csv "$RUNDIR/detector.csv" \
  --detector-calibration-window 100 --detector-stop-threshold 0.21 \
  --detector-stop-windows 3 --detector-min-adapt 500 \
  --detector-drift-threshold 0.70 --detector-sample-every 100 \
  --detector-snapshot-thresholds 0,30000,60000 \
  --detector-snapshot-labels A1,B,A2 \
  --detector-snapshot-prefix "$RUNDIR/esci_aba30k_adapted" \
  --ext-query-file esci_cluster_aba30k_queries.bin --true-gt esci_cluster_aba30k_gt.bin \
  --side-eval-c-query esci_cluster_aba30k_side_eval_c.bin \
  --side-eval-c-gt esci_cluster_aba30k_side_eval_c_gt.bin \
  --side-eval-ef 100 \
  --side-eval-csv "$RUNDIR/sideeval.csv" \
  --side-eval-per-query-csv "$RUNDIR/sideeval_perquery.csv" \
  --work-copy "$SLURM_TMPDIR/esci_aba30k_adaptive_s42.index" \
  > "$RUNDIR/run.log" 2>&1
RUN_EXIT_CODE=$?
cp "$SLURM_TMPDIR/esci_aba30k_adaptive_s42.index" "$RUNDIR/esci_aba30k_adaptive_s42_final.index"
echo "RUN_EXIT_CODE=$RUN_EXIT_CODE"
exit $RUN_EXIT_CODE
