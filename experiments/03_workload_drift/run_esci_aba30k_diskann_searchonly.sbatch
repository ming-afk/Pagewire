#!/bin/bash
#SBATCH --account=def-koudas_cpu
#SBATCH --job-name=esci_aba30k_diskann
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --tmp=20G
#SBATCH --output=/scratch/liming76/esci_full/results/esci_aba30k_diskann_searchonly_%j.out
#SBATCH --error=/scratch/liming76/esci_full/results/esci_aba30k_diskann_searchonly_%j.err

module load StdEnv/2023 gcc/12.3 cmake/3.31.0 boost/1.85.0 imkl/2024.2.0
BASE=/scratch/liming76/esci_full
RUNDIR=$BASE/repro_aba30k_diskann_searchonly
mkdir -p "$RUNDIR"
cd "$BASE"

# Vanilla DiskANN control for repro_aba30k_s42: identical index, stream, ef,
# beam, hot-node cache; --search-only disables all learning/rewiring. Stream
# file (esci_cluster_aba30k_*) is written by run_esci_aba30k_adaptive.sbatch's
# make_esci_cluster_stream.py call -- run that one first.
/project/def-koudas/liming76/DiskANN/build/apps/control_diskann \
  --index-prefix index_fp/esci_R32to16approx \
  --cache-npz dummy_cache.npz \
  --methods beam_ucb_recall --k 100 --ef 100 --sigma-index 0 --seed 42 \
  --n-queries 90000 --eval-every 250 --b-rewire 500 --beam-width 4 \
  --self-label --search-only \
  --hot-node-cache-unbounded --hot-cache-writethrough --hot-cache-cap 24576 \
  --hot-cache-lifetime 50 \
  --ext-query-file esci_cluster_aba30k_queries.bin --true-gt esci_cluster_aba30k_gt.bin \
  --work-copy "$SLURM_TMPDIR/esci_aba30k_diskann.index" \
  > "$RUNDIR/run.log" 2>&1
echo "RUN_EXIT_CODE=$?"
