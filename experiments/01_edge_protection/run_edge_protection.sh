#!/bin/bash
# Governed Edge Protection repro (exp_tr.tex fig:edge-protection), disk-resident
# control_diskann path. Reconstructed driver; baseline is a fresh run, not the
# separate in-RAM control_vamana_pctprotect harness.
#
# Arms: free_deletion (no protection), fixed_protection (k_p pinned at
# k_p^(0), no governor), governed (--edge-protection, governor on).
# R=64 build, direct I/O work copy, cache 2.5% of dataset size, 5000 queries,
# recall@100.
set -eu

BIN=./build/apps/control_diskann
N_QUERIES=5000
K=100
EF=100
SEED=42
OUT=${OUT:-./run_logs/edge_protection}
mkdir -p "$OUT"

# Fill in per-dataset paths before running.
declare -A INDEX_PREFIX=(
  [sift10m]="FILL_IN_sift10m_R64_pq32_index_prefix"
  [deep10m]="FILL_IN_deep10m_R64_pq32_index_prefix"
  [t2i10m]="FILL_IN_t2i10m_R64_pq32_index_prefix"
)
declare -A CACHE_NPZ=(
  [sift10m]="FILL_IN_sift10m_cache.npz"
  [deep10m]="FILL_IN_deep10m_cache.npz"
  [t2i10m]="FILL_IN_t2i10m_cache.npz"
)
declare -A SIGMA_INDEX=( [sift10m]=1 [deep10m]=0 [t2i10m]=0 )

# k_p^(0) stand-in (Cov(j) calibration unimplemented): R/2 = 32 at R=64.
# K_max = min(R, max(2*k_p^(0), ceil(R/4))) = 64.
declare -A K_P_INIT=( [sift10m]=32 [deep10m]=32 [t2i10m]=32 )
K_MAX=64

# C_blk hot-block cache size, bytes: 0.025 * on-disk record size * n_vectors.
declare -A CBLK_CAP_BYTES=(
  [sift10m]="FILL_IN_bytes"
  [deep10m]="FILL_IN_bytes"
  [t2i10m]="FILL_IN_bytes"
)

for ds in sift10m deep10m t2i10m; do
  COMMON=(--index-prefix "${INDEX_PREFIX[$ds]}" --cache-npz "${CACHE_NPZ[$ds]}"
          --methods beam_ucb_recall --k $K --ef $EF
          --sigma-index "${SIGMA_INDEX[$ds]}" --seed $SEED
          --n-queries $N_QUERIES --eval-every 500 --b-rewire 100 --no-ood
          --hot-node-cache-unbounded --hot-cache-writethrough --hot-cache-lru
          --hot-cache-cap "${CBLK_CAP_BYTES[$ds]}")

  echo "[$ds] free_deletion"
  "$BIN" "${COMMON[@]}" \
    > "$OUT/${ds}_free_deletion.csv" 2> "$OUT/${ds}_free_deletion.log"

  echo "[$ds] fixed_protection"
  "$BIN" "${COMMON[@]}" \
    --edge-protection --protect-fix-k-p --protect-k-p-init "${K_P_INIT[$ds]}" \
    --protect-k-p-max $K_MAX \
    > "$OUT/${ds}_fixed_protection.csv" 2> "$OUT/${ds}_fixed_protection.log"

  echo "[$ds] governed"
  "$BIN" "${COMMON[@]}" \
    --edge-protection --protect-k-p-init "${K_P_INIT[$ds]}" --protect-k-p-max $K_MAX \
    --protect-audit-rate 1.0 --protect-audit-ef 1 --protect-dwell 200 \
    --protect-slack 0.005 --protect-slo 0.11 \
    > "$OUT/${ds}_governed.csv" 2> "$OUT/${ds}_governed.log"
done

echo "Done. Per-checkpoint recall@100 is in"
echo "\$OUT/<dataset>_{free_deletion,fixed_protection,governed}.csv"
echo "(control_diskann stdout format: method,q,n,recall)."
