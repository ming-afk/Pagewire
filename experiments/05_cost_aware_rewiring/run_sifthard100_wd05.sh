#!/usr/bin/env bash
# The 0.5 ms lambda ladder on SIFT-10M, hard100 cache.
#
# THIS IS THE APPLES-TO-APPLES COUNTERPART OF THE DEEP-10M LADDER.  Every earlier
# SIFT ladder used a cluster cache from make_sift10m_cache.py, while the DEEP
# ladder uses deep10m_hard100_eval_cache.  Those are built differently in a way
# that changes the experiment:
#
#   cache                         sigma_levels[0]  jitter at --sigma-index 0  ceiling
#   deep10m_hard100_eval_cache         0.000          none                     1.0000
#   center_6813465 (SIFT cluster)     30.305          0.1089 * spread          0.9148
#
# With a cluster cache the stream serves a JITTERED ref scored against the
# UNJITTERED ref GT, so a perfect index tops out at ~0.915 and headroom above
# baseline is 0.245.  On hard100 the queries are real and the GT exact, ceiling
# 1.0, headroom 0.435 -- 1.8x more room.  hard100 is also not a spatial cluster
# (center_id = -100): it is the 100 hardest INDIVIDUAL queries, spread ~2x a
# cluster's.
#
# sift10m_hard100_eval_cache mirrors that on SIFT: the 100 queries with the
# lowest measured recall on the un-rewired index, exact top-100 GT,
# sigma_levels[0] = 0.  Its refs average 0.5817 per-query recall against DEEP
# hard100's 0.5652 baseline, so the workloads are matched on difficulty.
#
# Screen, held-out recall@100, --no-rewire vs un-gated:
#
#   workload                    baseline   free     gain
#   DEEP  hard100                0.5674   0.7149  +0.1475
#   SIFT  hard100 (this run)     0.5821   0.6502  +0.0681
#   SIFT  center_6813465         0.6689   0.6966  +0.0277
#   SIFT  center_504814          0.7216   0.7434  +0.0218
#
# Matching the construction more than doubles the SIFT gain and closes the
# DEEP:SIFT ratio from 4.7x to 2.2x.  The residual is a real dataset difference,
# not a settings difference: every learner flag is identical between the two
# ladders, verified by diffing the run commands.
#
# Learner tuple unchanged: unidirectional rewiring, overlay commit, theta 0.05,
# seed 42, 5000 queries, b_rewire 100, ef 100, k 100, beam_width 12, write price
# charged in-harness at 0.5 ms mean per sector.
set -u
BIN=$HOME/DiskANN-edge_protection/build/apps/control_diskann_wdelay
OUT=$HOME/dcu_sifthard100_wd05_logs
SIFT=/mnt/nvme/src_sift/sift10m_R64_pq32
SIFTC=$HOME/hnswlib/cache/realq_gt_sift10m/sift10m_hard100_eval_cache.npz
THETA=0.05
mkdir -p "$OUT"

run_arm() {
  local tag="$1"; shift
  echo "=== $tag start $(date -u +%FT%TZ)  flags: ${*:-none}" | tee -a "$OUT/runner.log"
  sync; sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"; sleep 2
  taskset -c 0 /usr/bin/time -v "$BIN" \
    --cache-npz "$SIFTC" --methods beam_ucb_recall --k 100 --ef 100 \
    --sigma-index 0 --seed 42 --n-queries 5000 --eval-every 500 \
    --b-rewire 100 --no-ood \
    --index-prefix "$SIFT" --beam-width 12 \
    --track-expansion-cost --learner-hot-cap 15643 \
    --hot-node-cache-unbounded --hot-cache-writethrough --hot-cache-lifetime 0 \
    --hot-cache-cap 32768 --hot-cache-lru \
    --expansion-cost-theta "$THETA" "$@" \
    --post-test-n 1000 --post-test-seed 42 \
    --dump-query-results "$OUT/$tag.train.csv" \
    --post-test-results "$OUT/$tag.post.csv" \
    --work-copy "/mnt/nvme/$tag.index" \
    > "$OUT/$tag.out" 2> "$OUT/$tag.err"
  echo "$tag rc=$? done $(date -u +%FT%TZ)" | tee -a "$OUT/runner.log"
  for pat in "TIMEPROF: final" "IODEV: final" "WRITEDELAY: final" "TESTRECALL"; do
    grep -m1 "$pat" "$OUT/$tag.err" | sed "s/^/  /" | tee -a "$OUT/runner.log"
  done
  # sift-rewire-inert-at-priced-tuple: check the learner committed anything at all
  grep -m1 "THETA: rows_changed" "$OUT/$tag.err" | sed "s/^/  first /" | tee -a "$OUT/runner.log"
  grep "rewire: " "$OUT/$tag.err" | tail -1 | sed "s/^/  last /" | tee -a "$OUT/runner.log"
  rm -f "/mnt/nvme/$tag.index"
}

# baseline: no mutation, no writes, so no write price applies
run_arm sh_s42_norewire --no-rewire
# the ladder, unidirectional + overlay commit, writes priced at 0.5 ms
run_arm sh_s42_none --rewire-unidirectional --write-delay-us 1000 --write-delay-prob 0.5
for Q in 0.30 0.50 0.70; do
  run_arm "sh_s42_q${Q}" --rewire-unidirectional --write-delay-us 1000 --write-delay-prob 0.5 \
          --lambda-noise-floor-quantile $Q
done
echo "SIFTHARD100_DONE $(date -u +%FT%TZ)" | tee -a "$OUT/runner.log"
