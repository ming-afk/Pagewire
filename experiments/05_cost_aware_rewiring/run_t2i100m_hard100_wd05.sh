#!/usr/bin/env bash
# T2I-100M counterpart of run_sifthard100_wd05.sh (SIFT-10M) and the DEEP-10M
# unihalf ladder. SAME TEST CONFIG as both: unidirectional rewiring, overlay
# commit, theta 0.05, seed 42, 5000 queries, b_rewire 100, ef 100, k 100,
# beam_width 12, hard100 workload (exact GT, sigma_levels[0]=0 -- the same
# "similar noise" construction as SIFT/DEEP hard100, not a spatial cluster
# cache), write price 0.5 ms/sector mean, charged IN-HARNESS via
# --write-delay-us/--write-delay-prob (same mechanism as the SIFT hard100
# script -- preferred over DEEP's original dm-delay device rig per
# [[price-writes-in-harness-not-device]]: no loop device, no stripe sizing,
# agrees with dm-delay to 0.6%, and the rewire write path here is
# single-threaded synchronous O_DIRECT so the two are equivalent).
#
# BIN is control_diskann_fixed, not the earlier control_diskann_wdelay: the
# latter links a pre-fix npio.hpp that reinterpret_casts this cache's uint32
# proxy_gt as int64, corrupting every id and segfaulting do_rewire (see
# [[t2i100m-hard100-fig4-blocked]]). control_diskann_fixed is built from the
# current control_diskann.cpp + a dtype-checked npio.hpp.
set -u
BIN=$HOME/control_diskann_fixed
OUT=$HOME/dcu_t2i100m_hard100_wd05_logs
T2I=/mnt/nvme/src_t2i100m/t2i100m_R64_pq32
T2IC=$HOME/hnswlib/cache/realq_gt_t2i100m/t2i100m_hard100_eval_cache.npz
THETA=0.05
mkdir -p "$OUT"

run_arm() {
  local tag="$1"; shift
  echo "=== $tag start $(date -u +%FT%TZ)  flags: ${*:-none}" | tee -a "$OUT/runner.log"
  sync; sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"; sleep 2
  taskset -c 0 /usr/bin/time -v "$BIN" \
    --cache-npz "$T2IC" --methods beam_ucb_recall --k 100 --ef 100 \
    --sigma-index 0 --seed 42 --n-queries 5000 --eval-every 500 \
    --b-rewire 100 --no-ood \
    --index-prefix "$T2I" --beam-width 12 \
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
  grep -m1 "THETA: rows_changed" "$OUT/$tag.err" | sed "s/^/  first /" | tee -a "$OUT/runner.log"
  grep "rewire: " "$OUT/$tag.err" | tail -1 | sed "s/^/  last /" | tee -a "$OUT/runner.log"
  rm -f "/mnt/nvme/$tag.index"
}

# baseline: no mutation, no writes, so no write price applies
run_arm t2i_s42_norewire --no-rewire
# the ladder, unidirectional + overlay commit, writes priced at 0.5 ms
run_arm t2i_s42_none --rewire-unidirectional --write-delay-us 1000 --write-delay-prob 0.5
for Q in 0.30 0.50 0.70; do
  run_arm "t2i_s42_q${Q}" --rewire-unidirectional --write-delay-us 1000 --write-delay-prob 0.5 \
          --lambda-noise-floor-quantile $Q
done
echo "T2I100M_HARD100_DONE $(date -u +%FT%TZ)" | tee -a "$OUT/runner.log"
