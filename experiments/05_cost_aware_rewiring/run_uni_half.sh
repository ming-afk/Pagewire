#!/usr/bin/env bash
# Lambda ladder under UNIDIRECTIONAL rewiring, OVERLAY commit, on a device that
# charges an EFFECTIVE 0.5 ms per write.
#
# dm-delay takes integer milliseconds only (CONFIG_HZ=1000), so 0.5 ms cannot be
# asked for directly -- dmsetup rejects a 0.5 argument outright.  Instead the
# table charges 1 ms to exactly half the address space, in alternating 4 MB
# stripes, with reads undelayed everywhere.  The work copy is a nearly
# contiguous 6.8 GB file, so it covers alternating stripes evenly and the
# charged fraction is 0.5 by construction rather than in expectation.
#
# Measured on this box with fio randwrite 4k, direct, no sync:
#   linear (no delay)        27.8 us/write
#   1 ms on every write    1000.2 us/write
#   50% striped, 4 MB       509.3 us/write   <- what this script uses
# Coarser stripes drift off 500 us (512 MB -> 465, 2 GB -> 612) because they
# sample the address space unevenly, so the stripe stays at 4 MB.
#
# fio must NOT run with --sync=1 when measuring this: that pulls ext4 journal
# traffic and flushes through the same delay and reports 65 ms/write for the
# striped table and 4 ms for the full one.  The rewire path issues O_DIRECT
# writes without a per-write fsync, which is why the undelayed-vs-1ms runs come
# out at ~0.95 ms/sector, matching the no-sync probe.
#
# The work copy (6.8 GB) is staged with the delay OFF and the delay is switched
# in once the copy is complete: a delayed copy would be charged identically to
# every arm and would add ~14 minutes to each.
set -u
BIN=$HOME/DiskANN-edge_protection/build/apps/control_diskann_unidir
OUT=$HOME/dcu_unihalf_logs
DEEP=/mnt/nvme/src_deep/deep10m_R64_pq32
DEEPC=$HOME/hnswlib/cache/realq_gt_deep10m/deep10m_hard100_eval_cache.npz
DELAY_MS=1
STRIPE_SECTORS=8192   # 4 MB
THETA=0.05
mkdir -p "$OUT"

sudo umount /mnt/delayed 2>/dev/null; sudo dmsetup remove delayed 2>/dev/null
sudo losetup -D 2>/dev/null; sleep 1
rm -f /mnt/nvme/delaystore.img; fallocate -l 24G /mnt/nvme/delaystore.img
LOOP=$(sudo losetup --find --show /mnt/nvme/delaystore.img)
SECTORS=$(sudo blockdev --getsz "$LOOP")
python3 $HOME/gen_delay_table.py "$LOOP" "$SECTORS" "$STRIPE_SECTORS" "$DELAY_MS" linear  > /tmp/tbl.linear
python3 $HOME/gen_delay_table.py "$LOOP" "$SECTORS" "$STRIPE_SECTORS" "$DELAY_MS" striped > /tmp/tbl.striped
sudo dmsetup create delayed /tmp/tbl.linear
sudo mkfs.ext4 -q -F /dev/mapper/delayed
sudo mkdir -p /mnt/delayed && sudo mount -o noatime /dev/mapper/delayed /mnt/delayed
sudo chown -R ubuntu:ubuntu /mnt/delayed
echo "loop=$LOOP  striped table targets=$(wc -l < /tmp/tbl.striped)  effective 509 us/write" \
  | tee -a "$OUT/runner.log"

set_table() { sudo dmsetup suspend delayed; sudo dmsetup reload delayed "$1"; sudo dmsetup resume delayed; }
SRC_BYTES=$(stat -c %s "$DEEP"_disk.index)

run_arm() {
  local tag="$1"; shift
  set_table /tmp/tbl.linear
  echo "=== $tag start $(date -u +%FT%TZ)  flags: ${*:-none}" | tee -a "$OUT/runner.log"
  sync; sudo sh -c "echo 3 > /proc/sys/vm/drop_caches"; sleep 2
  taskset -c 0 /usr/bin/time -v "$BIN" \
    --cache-npz "$DEEPC" --methods beam_ucb_recall --k 100 --ef 100 \
    --sigma-index 0 --n-queries 5000 --eval-every 500 \
    --b-rewire 100 --no-ood \
    --index-prefix "$DEEP" --beam-width 12 \
    --track-expansion-cost --learner-hot-cap 15643 \
    --hot-node-cache-unbounded --hot-cache-writethrough --hot-cache-lifetime 0 \
    --hot-cache-cap 32768 --hot-cache-lru \
    --expansion-cost-theta "$THETA" --rewire-unidirectional "$@" \
    --post-test-n 1000 --post-test-seed 42 \
    --dump-query-results "$OUT/$tag.train.csv" \
    --post-test-results "$OUT/$tag.post.csv" \
    --work-copy "/mnt/delayed/$tag.index" \
    > "$OUT/$tag.out" 2> "$OUT/$tag.err" &
  local pid=$!
  ( while kill -0 $pid 2>/dev/null; do
      sz=$(stat -c %s "/mnt/delayed/$tag.index" 2>/dev/null || echo 0)
      if [ "$sz" -ge "$SRC_BYTES" ]; then
        set_table /tmp/tbl.striped
        echo "  0.5ms-effective write delay on at $(date -u +%FT%TZ)" >> "$OUT/runner.log"; break
      fi; sleep 1; done ) & local w=$!
  wait $pid; local rc=$?; wait $w 2>/dev/null; set_table /tmp/tbl.linear
  echo "$tag rc=$rc done $(date -u +%FT%TZ)" | tee -a "$OUT/runner.log"
  for pat in "TIMEPROF: final" "IODEV: final" "TESTRECALL"; do
    grep -m1 "$pat" "$OUT/$tag.err" | sed "s/^/  /" | tee -a "$OUT/runner.log"
  done
  rm -f "/mnt/delayed/$tag.index"
}

for SEED in 42 43; do
  run_arm "uh_s${SEED}_none" --seed $SEED
  for Q in 0.30 0.50 0.70; do
    run_arm "uh_s${SEED}_q${Q}" --seed $SEED --lambda-noise-floor-quantile $Q
  done
done
echo "UNIHALF_DONE $(date -u +%FT%TZ)" | tee -a "$OUT/runner.log"
