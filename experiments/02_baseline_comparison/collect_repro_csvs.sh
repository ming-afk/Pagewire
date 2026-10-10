#!/bin/bash
# Collect every SIFT-10M reproduction run on the box into recall-QPS csvs.
# Run ON the box; writes into ~/repro_csv/.
set -u
OUT=/home/ubuntu/repro_csv
mkdir -p $OUT
EX=/home/ubuntu/extract_efsweep_csv.py

emit () {  # emit <run.err> <name>
  [ -f "$1" ] || { echo "skip (missing) $2"; return; }
  python3 $EX "$1" "$OUT/$2.csv" 8 2>/dev/null || echo "skip (no points) $2"
}

# --- R=64 base graph ---
emit /home/ubuntu/sift_norewire_sigma1/run.err                  r64_s35.8_diskann
emit /home/ubuntu/sift10m_hard100_sigma1_pagewire_w8/run.err    r64_s35.8_pagewire
emit /home/ubuntu/sift10m_hard100_sigma1_pagewire_w8_noprotect/run.err r64_s35.8_pagewire_noprot
emit /home/ubuntu/sigma_ladder/pristine_s1/run.err              r64_s8_diskann
emit /home/ubuntu/sigma_ladder/pagewire_s1/run.err              r64_s8_pagewire
emit /home/ubuntu/sigma_ladder/pristine_s2/run.err              r64_s16_diskann
emit /home/ubuntu/sigma_ladder/pagewire_s2/run.err              r64_s16_pagewire
emit /home/ubuntu/sigma_ladder/pristine_s3/run.err              r64_s24_diskann
emit /home/ubuntu/sigma_ladder/pagewire_s3/run.err              r64_s24_pagewire

# --- R=32 base graph ---
emit /home/ubuntu/r32_pristine/run.err                          r32_s35.8_diskann
emit /home/ubuntu/r32_pagewire/run.err                          r32_s35.8_pagewire
emit /home/ubuntu/r32_ladder/pristine_s8/run.err                r32_s8_diskann
emit /home/ubuntu/r32_ladder/pagewire_s8/run.err                r32_s8_pagewire
emit /home/ubuntu/r32_ladder/pristine_t4/run.err                r32_s4_diskann
emit /home/ubuntu/r32_ladder/pagewire_t4/run.err                r32_s4_pagewire
emit /home/ubuntu/r32_ladder/pristine_t2/run.err                r32_s2_diskann
emit /home/ubuntu/r32_ladder/pagewire_t2/run.err                r32_s2_pagewire

echo "--- collected ---"
ls -la $OUT/
