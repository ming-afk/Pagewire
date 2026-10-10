#!/usr/bin/env python3
"""
Build an A->B->A drift stream sampled from the FULL cluster-9 ("A")
and cluster-3 ("B") query populations in esci_query_clusters.csv --
exactly exp_tr.tex's "Adaptation Under Workload Drift" description
("Cluster 9 (lawn-mower/tire) forms A, and cluster 3 (tools/hardware)
forms B"), with no further narrowing.

Supersedes make_esci_tightcore_stream.py (removed, see ../../VIOLATIONS.md
item 10): that script drew only from tight_core_{a,b}_{train,heldout}.npy,
a pre-filtered subset (the N queries per cluster closest to the cluster's
own centroid, built by the main repo's select_tight_core.py) that
exp_tr.tex's text never describes. This script samples the train/held-out
pools directly from the raw cluster membership in esci_query_clusters.csv,
with no centroid-distance filtering of any kind.

Sampling: phases draw WITH REPLACEMENT (fixed seed) from their train-pool
index set (a random, fixed-seed split of the full cluster, disjoint from
the held-out pool). Held-out side-eval sets (500 each) are used directly,
with no further sampling. Both training and held-out GT come from
esci_gt_k100 (verified exact brute-force top-100 L2) -- never the zeroed
proxy_gt placeholder in dummy_cache.npz, which --ext-query-file/--true-gt/
--side-eval-*-gt bypass entirely.

Run on the box: python3 make_esci_cluster_stream.py --base-dir ~/esci_full
"""
import argparse
import csv
import struct

import numpy as np

TRAIN_PER_PHASE = 1000
EVAL_PER_SET = 500
CAT_A = 9
CAT_B = 3
K = 100
SEED = 42


def read_query_bin(path):
    with open(path, "rb") as f:
        n, dim = struct.unpack("<ii", f.read(8))
        data = np.fromfile(f, dtype=np.float32, count=n * dim).reshape(n, dim)
    return data


def write_query_bin(path, arr):
    n, dim = arr.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<ii", n, dim))
        arr.astype(np.float32).tofile(f)
    print(f"  wrote {path}  {n}x{dim}")


def read_truthset(path):
    with open(path, "rb") as f:
        n, k = struct.unpack("<II", f.read(8))
        ids = np.fromfile(f, dtype=np.uint32, count=n * k).reshape(n, k)
        dists = np.fromfile(f, dtype=np.float32, count=n * k).reshape(n, k)
    return ids, dists


def write_truthset(path, ids, dists):
    n, k = ids.shape
    with open(path, "wb") as f:
        f.write(struct.pack("<II", n, k))
        ids.astype(np.uint32).tofile(f)
        dists.astype(np.float32).tofile(f)
    print(f"  wrote {path}  {n}x{k}")


def cluster_train_heldout_split(idx, n_heldout, label, rng):
    """Random, fixed-seed split of the FULL cluster membership `idx` into a
    held-out pool of size n_heldout and a train pool with everything else --
    no centroid-distance or other filtering of which cluster members qualify."""
    perm = rng.permutation(len(idx))
    heldout = idx[perm[:n_heldout]]
    train = idx[perm[n_heldout:]]
    print(f"  cluster {label}: {len(idx)} queries (full cluster, unfiltered) "
          f"-> train pool={len(train)} heldout={len(heldout)}")
    return train, heldout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default="~/esci_full")
    ap.add_argument("--clusters-csv", default="esci_query_clusters.csv")
    ap.add_argument("--out-prefix", default="esci_cluster_stream")
    ap.add_argument("--phase-queries", type=int, default=TRAIN_PER_PHASE)
    ap.add_argument("--phases", choices=("AB", "ABA"), default="ABA")
    ap.add_argument("--cat-a", type=int, default=CAT_A)
    ap.add_argument("--cat-b", type=int, default=CAT_B)
    ap.add_argument("--n-heldout", type=int, default=EVAL_PER_SET)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--monitor-queries", type=int, default=0,
                     help="If >0, also sample this many queries uniformly at "
                          "random from the WHOLE esci_queries.bin (not "
                          "cluster-conditioned), disjoint from the A/B "
                          "train+heldout pools, and write them as "
                          "{out-prefix}_side_eval_c[.bin/_gt.bin] -- a "
                          "non-cluster monitor set for --side-eval-c-query/-gt.")
    ap.add_argument("--monitor-seed", type=int, default=1000)
    args = ap.parse_args()
    import os
    base = os.path.expanduser(args.base_dir)
    rng = np.random.default_rng(args.seed)
    phase_n = args.phase_queries

    print("Loading queries/GT/cluster labels...")
    queries = read_query_bin(f"{base}/esci_queries.bin")
    gt_ids, gt_dists = read_truthset(f"{base}/esci_gt_k100")

    labels = []
    with open(f"{base}/{args.clusters_csv}") as f:
        for row in csv.DictReader(f):
            labels.append(int(row["cluster_id"]))
    labels = np.array(labels)
    assert len(labels) == queries.shape[0]

    idx_a = np.where(labels == args.cat_a)[0]
    idx_b = np.where(labels == args.cat_b)[0]

    train_a, heldout_a = cluster_train_heldout_split(idx_a, args.n_heldout, f"A={args.cat_a}", rng)
    train_b, heldout_b = cluster_train_heldout_split(idx_b, args.n_heldout, f"B={args.cat_b}", rng)

    a1 = rng.choice(train_a, size=phase_n, replace=True)
    b_train = rng.choice(train_b, size=phase_n, replace=True)
    a2 = rng.choice(train_a, size=phase_n, replace=True)
    a_eval = heldout_a
    b_eval = heldout_b

    stream_idx = np.concatenate([a1, b_train] + ([a2] if args.phases == "ABA" else []))
    print(f"  stream: A[0:{phase_n}) B[{phase_n}:{2*phase_n})"
          + (f" A2[{2*phase_n}:{3*phase_n})" if args.phases == "ABA" else "")
          + f"  total={len(stream_idx)}")

    write_query_bin(f"{base}/{args.out_prefix}_queries.bin", queries[stream_idx])
    write_truthset(f"{base}/{args.out_prefix}_gt.bin", gt_ids[stream_idx], gt_dists[stream_idx])

    write_query_bin(f"{base}/{args.out_prefix}_side_eval_a.bin", queries[a_eval])
    write_truthset(f"{base}/{args.out_prefix}_side_eval_a_gt.bin", gt_ids[a_eval], gt_dists[a_eval])
    write_query_bin(f"{base}/{args.out_prefix}_side_eval_b.bin", queries[b_eval])
    write_truthset(f"{base}/{args.out_prefix}_side_eval_b_gt.bin", gt_ids[b_eval], gt_dists[b_eval])

    if args.monitor_queries > 0:
        used = set(np.concatenate([idx_a, idx_b]).tolist())
        all_idx = np.arange(len(queries))
        candidates = all_idx[~np.isin(all_idx, list(used))]
        mon_rng = np.random.default_rng(args.monitor_seed)
        monitor_idx = mon_rng.choice(candidates, size=args.monitor_queries, replace=False)
        print(f"  monitor set (whole-dataset, non-cluster): n={len(monitor_idx)} "
              f"(sampled from {len(candidates)} candidates outside the A/B clusters)")
        write_query_bin(f"{base}/{args.out_prefix}_side_eval_c.bin", queries[monitor_idx])
        write_truthset(f"{base}/{args.out_prefix}_side_eval_c_gt.bin",
                        gt_ids[monitor_idx], gt_dists[monitor_idx])

    print("Writing dummy cache npz (unused array contents; --ext-query-file bypasses it)...")
    dim = queries.shape[1]
    np.savez_compressed(
        f"{base}/dummy_cache.npz",
        refs=np.zeros((2, dim), dtype=np.float32),
        proxy_gt=np.zeros((2, K), dtype=np.int64),
        sigma_levels=np.array([0.0], dtype=np.float32),
    )
    print(f"PHASE_BOUNDARIES: A_end={phase_n} B_end={2*phase_n}"
          + (f" A2_end={3*phase_n}" if args.phases == "ABA" else ""))
    print("DONE")


if __name__ == "__main__":
    main()
