#!/usr/bin/env python3
"""Brute-force exact top-100 L2 ground truth for a dumped DiskANN query stream
against the recovered SIFT-10M base vectors.

Usage:
  compute_exact_gt_sift10m.py BASE.bin QUERIES.bin OUT_GT.bin
"""
import struct
import sys

import numpy as np

TOP_K = 100
BASE_CHUNK = 250000


def load_float_bin(path):
    with open(path, "rb") as handle:
        n, dim = struct.unpack("<ii", handle.read(8))
    return np.memmap(path, dtype=np.float32, mode="r", offset=8, shape=(n, dim)), n, dim


def write_truthset_bin(path, ids, dists):
    # DiskANN truthset format: npts:uint32, K:uint32, then npts*K uint32 ids,
    # then npts*K float32 distances (control_diskann.cpp's load_truthset_ids).
    with open(path, "wb") as handle:
        handle.write(struct.pack("<II", ids.shape[0], ids.shape[1]))
        ids.astype(np.uint32).tofile(handle)
        dists.astype(np.float32).tofile(handle)


def exact_top_k(queries, base, num_base, top_k):
    best_distance = np.full((queries.shape[0], top_k), np.inf, dtype=np.float32)
    best_index = np.zeros((queries.shape[0], top_k), dtype=np.int64)
    for start in range(0, num_base, BASE_CHUNK):
        chunk = np.asarray(base[start:start + BASE_CHUNK], dtype=np.float32)
        distance = (chunk * chunk).sum(1)[None, :] - 2.0 * (queries @ chunk.T)
        keep = min(top_k, distance.shape[1])
        columns = np.argpartition(distance, keep - 1, axis=1)[:, :keep]
        candidate = np.take_along_axis(distance, columns, axis=1)
        merged_d = np.concatenate([best_distance, candidate], axis=1)
        merged_i = np.concatenate([best_index, columns + start], axis=1)
        order = np.argpartition(merged_d, top_k - 1, axis=1)[:, :top_k]
        best_distance = np.take_along_axis(merged_d, order, axis=1)
        best_index = np.take_along_axis(merged_i, order, axis=1)
        print(f"  base chunk {start}/{num_base}", flush=True)
    order = np.argsort(best_distance, axis=1)
    sorted_index = np.take_along_axis(best_index, order, axis=1)
    sorted_distance = np.take_along_axis(best_distance, order, axis=1)
    q_sq = (queries * queries).sum(1, keepdims=True)
    return sorted_index, sorted_distance + q_sq


def main():
    base_path, query_path, out_path = sys.argv[1:4]
    base, num_base, dim = load_float_bin(base_path)
    queries, num_queries, qdim = load_float_bin(query_path)
    assert qdim == dim, (qdim, dim)
    queries = np.asarray(queries, dtype=np.float32)
    print(f"base n={num_base} dim={dim}  queries n={num_queries}", flush=True)
    gt_ids, gt_dists = exact_top_k(queries, base, num_base, TOP_K)
    write_truthset_bin(out_path, gt_ids, gt_dists)
    print(f"wrote {out_path}  shape={gt_ids.shape}")
    print("EXACT_GT_DONE")


if __name__ == "__main__":
    main()
