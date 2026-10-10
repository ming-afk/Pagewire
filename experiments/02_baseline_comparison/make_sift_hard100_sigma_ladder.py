#!/usr/bin/env python3
"""Re-emit a hard100 eval cache with a chosen ladder of query-concentration sigmas.

The refs (the 100 hardest individual queries) and their exact top-100 ground
truth are copied unchanged from the source cache; only `sigma_levels` is
replaced.  `sigma` is the expected L2 norm of the Gaussian perturbation added to
a reference vector when the harness draws a query, so it is the concentration
parameter of the query distribution: sigma=0 replays the reference exactly,
larger sigma spreads the queries out around it.

Train and test streams are drawn with different seeds, so for any sigma>0 they
are disjoint samples from the same distribution -- no train/test query overlap.

Usage:
  make_sift_hard100_sigma_ladder.py SRC.npz OUT.npz SIGMA [SIGMA ...]
"""
import sys

import numpy as np


def main():
    source_path, out_path = sys.argv[1:3]
    sigma_levels = np.array([float(value) for value in sys.argv[3:]], dtype=np.float32)
    source = np.load(source_path)
    refs = np.asarray(source["refs"], dtype=np.float32)
    proxy_gt = np.asarray(source["proxy_gt"])
    num_refs, dim = refs.shape

    spread = float(np.linalg.norm(refs - refs.mean(0)[None, :], axis=1).mean())
    print(f"refs {refs.shape}  spread {spread:.3f}")
    print(f"source sigma_levels {np.asarray(source['sigma_levels'])}")
    print(f"new    sigma_levels {sigma_levels}  (ratio of spread: "
          f"{[round(float(s) / spread, 4) for s in sigma_levels]})")

    generator = np.random.default_rng(20260923)
    eval_vecs = np.stack([
        refs + generator.normal(0.0, sigma / np.sqrt(dim), refs.shape).astype(np.float32)
        for sigma in sigma_levels
    ]).astype(np.float32)
    eval_gt = np.repeat(proxy_gt[None], len(sigma_levels), axis=0)

    np.savez(
        out_path,
        center_id=np.int64(source["center_id"]),
        refs=refs,
        ref_qids=np.asarray(source["ref_qids"]),
        proxy_gt=proxy_gt,
        sigma_levels=sigma_levels,
        eval_vecs=eval_vecs,
        eval_gt=eval_gt,
        d_to_base0=np.float32(source["d_to_base0"]),
    )
    print(f"wrote {out_path}")
    print("SIGMA_LADDER_DONE")


if __name__ == "__main__":
    main()
