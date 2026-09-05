#!/usr/bin/env python3
"""
build_advbench_split.py

Builds a cluster-aware probe/eval partition of a harmful-behaviour benchmark for
the LIBRA query-efficient configuration selection experiments.

Rationale
---------
AdvBench contains substantial near-duplication: many behaviours are paraphrases of
the same underlying request. A uniformly random split therefore leaks, because a
probe instruction and its eval twin differ only in wording. This script embeds all
behaviours, clusters them, and partitions at CLUSTER level so that no cluster spans
both halves.

Hard constraint: no cluster is split across probe and eval.

Note on provenance
------------------
The exp1-exp6 results were produced on the 50 hand-written instructions in
config/experiment_config.py (HARMFUL_INSTRUCTIONS), NOT on a sample of AdvBench.
Those two instruction sets are unrelated, so there is no canonical-subset
constraint to enforce here. The custom 50 are handled as a separate benchmark
with their own split.

Outputs a single frozen JSON config. Write once, commit, never regenerate silently.

Usage
-----
  python build_advbench_split.py \
      --dataset advbench \
      --out     config/advbench_split_v1.json \
      --target-probe 180 \
      --distance-threshold 0.35
"""

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def load_instructions(dataset_name: str, revision: str | None) -> pd.DataFrame:
    """
    Load via HarmfulDatasetLoader's registry so the split matches the experiment
    source. Import is done from the repo root, which must contain utils/.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from utils.dataset_loader import HarmfulDatasetLoader

    info = HarmfulDatasetLoader.DATASETS[dataset_name]
    from datasets import load_dataset
    ds = load_dataset(info["hf_name"], revision=revision)[info["split"]]

    if info["column"] not in ds.column_names:
        raise SystemExit(
            f"Column {info['column']!r} not in {ds.column_names} for "
            f"{info['hf_name']}. Fix the registry entry in "
            f"utils/dataset_loader.py before proceeding -- a mismatch here "
            f"means the experiments and the split disagree on the source data."
        )

    if dataset_name == "harmbench":
        key = next((c for c in ds.column_names if c.lower() in
                    ("functionalcategory", "functional_category")), None)
        if key is None:
            raise ValueError(f"No functional-category column in {ds.column_names}; "
                             f"cannot filter to the standard subset.")
        before = len(ds)
        ds = ds.filter(lambda r: str(r[key]).strip().lower() == "standard")
        print(f"[ok] harmbench standard subset: {len(ds)}/{before} behaviours")

    goals = [g.strip() for g in ds[info["column"]]]
    df = pd.DataFrame({"goal": goals})
    df["instruction_id"] = df.index          # position in SOURCE order, not sorted

    n_dupes = int(df["goal"].duplicated().sum())
    if n_dupes:
        print(f"[warn] {n_dupes} exact-duplicate instructions; they will cluster "
              f"together and stay on one side of the split.")

    print(f"[ok] loaded {len(df)} instructions from {info['hf_name']} "
          f"(column {info['column']!r})")
    return df


# --------------------------------------------------------------------------- #
# Embedding + clustering
# --------------------------------------------------------------------------- #

def embed(goals: list[str], model_name: str) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    print(f"[..] embedding {len(goals)} instructions with {model_name}")
    model = SentenceTransformer(model_name)
    emb = model.encode(goals, batch_size=64, show_progress_bar=True,
                       convert_to_numpy=True, normalize_embeddings=True)
    return emb


def cluster(emb: np.ndarray, threshold: float) -> np.ndarray:
    """
    Agglomerative clustering with average linkage on cosine distance.

    distance_threshold rather than n_clusters: we do not know the true number of
    semantic families a priori, and fixing k would force unrelated behaviours
    together. Lower threshold -> more, tighter clusters.
    """
    from sklearn.cluster import AgglomerativeClustering

    clus = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clus.fit_predict(emb)
    print(f"[ok] {labels.max() + 1} clusters at distance_threshold={threshold}")
    return labels


# --------------------------------------------------------------------------- #
# Partitioning
# --------------------------------------------------------------------------- #

def partition(df: pd.DataFrame, labels: np.ndarray,
              target_probe: int, seed: int) -> tuple[set[int], set[int]]:
    """
    Assign whole clusters to probe or eval.

    Clusters are shuffled, then added largest-first until the probe half is as
    close to target_probe as a whole-cluster assignment allows. Largest-first
    keeps the residual error small: big clusters are placed while there is still
    room, and singletons fine-tune the remainder.
    """
    rng = np.random.default_rng(seed)

    members = defaultdict(list)
    for idx, lab in enumerate(labels):
        members[int(lab)].append(idx)

    # Shuffle first so equal-sized clusters break ties randomly, then sort by size.
    order = (sorted(members.keys()))
    rng.shuffle(order)
    # order = sorted(order.tolist(), key=lambda c: len(members[c]), reverse=True)

    probe_clusters: set[int] = set()
    size = 0
    for c in order:
        if size >= target_probe:
            break
        # Accept the cluster only if doing so lands us closer to target than stopping.
        if abs(size + len(members[c]) - target_probe) < abs(size - target_probe):
            probe_clusters.add(c)
            size += len(members[c])

    eval_clusters = set(members) - probe_clusters
    probe_idx = {i for c in probe_clusters for i in members[c]}
    eval_idx = {i for c in eval_clusters for i in members[c]}

    print(f"[..] probe uses {len(probe_clusters)}/{len(members)} clusters")

    assert not (probe_idx & eval_idx), "probe/eval overlap"
    assert len(probe_idx) + len(eval_idx) == len(df), "instructions lost in partition"
    assert probe_idx and eval_idx, "one side of the split is empty"
    return probe_idx, eval_idx


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #

def leakage_report(emb: np.ndarray, probe_idx: set[int], eval_idx: set[int],
                   df: pd.DataFrame, top_n: int = 5) -> dict:
    """
    Cross-split similarity diagnostics. Embeddings are normalised, so the dot
    product IS cosine similarity.

    max is the worst case and goes in the paper, but on its own it invites the
    question of how typical it is. The neighbour fractions answer that: for each
    eval instruction, how close is its nearest probe instruction.
    """
    p = sorted(probe_idx)
    e = sorted(eval_idx)
    sims = emb[p] @ emb[e].T

    flat = np.dstack(np.unravel_index(np.argsort(sims, axis=None)[::-1], sims.shape))[0]
    pairs = []
    for pi, ei in flat[:top_n]:
        pairs.append({
            "similarity": round(float(sims[pi, ei]), 4),
            "probe": df.loc[p[pi], "goal"],
            "eval": df.loc[e[ei], "goal"],
        })

    nearest = sims.max(axis=0)          # per eval item, its closest probe item
    frac_80 = float((nearest > 0.80).mean())
    frac_85 = float((nearest > 0.85).mean())
    frac_90 = float((nearest > 0.90).mean())

    print(f"\n[leakage] max cross-split cosine similarity:  {sims.max():.4f}")
    print(f"[leakage] mean cross-split cosine similarity: {sims.mean():.4f}")
    print(f"[leakage] median nearest-probe similarity:    {np.median(nearest):.4f}")
    print(f"[leakage] eval items with a probe neighbour >0.80: {frac_80:.1%} "
          f"({int((nearest > 0.80).sum())}/{len(e)})")
    print(f"[leakage] eval items with a probe neighbour >0.85: {frac_85:.1%} "
          f"({int((nearest > 0.85).sum())}/{len(e)})")
    print(f"[leakage] eval items with a probe neighbour >0.90: {frac_90:.1%} "
          f"({int((nearest > 0.90).sum())}/{len(e)})")
    print("[leakage] closest cross-split pairs:")
    for pr in pairs:
        print(f"    {pr['similarity']:.4f}")
        print(f"      probe: {pr['probe'][:88]}")
        print(f"      eval : {pr['eval'][:88]}")

    return {
        "max_cross_split_cosine": round(float(sims.max()), 4),
        "mean_cross_split_cosine": round(float(sims.mean()), 4),
        "median_nearest_probe_cosine": round(float(np.median(nearest)), 4),
        "frac_eval_with_probe_neighbour_above_0.80": round(frac_80, 4),
        "frac_eval_with_probe_neighbour_above_0.85": round(frac_85, 4),
        "frac_eval_with_probe_neighbour_above_0.90": round(frac_90, 4),
        "closest_pairs": pairs,
    }


def cluster_report(labels: np.ndarray) -> dict:
    sizes = np.bincount(labels)
    print(f"\n[clusters] n={len(sizes)}  singletons={int((sizes == 1).sum())}  "
          f"max={int(sizes.max())}  median={float(np.median(sizes)):.1f}")
    return {
        "n_clusters": int(len(sizes)),
        "n_singletons": int((sizes == 1).sum()),
        "max_cluster_size": int(sizes.max()),
        "median_cluster_size": float(np.median(sizes)),
    }


# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="advbench",
                    choices=["advbench", "harmbench", "malicious_instruct"])
    ap.add_argument("--revision", default=None,
                    help="HuggingFace dataset revision/commit SHA. Ignored when "
                         "the Hub is unreachable and the local cache is used.")
    ap.add_argument("--cache-config-hash", default=None,
                    help="Local HF cache config hash, recorded in provenance when "
                         "loading offline (the path segment printed by datasets).")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--embed-model", default="sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--distance-threshold", type=float, default=0.35)
    ap.add_argument("--target-probe", type=int, default=180)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing split file")
    args = ap.parse_args()

    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} exists. This file is meant to be frozen. "
                         f"Pass --force only if you intend to invalidate prior runs.")

    df = load_instructions(args.dataset, args.revision)

    emb = embed(df["goal"].tolist(), args.embed_model)
    labels = cluster(emb, args.distance_threshold)

    probe_idx, eval_idx = partition(df, labels, args.target_probe, args.seed)
    print(f"\n[split] probe={len(probe_idx)}  eval={len(eval_idx)}  total={len(df)}")

    cstats = cluster_report(labels)
    lstats = leakage_report(emb, probe_idx, eval_idx, df)

    payload = {
        "schema_version": 2,
        "provenance": {
            "dataset": args.dataset,
            "hf_revision": args.revision,
            "hf_cache_config_hash": args.cache_config_hash,
            "instructions_sha256": hashlib.sha256(
                "\n".join(df["goal"]).encode("utf-8")).hexdigest(),
            "n_instructions": int(len(df)),
            "embed_model": args.embed_model,
            "distance_threshold": args.distance_threshold,
            "target_probe": args.target_probe,
            "seed": args.seed,
        },
        "cluster_stats": cstats,
        "leakage": lstats,
        "n_probe": len(probe_idx),
        "n_eval": len(eval_idx),
        "probe": [
            {"instruction_id": int(i), "goal": df.loc[i, "goal"],
             "cluster": int(labels[i])}
            for i in sorted(probe_idx)
        ],
        "eval": [
            {"instruction_id": int(i), "goal": df.loc[i, "goal"],
             "cluster": int(labels[i])}
            for i in sorted(eval_idx)
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[ok] wrote {args.out}")
    print(f"[ok] instructions_sha256 = {payload['provenance']['instructions_sha256'][:16]}...")
    print("[ok] commit this file before launching any grid runs.")


if __name__ == "__main__":
    sys.exit(main())