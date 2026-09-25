"""High-Speed Multiprocessing Training Pipeline: Blocker Hard Negatives + LightGBM + Vectorized Macro F0.5.

Key Innovations
---------------
1. Multi-Core Parallel Mining: Distributes S1 entity blocker queries and feature extraction
   across all available CPU cores using multiprocessing (with native zero-copy fork on Linux).
2. Blocker Hard Negative Mining: Mines realistic co-located/address imposters from the blocker,
   teaching LightGBM the exact boundary needed for 98%+ precision.
3. Vectorized Validation Scoring: Evaluates all validation candidates in a single batch C++ call.
"""

import os
import sys
import time
import random
import multiprocessing as mp
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.append(os.path.dirname(__file__))
from preprocess import (clean_ascii, clean_unicode, extract_core_name,
                        extract_digits, has_non_ascii)
from blocking import CandidateBlocker
from matcher import (extract_pairwise_features, select_matches,
                     EntityMatcher, NUM_FEATURES)

BASE_DIR = "6ab10eb3b23ba_student_resource/student_resource"
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")

# Global context dictionary for worker processes
_worker_ctx = {}


def _init_worker(blockers, s1_data, s1_raw, cand_data, match_map, train_id_set, neg_rate):
    """Initialize read-only memory structures inside worker process."""
    global _worker_ctx
    _worker_ctx = {
        'blockers': blockers,
        's1_data': s1_data,
        's1_raw': s1_raw,
        'cand_data': cand_data,
        'match_map': match_map,
        'train_id_set': train_id_set,
        'neg_rate': neg_rate,
    }


def _process_chunk(s1_keys_chunk):
    """Worker task: extract candidate pairs and features for a chunk of S1 entities."""
    blockers = _worker_ctx['blockers']
    s1_data = _worker_ctx['s1_data']
    s1_raw = _worker_ctx['s1_raw']
    cand_data = _worker_ctx['cand_data']
    match_map = _worker_ctx['match_map']
    train_id_set = _worker_ctx['train_id_set']
    neg_rate = _worker_ctx['neg_rate']

    local_X = []
    local_y = []
    local_pos = 0
    local_neg = 0
    local_val_candidates = {}

    for s1_id in s1_keys_chunk:
        s1 = s1_data[s1_id]
        s1_name, s1_addr = s1_raw[s1_id]
        country = s1[7]
        true_ids = match_map[s1_id]
        is_train = s1_id in train_id_set

        blocker = blockers.get(country)
        cands = blocker.get_candidates(s1_name, s1_addr) if blocker else []

        found_ids = set()
        cands_for_eval = []

        for cand_id, cand_rec in cands:
            found_ids.add(cand_id)
            c_proc = cand_data.get(cand_id)
            if not c_proc:
                continue

            feats = extract_pairwise_features(
                s1[0], s1[1], s1[2], s1[3],
                c_proc[0], c_proc[1], c_proc[2], c_proc[3],
                s1_uname=s1[4], c_uname=c_proc[4],
                s1_uaddr=s1[5], c_uaddr=c_proc[5],
                s1_non_ascii=s1[6], c_non_ascii=c_proc[6],
            )
            label = 1 if cand_id in true_ids else 0

            if is_train:
                if label == 1:
                    local_X.append(feats)
                    local_y.append(1)
                    local_pos += 1
                elif label == 0 and random.random() < neg_rate:
                    local_X.append(feats)
                    local_y.append(0)
                    local_neg += 1
            else:
                cands_for_eval.append((cand_id, feats))

        # Add true matches missed by blocker so positive distribution is fully learned
        for tid in true_ids:
            if tid not in found_ids and tid in cand_data:
                c_proc = cand_data[tid]
                feats = extract_pairwise_features(
                    s1[0], s1[1], s1[2], s1[3],
                    c_proc[0], c_proc[1], c_proc[2], c_proc[3],
                    s1_uname=s1[4], c_uname=c_proc[4],
                    s1_uaddr=s1[5], c_uaddr=c_proc[5],
                    s1_non_ascii=s1[6], c_non_ascii=c_proc[6],
                )
                if is_train:
                    local_X.append(feats)
                    local_y.append(1)
                    local_pos += 1
                else:
                    cands_for_eval.append((tid, feats))

        if not is_train:
            local_val_candidates[s1_id] = cands_for_eval

    return local_X, local_y, local_pos, local_neg, local_val_candidates


def _process_record(parts):
    """Extract preprocessed fields from TSV row."""
    name = parts[1] if len(parts) > 1 else ""
    addr = parts[2] if len(parts) > 2 else ""
    country = parts[3] if len(parts) > 3 else ""
    return (
        clean_ascii(name),          # 0: c_name
        extract_core_name(name),    # 1: core_name
        clean_ascii(addr),          # 2: c_addr
        extract_digits(addr),       # 3: digits
        clean_unicode(name),        # 4: u_name
        clean_unicode(addr),        # 5: u_addr
        has_non_ascii(name),        # 6: non_ascii
        country,                    # 7: country
    )


def compute_macro_f05(pred_map, true_map, beta=0.5):
    """Official competition Macro-Averaged F0.5 metric."""
    f_scores = []
    for s1_id, true_set in true_map.items():
        pred_set = pred_map.get(s1_id, set())

        if not true_set:
            f_scores.append(1.0 if not pred_set else 0.0)
            continue

        tp = len(pred_set & true_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if precision + recall > 0:
            score = (1 + beta**2) * precision * recall / (beta**2 * precision + recall)
        else:
            score = 0.0
        f_scores.append(score)

    return float(np.mean(f_scores))


def train_pipeline(num_s1_samples=200000):
    """High-speed multi-core training pipeline."""
    random.seed(42)
    np.random.seed(42)
    t0 = time.time()

    label = "ALL" if num_s1_samples <= 0 else f"{num_s1_samples:,}"
    print("=" * 75)
    print(f"Training LightGBM Matcher on {label} Reference Entities [MULTI-CORE ACCELERATED]...")
    print("  Mining HARD NEGATIVES from Blocker | Macro F0.5 Optimization")
    print("=" * 75)

    # ------------------------------------------------------------------ 1. GT
    print("\n[1/6] Reading ground truth...")
    gt_path = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")
    if num_s1_samples > 0:
        df_gt = pd.read_csv(gt_path, sep="\t", nrows=num_s1_samples)
    else:
        df_gt = pd.read_csv(gt_path, sep="\t")

    match_map = {}
    for _, row in df_gt.iterrows():
        s1_id = str(row["source1_entity_id"])
        val = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
        match_map[s1_id] = set(val.split(",")) if val else set()

    all_target_ids = set()
    for ids in match_map.values():
        all_target_ids.update(ids)

    n_singletons = sum(1 for ids in match_map.values() if not ids)
    print(f"  {len(match_map):,} S1 entities | {len(all_target_ids):,} unique true matches | {n_singletons:,} singletons")

    # ------------------------------------------------------------------ 2. S1
    print("\n[2/6] Reading Source 1 records...")
    s1_data = {}
    s1_raw = {}
    with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if parts[0] in match_map:
                s1_data[parts[0]] = _process_record(parts)
                s1_raw[parts[0]] = (parts[1] if len(parts) > 1 else "",
                                    parts[2] if len(parts) > 2 else "")

    print(f"  Loaded {len(s1_data):,} S1 records")

    # ------------------------------------------------------------------ 3. S2/S3
    print("\n[3/6] Reading Source 2 & 3 records and building Blocker...")
    cand_data = {}
    blocker_by_country = defaultdict(lambda: CandidateBlocker(
        max_token_freq=2500, max_addr_token_freq=5000,
        top_k_candidates=50, min_candidate_score=3
    ))

    for filename in ["train_source2.tsv", "train_source3.tsv"]:
        filepath = os.path.join(TRAIN_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            next(f)
            batch = defaultdict(list)
            for i, line in enumerate(f):
                parts = line.strip().split("\t")
                if not parts or not parts[0]:
                    continue
                cid = parts[0]
                country = parts[3] if len(parts) > 3 else "US"
                # Keep true matches + 1 in 4 sample for realistic hard negative distribution
                if cid in all_target_ids or (i % 4 == 0):
                    rec = _process_record(parts)
                    cand_data[cid] = rec
                    batch[country].append((cid, parts[1] if len(parts) > 1 else "",
                                           parts[2] if len(parts) > 2 else "", country))
                    if len(batch[country]) >= 50000:
                        blocker_by_country[country].add_records(batch[country])
                        batch[country] = []

            for c, records in batch.items():
                if records:
                    blocker_by_country[c].add_records(records)

    for c in blocker_by_country:
        blocker_by_country[c].prune_frequent_tokens()
        print(f"  Country [{c}] Blocker ready: {len(blocker_by_country[c].records):,} indexed records")

    # ------------------------------------------------------------------ 4. Mining (Parallel)
    print("\n[4/6] Mining Hard Negatives from Blocker & building feature pairs [PARALLEL]...")
    s1_keys = list(s1_data.keys())
    train_ids, val_ids = train_test_split(s1_keys, test_size=0.15, random_state=42)
    train_id_set = set(train_ids)

    n_workers = min(16, max(1, os.cpu_count() or 4))
    print(f"  Spawning {n_workers} parallel CPU workers across {len(s1_keys):,} entities...")

    chunk_size = max(500, len(s1_keys) // (n_workers * 4))
    chunks = [s1_keys[i : i + chunk_size] for i in range(0, len(s1_keys), chunk_size)]

    ctx = mp.get_context("fork") if hasattr(os, "fork") else mp.get_context("spawn")

    X, y = [], []
    pos_count = neg_count = 0
    val_candidates = {}

    mining_t0 = time.time()
    with ctx.Pool(
        processes=n_workers,
        initializer=_init_worker,
        initargs=(
            dict(blocker_by_country),
            s1_data,
            s1_raw,
            cand_data,
            match_map,
            train_id_set,
            0.50,
        ),
    ) as pool:
        processed_chunks = 0
        total_entities_processed = 0
        for l_x, l_y, l_pos, l_neg, l_val in pool.imap_unordered(_process_chunk, chunks):
            X.extend(l_x)
            y.extend(l_y)
            pos_count += l_pos
            neg_count += l_neg
            val_candidates.update(l_val)
            processed_chunks += 1
            total_entities_processed += len(chunks[processed_chunks - 1])
            if processed_chunks % 5 == 0 or processed_chunks == len(chunks):
                pct = (processed_chunks / len(chunks)) * 100.0
                rate = total_entities_processed / max(time.time() - mining_t0, 0.01)
                print(f"    Progress: {pct:.1f}% ({total_entities_processed:,}/{len(s1_keys):,} entities | {rate:.0f} ent/s)")

    print(f"  Generated {len(X):,} training pairs in {time.time() - mining_t0:.1f}s | "
          f"Positives: {pos_count:,} | Hard Negatives: {neg_count:,}")

    # ------------------------------------------------------------------ 5. Train
    print("\n[5/6] Training LightGBM model on blocker hard negatives...")
    matcher = EntityMatcher(model_path="models/lgbm_matcher.txt")
    matcher.train_and_save(X, y)

    # ------------------------------------------------------------------ 6. Optimize F0.5 (Vectorized)
    print("\n[6/6] Optimizing Macro F0.5 threshold on held-out validation split [VECTORIZED]...")
    val_true_map = {sid: match_map[sid] for sid in val_ids}

    # Flatten all validation candidate pairs for a single vectorized C++ LightGBM predict call
    all_val_feats = []
    val_meta = []
    for sid, cands in val_candidates.items():
        for cid, feats in cands:
            all_val_feats.append(feats)
            val_meta.append((sid, cid))

    val_scored = defaultdict(list)
    if all_val_feats:
        print(f"  Running batch inference on {len(all_val_feats):,} validation pairs...")
        all_probs = matcher.predict_matches(all_val_feats)
        for (sid, cid), prob in zip(val_meta, all_probs):
            val_scored[sid].append((cid, float(prob), None))

    # Grid search threshold to maximize official competition Macro F0.5
    best_f05, best_thresh = 0.0, 0.75
    threshold_candidates = [
        0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
        0.72, 0.74, 0.76, 0.78, 0.80,
        0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.95
    ]

    for t in threshold_candidates:
        pred_map = {}
        for sid, scored_list in val_scored.items():
            selected = select_matches(scored_list, threshold=t)
            pred_map[sid] = set(selected)

        score = compute_macro_f05(pred_map, val_true_map)
        print(f"  Threshold {t:.2f} -> Validation Macro F0.5 = {score:.5f}")
        if score > best_f05:
            best_f05 = score
            best_thresh = t

    print(f"\n[DONE] Optimal Threshold = {best_thresh:.2f} with Macro F0.5 = {best_f05:.5f}")
    os.makedirs("models", exist_ok=True)
    with open("models/threshold.txt", "w") as f:
        f.write(f"{best_thresh:.4f}\n")

    print(f"  Saved optimal threshold {best_thresh:.4f} to models/threshold.txt")
    print(f"  Total pipeline time: {time.time() - t0:.1f}s")
    return best_thresh


if __name__ == "__main__":
    train_pipeline()
