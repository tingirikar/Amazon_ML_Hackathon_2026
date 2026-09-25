"""Training pipeline: reads ground truth, generates features, trains LightGBM,
optimises the F0.5 threshold on a held-out validation split, and saves both the
model and the optimal threshold to disk.

Major improvements over v1
--------------------------
* Trains on 200K S1 entities (8× more data)
* Uses the expanded 22-feature set including Unicode/cross-script features
* Samples 6 hard negatives per S1 (3× more)
* Includes singleton S1 entities with negative-only candidate pools
* Optimises the threshold for pair-level F_0.5 on a 15% validation split
"""

import os
import sys
import time
import random
import numpy as np
import pandas as pd

# Ensure src modules are importable
sys.path.append(os.path.dirname(__file__))
from preprocess import clean_ascii, clean_unicode, extract_core_name, extract_digits, has_non_ascii
from matcher import extract_pairwise_features, EntityMatcher, NUM_FEATURES

BASE_DIR = "6ab10eb3b23ba_student_resource/student_resource"
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")


def _process_record(parts):
    """Extract all fields from a TSV line's split parts."""
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


def _make_features(s1, cand):
    """Build 22-feature vector from two processed records."""
    return extract_pairwise_features(
        s1[0], s1[1], s1[2], s1[3],          # ASCII S1
        cand[0], cand[1], cand[2], cand[3],  # ASCII candidate
        s1_uname=s1[4], c_uname=cand[4],     # Unicode names
        s1_uaddr=s1[5], c_uaddr=cand[5],     # Unicode addresses
        s1_non_ascii=s1[6], c_non_ascii=cand[6],  # Script flags
    )


def optimize_threshold(probs, labels, beta=0.5):
    """Find the threshold that maximises pair-level F_beta."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    best_f, best_t = 0.0, 0.50
    for t_int in range(25, 90):
        t = t_int / 100.0
        preds = (probs >= t).astype(int)
        tp = int(np.sum((preds == 1) & (labels == 1)))
        fp = int(np.sum((preds == 1) & (labels == 0)))
        fn = int(np.sum((preds == 0) & (labels == 1)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        if precision + recall > 0:
            f_beta = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
        else:
            f_beta = 0
        if f_beta > best_f:
            best_f = f_beta
            best_t = t
    return best_t, best_f


def train_pipeline(num_s1_samples=200000):
    random.seed(42)
    t0 = time.time()
    print("=" * 70)
    print(f"Training LightGBM Matcher on {num_s1_samples:,} Reference Entities...")
    print("  22-feature set | 500 trees | F0.5 threshold optimisation")
    print("=" * 70)

    # ------------------------------------------------------------------ 1. GT
    print("\n[1/6] Reading ground truth...")
    df_gt = pd.read_csv(
        os.path.join(TRAIN_DIR, "train_ground_truth.tsv"),
        sep="\t", nrows=num_s1_samples,
    )
    match_map = dict(
        zip(df_gt["source1_entity_id"],
            df_gt["matched_entity_ids"].fillna(""))
    )

    all_target_ids = set()
    for m in match_map.values():
        if m:
            all_target_ids.update(m.split(","))

    n_singletons = sum(1 for v in match_map.values() if not v)
    print(f"  {len(match_map):,} S1 entities | "
          f"{len(all_target_ids):,} unique true matches | "
          f"{n_singletons:,} singletons")

    # ------------------------------------------------------------------ 2. S1
    print("\n[2/6] Reading Source 1 records...")
    s1_data = {}
    with open(os.path.join(TRAIN_DIR, "train_source1.tsv"), "r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if parts[0] in match_map:
                s1_data[parts[0]] = _process_record(parts)
    print(f"  Loaded {len(s1_data):,} S1 records")

    # ------------------------------------------------------------------ 3. S2/S3
    print("\n[3/6] Reading Source 2 & 3 records...")
    cand_data = {}
    neg_pool = []       # random pool for negative sampling
    neg_pool_max = 300000

    for filename in ["train_source2.tsv", "train_source3.tsv"]:
        filepath = os.path.join(TRAIN_DIR, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            next(f)
            for i, line in enumerate(f):
                parts = line.strip().split("\t")
                if not parts or not parts[0]:
                    continue
                cid = parts[0]
                if cid in all_target_ids:
                    cand_data[cid] = _process_record(parts)
                elif len(neg_pool) < neg_pool_max and i % 15 == 0:
                    rec = _process_record(parts)
                    neg_pool.append((cid, rec))

    print(f"  {len(cand_data):,} true positive candidates | "
          f"{len(neg_pool):,} negative pool records")

    # Build per-country negative pools for efficient sampling
    country_negs = {}
    for cid, rec in neg_pool:
        c = rec[7]
        country_negs.setdefault(c, []).append((cid, rec))
    for c in country_negs:
        print(f"    Negative pool [{c}]: {len(country_negs[c]):,}")

    # ------------------------------------------------------------------ 4. Pairs
    print("\n[4/6] Generating feature vectors...")
    X, y = [], []
    pos_count = neg_count = 0

    for s1_id, match_str in match_map.items():
        s1 = s1_data.get(s1_id)
        if not s1:
            continue

        s1_country = s1[7]
        true_ids = set(match_str.split(",")) if match_str else set()

        # --- Positive pairs ---
        for tid in true_ids:
            cand = cand_data.get(tid)
            if cand and cand[7] == s1_country:
                X.append(_make_features(s1, cand))
                y.append(1)
                pos_count += 1

        # --- Negative pairs (6 per S1 entity) ---
        cn = country_negs.get(s1_country, [])
        if not cn:
            continue
        sampled = 0
        for _ in range(30):
            neg_cid, neg_rec = random.choice(cn)
            if neg_cid not in true_ids:
                X.append(_make_features(s1, neg_rec))
                y.append(0)
                neg_count += 1
                sampled += 1
                if sampled >= 6:
                    break

    print(f"  {len(X):,} total pairs | "
          f"Positives: {pos_count:,} | Negatives: {neg_count:,}")

    # ------------------------------------------------------------------ 5. Train
    print("\n[5/6] Training LightGBM model...")
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    # Stratified train/val split (85/15)
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    print(f"  Train: {len(X_train):,} | Val: {len(X_val):,}")

    matcher = EntityMatcher(model_path="models/lgbm_matcher.txt")
    matcher.train_and_save(X_train, y_train)

    # ------------------------------------------------------------------ 6. Threshold
    print("\n[6/6] Optimising F0.5 threshold on validation split...")
    probs_val = matcher.predict_matches(X_val.tolist())
    best_threshold, best_f05 = optimize_threshold(probs_val, y_val)

    print(f"  Best pair-level F0.5 = {best_f05:.5f} at threshold = {best_threshold:.2f}")

    # Save threshold
    os.makedirs("models", exist_ok=True)
    with open("models/threshold.txt", "w") as f:
        f.write(f"{best_threshold:.4f}")
    print(f"  Threshold saved to models/threshold.txt")

    # Feature importances
    importance = matcher.model.feature_importance(importance_type='gain')
    from matcher import FEATURE_NAMES
    sorted_imp = sorted(zip(FEATURE_NAMES, importance), key=lambda x: -x[1])
    print("\n  Top-10 feature importances (gain):")
    for fname, imp in sorted_imp[:10]:
        print(f"    {fname:35s} {imp:12.1f}")

    elapsed = time.time() - t0
    print(f"\n[DONE] Training finished in {elapsed:.1f}s")
    print(f"  Model: models/lgbm_matcher.txt")
    print(f"  Threshold: {best_threshold:.4f}")
    return best_threshold


if __name__ == "__main__":
    train_pipeline(num_s1_samples=200000)
