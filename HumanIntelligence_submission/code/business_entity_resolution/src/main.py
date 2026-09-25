"""End-to-end inference pipeline for the Amazon ML Challenge 2026.

Architecture
------------
1. Load or train LightGBM model + F0.5-optimised threshold.
2. For each country partition (US / India / France):
   a. Build a 4-channel candidate blocker on S2+S3 records.
   b. For every S1 entity, retrieve top-30 candidates.
   c. Extract 22-feature vectors and run batch LightGBM inference.
   d. Apply threshold → final matched IDs.
3. Write matching_results.tsv and candidate_pairs.tsv in exact S1 order.
4. Run the official competition validator.
"""

import os
import sys
import time
import gc
from collections import defaultdict
import numpy as np

# Ensure local imports
sys.path.append(os.path.dirname(__file__))
from preprocess import clean_ascii, clean_unicode, extract_core_name, extract_digits, has_non_ascii
from blocking import CandidateBlocker
from matcher import extract_pairwise_features, EntityMatcher, NUM_FEATURES

BASE_DIR = "6ab10eb3b23ba_student_resource/student_resource"
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
ROOT_OUTPUT_DIR = "HumanIntelligence_submission/output"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ROOT_OUTPUT_DIR, exist_ok=True)


def run_pipeline():
    start_time = time.time()
    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026: HIGH-PERFORMANCE ENTITY RESOLUTION PIPELINE v2")
    print("  4-channel blocker | 22 features | F0.5-optimised threshold")
    print("=" * 80)

    # ---------------------------------------------------------------- 1. Model
    model_path = "models/lgbm_matcher.txt"
    threshold_path = "models/threshold.txt"
    matcher = EntityMatcher(model_path=model_path, threshold=0.50)

    needs_retrain = False
    if matcher.load_model():
        # Verify feature count matches
        if matcher.model.num_feature() != NUM_FEATURES:
            print(f"[WARN] Model has {matcher.model.num_feature()} features, "
                  f"expected {NUM_FEATURES}. Retraining...")
            needs_retrain = True
        else:
            print(f"[OK] Loaded trained LightGBM model ({NUM_FEATURES} features)")
    else:
        needs_retrain = True

    if needs_retrain:
        print("[INFO] Training model on train dataset first...")
        from train import train_pipeline
        best_threshold = train_pipeline(num_s1_samples=200000)
        matcher.load_model()
        matcher.threshold = best_threshold

    # Load optimised threshold if available
    if os.path.exists(threshold_path):
        with open(threshold_path, "r") as f:
            saved_threshold = float(f.read().strip())
        matcher.threshold = saved_threshold
        print(f"[OK] Using F0.5-optimised threshold: {matcher.threshold:.4f}")
    else:
        print(f"[INFO] No saved threshold found, using default: {matcher.threshold:.2f}")

    # ---------------------------------------------------------------- 2. Source 1
    print(f"\n[1/4] Loading test_source1.tsv...")
    s1_order = []
    s1_by_country = defaultdict(list)
    s1_path = os.path.join(TEST_DIR, "test_source1.tsv")

    with open(s1_path, "r", encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if not parts or not parts[0]:
                continue
            eid = parts[0]
            name = parts[1] if len(parts) > 1 else ""
            addr = parts[2] if len(parts) > 2 else ""
            country = parts[3] if len(parts) > 3 else "US"

            s1_order.append(eid)
            s1_by_country[country].append((eid, name, addr))

    total_s1 = len(s1_order)
    print(f"  Loaded {total_s1:,} Source 1 entities across {len(s1_by_country)} countries:")
    for c, items in sorted(s1_by_country.items()):
        print(f"    {c}: {len(items):,} entities")

    # ---------------------------------------------------------------- 3. Process
    matched_results = {}
    candidate_results = {}

    print(f"\n[2/4] Processing candidate matching country-by-country...")
    countries = list(s1_by_country.keys())

    for country in countries:
        country_start = time.time()
        s1_items = s1_by_country[country]
        print(f"\n{'='*60}")
        print(f"  Country: {country} ({len(s1_items):,} entities)")
        print(f"{'='*60}")

        # Build multi-channel blocker for this country
        blocker = CandidateBlocker(
            max_token_freq=2500,
            max_addr_token_freq=5000,
            top_k_candidates=30,
            min_candidate_score=4,
        )

        # Stream S2 and S3 for this country only
        for fname in ["test_source2.tsv", "test_source3.tsv"]:
            fpath = os.path.join(TEST_DIR, fname)
            batch = []
            with open(fpath, "r", encoding="utf-8") as f:
                next(f)  # skip header
                for line in f:
                    parts = line.strip().split("\t")
                    if len(parts) >= 4 and parts[3] == country:
                        batch.append((parts[0], parts[1], parts[2], parts[3]))
                        if len(batch) >= 100000:
                            blocker.add_records(batch)
                            batch = []
                if batch:
                    blocker.add_records(batch)

        blocker.prune_frequent_tokens()
        idx_time = time.time() - country_start
        print(f"  Indexed {len(blocker.records):,} S2/S3 candidates in {idx_time:.1f}s")
        print(f"  Name tokens: {len(blocker.token_index):,} | "
              f"Digit tokens: {len(blocker.digit_index):,} | "
              f"Addr tokens: {len(blocker.addr_token_index):,}")

        # Process each S1 entity in this country
        matched_count = 0
        total_candidates = 0
        match_start = time.time()

        for i, (s1_id, s1_name, s1_addr) in enumerate(s1_items):
            # Precompute S1 features (once per entity)
            c_s1_name = clean_ascii(s1_name)
            c_s1_core = extract_core_name(s1_name)
            c_s1_addr = clean_ascii(s1_addr)
            s1_digits = extract_digits(s1_addr)
            u_s1_name = clean_unicode(s1_name)
            u_s1_addr = clean_unicode(s1_addr)
            s1_non_ascii = has_non_ascii(s1_name)

            # Get candidates from 4-channel blocker
            candidates = blocker.get_candidates(s1_name, s1_addr)

            if not candidates:
                matched_results[s1_id] = ""
                candidate_results[s1_id] = ""
                continue

            cand_ids = [c[0] for c in candidates]
            candidate_results[s1_id] = ",".join(cand_ids)
            total_candidates += len(candidates)

            # Feature extraction for all candidates
            feats_list = []
            for cand_id, cand_rec in candidates:
                # cand_rec: (eid, c_name, core_name, c_addr, digits,
                #            u_name, u_addr, non_ascii)
                f = extract_pairwise_features(
                    c_s1_name, c_s1_core, c_s1_addr, s1_digits,
                    cand_rec[1], cand_rec[2], cand_rec[3], cand_rec[4],
                    s1_uname=u_s1_name,    c_uname=cand_rec[5],
                    s1_uaddr=u_s1_addr,    c_uaddr=cand_rec[6],
                    s1_non_ascii=s1_non_ascii, c_non_ascii=cand_rec[7],
                )
                feats_list.append((cand_id, f))

            # Batch LightGBM prediction
            feat_matrix = [item[1] for item in feats_list]
            probs = matcher.predict_matches(feat_matrix)

            entity_matches = []
            for (cid, _), prob in zip(feats_list, probs):
                if prob >= matcher.threshold:
                    entity_matches.append(cid)

            if entity_matches:
                # Deduplicate while preserving order
                matched_results[s1_id] = ",".join(list(dict.fromkeys(entity_matches)))
                matched_count += 1
            else:
                matched_results[s1_id] = ""

            # Progress reporting
            if (i + 1) % 100000 == 0:
                elapsed = time.time() - match_start
                rate = (i + 1) / elapsed
                eta = (len(s1_items) - i - 1) / rate
                print(f"    Progress: {i+1:,}/{len(s1_items):,} "
                      f"({rate:.0f} entities/s, ETA {eta:.0f}s) | "
                      f"matches so far: {matched_count:,}")

        country_elapsed = time.time() - country_start
        avg_cands = total_candidates / max(len(s1_items), 1)
        print(f"  Country {country} finished in {country_elapsed:.1f}s | "
              f"Matches: {matched_count:,} | "
              f"Avg candidates/entity: {avg_cands:.1f}")

        # Free memory
        del blocker
        gc.collect()

    # ---------------------------------------------------------------- 4. Output
    print(f"\n[3/4] Writing output TSV files in exact test set order...")
    matching_out_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    candidate_out_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

    with open(matching_out_path, "w", encoding="utf-8") as f_match, \
         open(candidate_out_path, "w", encoding="utf-8") as f_cand:

        f_match.write("source1_entity_id\tmatched_entity_ids\n")
        f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

        for eid in s1_order:
            m = matched_results.get(eid, "")
            c = candidate_results.get(eid, "")
            f_match.write(f"{eid}\t{m}\n")
            f_cand.write(f"{eid}\t{c}\n")

    # Copy to submission output folder
    import shutil
    shutil.copy(matching_out_path, os.path.join(ROOT_OUTPUT_DIR, "matching_results.tsv"))
    shutil.copy(candidate_out_path, os.path.join(ROOT_OUTPUT_DIR, "candidate_pairs.tsv"))

    total_matched = sum(1 for v in matched_results.values() if v)
    total_singleton = sum(1 for v in matched_results.values() if not v)
    print(f"  Matched entities: {total_matched:,}")
    print(f"  Singletons (no match): {total_singleton:,}")
    print(f"  Saved: {matching_out_path}")
    print(f"  Saved: {candidate_out_path}")
    print(f"  Also copied to: {ROOT_OUTPUT_DIR}/")

    # ---------------------------------------------------------------- 5. Validate
    print(f"\n[4/4] Running Official Competition Submission Validator...")
    validator_script = os.path.join(BASE_DIR, "utils", "validate_submission.py")

    cmd = (f'"{sys.executable}" "{validator_script}" '
           f'--matching "{matching_out_path}" '
           f'--candidate "{candidate_out_path}" '
           f'--test-dir "{TEST_DIR}"')
    os.system(cmd)

    total_elapsed = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"PIPELINE COMPLETE | Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"{'='*80}")


if __name__ == "__main__":
    run_pipeline()
