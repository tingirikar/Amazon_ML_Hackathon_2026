# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** HumanIntelligence  
**Team Members:**
Tingirikar Lokesh
Vivekananda Reddy Challa
Jamboju Sri Varun
Kalakoti Sai Charan

**Submission Date:** September 2026

---

## 1. Executive Summary
We present a high-precision, country-partitioned entity resolution pipeline combining a 4-channel multi-modal candidate blocker and a 22-feature LightGBM gradient boosted decision tree classifier with an F_0.5-optimised decision threshold (0.89). The system scales efficiently across all 1.73M test entities spanning the United States, France, and India, with explicit handling for cross-script Indic transliterations, URL stripping, legal suffix normalization, and address token matching. On held-out validation data, the model achieved a pair-level F_0.5 score of **0.99934**, successfully passing all official competition validation constraints.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory analysis of business entities across Source 1, 2, and 3 revealed several distinct noise patterns:
1. **Multilingual and Cross-Script Noise:** In India, business names frequently appear in native Indic scripts (Tamil, Devanagari) in one source and Latin transliteration in another, while shared street addresses are recorded in Latin characters.
2. **Legal and Corporate Suffix Variations:** Frequent discrepancies across entity registrations (`Inc`, `Incorporated`, `LLC`, `Pvt Ltd`, `Private Limited`, `SARL`).
3. **Address Inconsistencies:** Landmark-based references ("Near SBI ATM", "Opposite City Post Office"), missing postal codes, and word order transpositions.
4. **URL and Domain Contamination:** Business names often contain website artifacts (`.com`, `.in`, `.org`, `www.`).
5. **Class Imbalance & Evaluation Metric:** F_0.5 penalizes false merges (precision errors) twice as heavily as missed matches (recall errors), requiring extreme confidence before linking candidate pairs and protecting singleton integrity.

### 2.2 Solution Strategy
We developed a two-stage **Blocking + GBDT Classifier** architecture:
- **Dual Representation Preprocessing:** Records are maintained in both canonical ASCII (NFKD normalized, punctuation stripped, legal tokens removed) and original Unicode form to prevent losing Indic script tokens.
- **Country Partitioning:** Partitioning by country (`US`, `India`, `France`) prevents cross-country candidate explosion and ensures linear scaling.
- **Precision-Optimized Thresholding:** Explicit threshold optimization directly maximizing F_0.5 on validation data.

**Approach Type:** Hybrid Multi-Channel Inverted Index Blocking + 22-Feature LightGBM GBDT  
**Core Innovation:** A 4-channel candidate blocker featuring an address-token cross-script channel that reliably connects native-script Indic names to Latin-transliterated counterparts via shared street and landmark tokens.

---

## 3. Candidate Generation (Blocking)
To reduce the $1.73 \times 10^6 \times 10^7$ comparison space without sacrificing recall, we deployed a 4-channel inverted index blocker per country partition:

- **Channel 1 (Exact Core Name):** Strips corporate legal suffixes and indexes normalized core names (+10 score).
- **Channel 2 (Distinctive Name Tokens):** Inverted index over non-stopword business name words (+3 score per match).
- **Channel 3 (Address Digits & PIN Codes):** Inverted index over numerical tokens such as street numbers, plot codes, and PIN codes (+6 score per match).
- **Channel 4 (Address Token Overlap):** Inverted index over address words, serving as a critical lifeline for cross-script matches where names cannot overlap lexically (+3 score per match).
- **Pruning & Filtering:** Ultra-frequent generic tokens (frequency > 2,500 for names, > 5,000 for addresses) are dynamically pruned. Candidates are ranked by cumulative score, filtered with a minimum score threshold of 4, and capped at top-30 candidates per entity.
- **Candidate Pool Generated:** ~40 million candidate pairs across 1,732,544 test entities (~23 candidates/entity average).

---

## 4. Matching Model

**Features Used (22 Total Pairwise Features):**
- **ASCII Name Similarity (Features 0–4):** RapidFuzz token set ratio, token sort ratio, full Levenshtein ratio, core name token set ratio, and core name Levenshtein ratio.
- **Address Similarity (Features 5–6):** Address token set ratio, address token sort ratio.
- **Numerical & Spatial Signals (Features 7–9):** Shared digit count, digit Jaccard coefficient, and binary exact digit match indicator.
- **Token Overlap (Features 10–11, 19):** First token match indicator, name word-level Jaccard, address word-level Jaccard.
- **Length Disparity (Features 12–13):** Normalized absolute length differences in name and address.
- **Unicode & Cross-Script Features (Features 14–15, 18):** Original Unicode token set ratio, Unicode Levenshtein ratio, and script mismatch indicator.
- **N-Gram Overlap (Features 16–17):** Character trigram Jaccard similarity on names and addresses.
- **Partial Matching (Features 20–21):** Best partial substring match ratios on names and addresses.

**Model Type:** LightGBM Gradient Boosted Decision Tree (500 trees, learning rate 0.05, max depth 8, num_leaves 63).  
**Training Set:** 200,000 reference entities with 1,893,069 pairs (693,069 positive matches and 1,200,000 balanced hard negatives).  
**Threshold Selection:** Held-out validation split (283,961 pairs). A sweep over thresholds [0.10, 0.95] pinpointed **0.89** as the optimal point, achieving an F_0.5 score of **0.99934**.

---

## 5. Results & Error Analysis

- **Validation F_0.5 Score:** **0.99934** (pair-level on held-out split)
- **Test Set Predictions:**
  - Total Source 1 Entities: 1,732,544
  - Entities with Matches: 1,720,088
  - Confirmed Singletons (No Match): 12,456
- **Common False Positives Mitigated:** By elevating the decision threshold to 0.89, chain stores or co-located businesses sharing identical address numbers without strong name corroboration are cleanly separated.
- **Common False Negatives Mitigated:** Cross-script Indic names that failed lexical name matching are rescued by address digit and token channels.

---

## 6. Conclusion
The HumanIntelligence entity resolution pipeline delivers an end-to-end, highly scalable, and mathematically rigorous solution for multi-source business record linkage. By uniting a 4-channel candidate blocker with a 22-feature LightGBM model tuned explicitly for F_0.5 precision, the system eliminates false merges while ensuring complete test coverage across diverse international geographies.

---

## Appendix

### A. Code Artefacts
All reproducible code is contained under `code/business_entity_resolution/`:
- `src/preprocess.py`: Unicode normalization, legal suffix stripping, token extraction.
- `src/blocking.py`: 4-channel inverted index blocker.
- `src/matcher.py`: 22-feature vector extraction and LightGBM inference.
- `src/train.py`: Large-scale training and F_0.5 threshold optimization pipeline.
- `src/main.py`: Complete inference orchestrator generating `matching_results.tsv` and `candidate_pairs.tsv`.
- `requirements.txt`: Python package requirements.
- `README.md`: Step-by-step reproduction instructions.

Reproducing results end-to-end:
```bash
python code/business_entity_resolution/src/main.py
```
