# Business Entity Resolution Pipeline — Team HumanIntelligence

This package contains the complete, reproducible end-to-end entity resolution solution for the Amazon ML Challenge 2026.

## Architecture Overview
1. **Hierarchical Preprocessing & Normalization (`src/preprocess.py`)**:
   - Unicode NFKD canonical normalization to ASCII.
   - Domain extension stripping (`.com`, `.in`, `.org`, `.net`, etc.).
   - Multi-country legal suffix normalization (`inc`, `pvt`, `ltd`, `sarl`, `llc`, `corp`, etc.).
   - Distinctive numerical token extraction (house/plot numbers, PIN/zip codes).
2. **Country-Partitioned Inverted Index Blocking (`src/blocking.py`)**:
   - Partitions processing by country (`US`, `India`, `France`).
   - Dual inverted indexes on core business names, distinctive name tokens, and address numerical tokens.
   - Prunes high-frequency tokens to keep retrieval ultra-fast with high recall.
3. **Pairwise Feature Engineering & GBDT Matcher (`src/matcher.py`, `src/train.py`)**:
   - RapidFuzz C++ token set, token sort, and partial string similarities.
   - Address overlap and numerical Jaccard coefficients.
   - LightGBM Gradient Boosted Decision Tree trained on ground truth matching and hard negative pairs.
   - F_0.5-optimized decision thresholding to penalize false merges and maximize singleton accuracy.

## How to Reproduce Results

### 1. Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Train the Matcher (Optional - Pretrained model included)
```bash
python src/train.py
```
This trains the LightGBM classifier on the training set ground truth and saves the model to `models/lgbm_matcher.txt`.

### 3. Generate Submission Outputs
```bash
python src/main.py
```
This executes candidate blocking and model inference over all 1,732,544 test entities and generates:
- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`

It automatically runs the official validator script (`validate_submission.py`) upon completion.
