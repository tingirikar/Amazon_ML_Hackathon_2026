import os
import numpy as np
import lightgbm as lgb
from rapidfuzz import fuzz
from preprocess import char_ngrams

FEATURE_NAMES = [
    # --- ASCII-cleaned name features (0-4) ---
    'name_token_set_ratio',          # 0: Token set similarity (handles word reordering)
    'name_token_sort_ratio',         # 1: Token sort similarity
    'name_ratio',                    # 2: Levenshtein ratio on cleaned names
    'core_name_token_set_ratio',     # 3: Token set on core names (legal suffixes removed)
    'core_name_ratio',               # 4: Levenshtein ratio on core names
    # --- ASCII-cleaned address features (5-6) ---
    'addr_token_set_ratio',          # 5: Address token set similarity
    'addr_token_sort_ratio',         # 6: Address token sort similarity
    # --- Digit features (7-9) ---
    'digit_intersection',            # 7: Count of shared numerical tokens
    'digit_jaccard',                 # 8: Jaccard coefficient on digit sets
    'addr_digit_full_match',         # 9: 1.0 if all digits match exactly
    # --- Token-level features (10-11) ---
    'first_token_match',             # 10: First root word matches
    'name_word_jaccard',             # 11: Word-level Jaccard on cleaned names
    # --- Length features (12-13) ---
    'name_len_diff',                 # 12: Absolute length disparity
    'addr_len_diff',                 # 13: Absolute length disparity
    # --- Unicode name features (14-15) — CRITICAL for cross-script ---
    'unicode_name_token_set_ratio',  # 14: Token set on ORIGINAL Unicode names
    'unicode_name_ratio',            # 15: Levenshtein ratio on original names
    # --- Character n-gram features (16-17) ---
    'char_ngram_jaccard_name',       # 16: Character trigram Jaccard on names
    'char_ngram_jaccard_addr',       # 17: Character trigram Jaccard on addresses
    # --- Cross-script features (18-19) ---
    'script_mismatch',               # 18: One name is ASCII, other has non-ASCII
    'addr_word_jaccard',             # 19: Word-level Jaccard on addresses
    # --- Partial match features (20-21) ---
    'name_partial_ratio',            # 20: Best partial substring match on names
    'addr_partial_ratio',            # 21: Best partial substring match on addresses
]

NUM_FEATURES = len(FEATURE_NAMES)  # 22


def _word_jaccard(s1, s2):
    """Word-level Jaccard similarity."""
    if not s1 or not s2:
        return 0.0
    w1 = set(s1.split())
    w2 = set(s2.split())
    if not w1 or not w2:
        return 0.0
    inter = len(w1 & w2)
    union = len(w1 | w2)
    return inter / union if union > 0 else 0.0


def _ngram_jaccard(s1, s2, n=3):
    """Character n-gram Jaccard similarity."""
    ng1 = char_ngrams(s1, n)
    ng2 = char_ngrams(s2, n)
    if not ng1 or not ng2:
        return 0.0
    inter = len(ng1 & ng2)
    union = len(ng1 | ng2)
    return inter / union if union > 0 else 0.0


def extract_pairwise_features(
    s1_name, s1_core, s1_addr, s1_digits,
    c_name, c_core, c_addr, c_digits,
    s1_uname='', c_uname='',
    s1_uaddr='', c_uaddr='',
    s1_non_ascii=False, c_non_ascii=False
):
    """Compute the full 22-feature vector for a candidate pair."""
    # --- Digit features ---
    d_inter = len(s1_digits & c_digits)
    d_union = len(s1_digits | c_digits)
    d_jaccard = (d_inter / d_union) if d_union > 0 else 0.5
    d_full_match = 1.0 if (d_union > 0 and d_inter == d_union) else 0.0

    # --- First token match ---
    w1 = s1_core.split()[0] if s1_core else ''
    w2 = c_core.split()[0] if c_core else ''
    first_match = 1.0 if (w1 and w2 and w1 == w2) else 0.0

    # --- Script mismatch ---
    script_mm = 1.0 if (s1_non_ascii != c_non_ascii) else 0.0

    # --- Use unicode names for unicode features; fallback to ASCII ---
    un1 = s1_uname or s1_name
    un2 = c_uname or c_name
    ua1 = s1_uaddr or s1_addr
    ua2 = c_uaddr or c_addr

    return [
        # 0-4: ASCII name features
        float(fuzz.token_set_ratio(s1_name, c_name)),
        float(fuzz.token_sort_ratio(s1_name, c_name)),
        float(fuzz.ratio(s1_name, c_name)),
        float(fuzz.token_set_ratio(s1_core, c_core)),
        float(fuzz.ratio(s1_core, c_core)),
        # 5-6: ASCII address features
        float(fuzz.token_set_ratio(s1_addr, c_addr)) if (s1_addr and c_addr) else 0.0,
        float(fuzz.token_sort_ratio(s1_addr, c_addr)) if (s1_addr and c_addr) else 0.0,
        # 7-9: Digit features
        float(d_inter),
        float(d_jaccard),
        d_full_match,
        # 10-11: Token features
        first_match,
        _word_jaccard(s1_name, c_name),
        # 12-13: Length features
        float(abs(len(s1_name) - len(c_name))),
        float(abs(len(s1_addr) - len(c_addr))),
        # 14-15: Unicode name features (CRITICAL for cross-script)
        float(fuzz.token_set_ratio(un1, un2)),
        float(fuzz.ratio(un1, un2)),
        # 16-17: Character n-gram Jaccard
        _ngram_jaccard(un1, un2),
        _ngram_jaccard(ua1, ua2),
        # 18-19: Cross-script + address Jaccard
        script_mm,
        _word_jaccard(s1_addr, c_addr),
        # 20-21: Partial ratio features
        float(fuzz.partial_ratio(s1_name, c_name)) if (s1_name and c_name) else 0.0,
        float(fuzz.partial_ratio(s1_addr, c_addr)) if (s1_addr and c_addr) else 0.0,
    ]


def is_plausible_match(feat_vector):
    """Precision gate: rejects candidates matched only on address with no name signal.

    Relaxed thresholds to avoid killing recall on noisy names like
    "ZEPHAY LESAI [[INC]]" vs "Zephay Labs Inc" which have moderate token overlap.
    """
    token_set = feat_vector[0]
    core_token_set = feat_vector[3]
    partial_ratio = feat_vector[20]
    word_jaccard = feat_vector[11]
    script_mismatch = feat_vector[18]
    unicode_token_set = feat_vector[14]
    addr_token_set = feat_vector[5]
    d_inter = feat_vector[7]

    # Case 1: Same script — need SOME name signal (relaxed)
    if script_mismatch == 0.0:
        if token_set >= 45.0 or core_token_set >= 40.0 or partial_ratio >= 55.0 or word_jaccard >= 0.20:
            return True
        return False

    # Case 2: Cross-script (Tamil/Hindi vs Latin)
    if script_mismatch == 1.0:
        # Strong address overlap with any digit/ngram match
        if addr_token_set >= 40.0 and (d_inter >= 1.0 or feat_vector[17] >= 0.20):
            return True
        # Unicode names still match across scripts
        if unicode_token_set >= 40.0:
            return True
        return False

    return False


def select_matches(candidate_scores, threshold=0.75):
    """Select all candidates above threshold.

    The LightGBM model trained on blocker hard negatives IS the precision gate.
    No hand-coded plausibility rules — the model learned to distinguish true
    matches from co-located businesses during training.

    Soft cap at top-8 per source to prevent extreme edge cases.
    """
    by_source = {'S2': [], 'S3': []}
    for cid, prob, fv in candidate_scores:
        if prob < threshold:
            continue
        src = cid[:2]
        if src in by_source:
            by_source[src].append((cid, prob))

    selected = []
    for src in ('S2', 'S3'):
        cands = by_source[src]
        if not cands:
            continue
        # Sort by probability descending, take top-8
        cands.sort(key=lambda x: x[1], reverse=True)
        for cid, prob in cands[:8]:
            selected.append(cid)

    return selected


class EntityMatcher:
    """LightGBM-based pairwise entity matcher with F0.5-optimized thresholding."""

    def __init__(self, model_path="models/lgbm_matcher.txt", threshold=0.75):
        self.model_path = model_path
        self.threshold = threshold
        self.model = None

    def train_and_save(self, X, y):
        """Train LightGBM model on extracted pair features and save to disk."""
        clf = lgb.LGBMClassifier(
            n_estimators=1200,
            learning_rate=0.04,
            num_leaves=127,
            max_depth=10,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        clf.fit(np.array(X), np.array(y))
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        clf.booster_.save_model(self.model_path)
        self.model = clf.booster_
        print(f"Model saved to {self.model_path}")

    def load_model(self):
        """Load pretrained LightGBM Booster for fastest C++ prediction."""
        if os.path.exists(self.model_path):
            self.model = lgb.Booster(model_file=self.model_path)
            return True
        return False

    def predict_matches(self, feature_rows):
        """Run batch inference. Returns array of match probabilities."""
        if feature_rows is None or len(feature_rows) == 0:
            return np.array([])
        X = feature_rows if isinstance(feature_rows, np.ndarray) else np.array(feature_rows)
        if self.model is not None:
            return self.model.predict(X)
        else:
            probs = (X[:, 0] * 0.35 + X[:, 3] * 0.35 +
                     X[:, 5] * 0.15 + X[:, 8] * 0.15) / 100.0
            return probs
