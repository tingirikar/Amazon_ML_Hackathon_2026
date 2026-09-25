import os
from collections import defaultdict, Counter
from preprocess import (clean_ascii, clean_unicode, extract_core_name,
                        extract_tokens, extract_addr_tokens, extract_digits,
                        has_non_ascii)


class CandidateBlocker:
    """Multi-channel country-partitioned candidate blocker.

    Four parallel retrieval channels ensure both ASCII-name and cross-script
    matches are found:
      Channel 1: Exact core name match           (+10 score)
      Channel 2: Name token inverted index        (+3 per token)
      Channel 3: Address digit inverted index     (+6 per digit)
      Channel 4: Address token inverted index     (+3 per token) ← NEW

    Channel 4 is the key fix for Indic-script matching: when the business
    name is in Tamil/Hindi and produces empty ASCII, the Latin-script
    address tokens still provide strong matching signal.
    """

    def __init__(self, max_token_freq=2500, max_addr_token_freq=5000,
                 top_k_candidates=30, min_candidate_score=4):
        self.max_token_freq = max_token_freq
        self.max_addr_token_freq = max_addr_token_freq
        self.top_k_candidates = top_k_candidates
        self.min_candidate_score = min_candidate_score
        self.records = []
        self.exact_name_index = defaultdict(list)
        self.token_index = defaultdict(list)
        self.digit_index = defaultdict(list)
        self.addr_token_index = defaultdict(list)

    def add_records(self, record_list):
        """Add S2/S3 records to all four indexes.

        record_list: list of (entity_id, business_name, business_address, country)
        """
        offset = len(self.records)
        for i, (eid, bname, baddr, country) in enumerate(record_list):
            idx = offset + i

            # ASCII-cleaned versions (for inverted indexes + features)
            c_name = clean_ascii(bname)
            core_name = extract_core_name(bname)
            c_addr = clean_ascii(baddr)
            digits = extract_digits(baddr)

            # Unicode-preserved versions (for cross-script features)
            u_name = clean_unicode(bname)
            u_addr = clean_unicode(baddr)
            non_ascii = has_non_ascii(bname)

            # Store compact record: 8-tuple
            #   0: eid, 1: c_name, 2: core_name, 3: c_addr,
            #   4: digits, 5: u_name, 6: u_addr, 7: non_ascii
            self.records.append(
                (eid, c_name, core_name, c_addr, digits, u_name, u_addr, non_ascii)
            )

            # --- Channel 1: Exact core name index ---
            if core_name and len(core_name) >= 3:
                self.exact_name_index[core_name].append(idx)

            # --- Channel 2: Significant name token index ---
            for t in extract_tokens(bname, is_name=True):
                self.token_index[t].append(idx)

            # --- Channel 3: Address digit index ---
            for d in digits:
                self.digit_index[d].append(idx)

            # --- Channel 4: Address token index (cross-script lifeline) ---
            for t in extract_addr_tokens(baddr):
                self.addr_token_index[t].append(idx)

    def prune_frequent_tokens(self):
        """Remove overly frequent tokens to keep retrieval fast."""
        self.token_index = {
            t: idxs for t, idxs in self.token_index.items()
            if len(idxs) <= self.max_token_freq
        }
        self.digit_index = {
            t: idxs for t, idxs in self.digit_index.items()
            if len(idxs) <= self.max_token_freq
        }
        self.addr_token_index = {
            t: idxs for t, idxs in self.addr_token_index.items()
            if len(idxs) <= self.max_addr_token_freq
        }

    def get_candidates(self, s1_name, s1_addr):
        """Retrieve top candidate indices for a Source 1 entity using all 4 channels."""
        s1_core = extract_core_name(s1_name)
        s1_tokens = extract_tokens(s1_name, is_name=True)
        s1_digits = extract_digits(s1_addr)
        s1_addr_tokens = extract_addr_tokens(s1_addr)

        scores = Counter()

        # Channel 1: Exact core name (+10)
        if s1_core and s1_core in self.exact_name_index:
            for idx in self.exact_name_index[s1_core]:
                scores[idx] += 10

        # Channel 2: Name token overlap (+3 per token)
        for t in s1_tokens:
            for idx in self.token_index.get(t, []):
                scores[idx] += 3

        # Channel 3: Address digit overlap (+6 per digit)
        for d in s1_digits:
            for idx in self.digit_index.get(d, []):
                scores[idx] += 6

        # Channel 4: Address token overlap (+3 per token) — cross-script lifeline
        for t in s1_addr_tokens:
            for idx in self.addr_token_index.get(t, []):
                scores[idx] += 3

        if not scores:
            return []

        # Filter by minimum score and take top-K
        top = scores.most_common(self.top_k_candidates)
        top_indices = [idx for idx, sc in top if sc >= self.min_candidate_score]

        return [(self.records[idx][0], self.records[idx]) for idx in top_indices]
