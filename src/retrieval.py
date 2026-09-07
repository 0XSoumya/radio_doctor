"""Retrieval engine for template-conditioned and dictation-similar cases."""

from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.data import RadiologyCase


class CaseRetriever:
    """Fast local retriever using TF-IDF and template-conditioned filtering."""

    def __init__(self, train_cases: List[RadiologyCase]):
        self.train_cases = train_cases
        self.dictations = [c.dictation or "" for c in train_cases]
        self.templates = [c.template_content or "" for c in train_cases]
        
        # Word + character n-gram TF-IDF vectorizers
        self.word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=10000)
        self.tfidf_dictations = self.word_vec.fit_transform(self.dictations)

        # Pre-index cases by template
        self.template_to_indices: Dict[str, List[int]] = {}
        for idx, c in enumerate(train_cases):
            self.template_to_indices.setdefault(c.template_content, []).append(idx)

        # Pre-index cases by modality and body part
        self.modality_bp_to_indices: Dict[Tuple[str, str], List[int]] = {}
        for idx, c in enumerate(train_cases):
            key = (c.modality.upper(), c.body_part.upper())
            self.modality_bp_to_indices.setdefault(key, []).append(idx)

    def retrieve(
        self,
        query: RadiologyCase,
        top_k: int = 3,
        exclude_case_id: Optional[str] = None
    ) -> List[Tuple[RadiologyCase, float]]:
        """Retrieve top_k similar training cases for a query case."""
        # 1. Candidate pool: exact template match if available
        candidate_indices = self.template_to_indices.get(query.template_content, [])
        if exclude_case_id:
            candidate_indices = [i for i in candidate_indices if self.train_cases[i].case_id != exclude_case_id]

        # If no candidates with exact template, fall back to same modality + body part
        if not candidate_indices:
            key = (query.modality.upper(), query.body_part.upper())
            candidate_indices = self.modality_bp_to_indices.get(key, [])
            if exclude_case_id:
                candidate_indices = [i for i in candidate_indices if self.train_cases[i].case_id != exclude_case_id]

        # If still none, use all train cases
        if not candidate_indices:
            candidate_indices = [i for i in range(len(self.train_cases)) if (not exclude_case_id or self.train_cases[i].case_id != exclude_case_id)]

        if not candidate_indices:
            return []

        # 2. Score candidates by dictation similarity
        q_vec = self.word_vec.transform([query.dictation or ""])
        cand_vecs = self.tfidf_dictations[candidate_indices]
        sims = cosine_similarity(q_vec, cand_vecs)[0]

        # Rank candidates
        ranked_order = np.argsort(-sims)
        results = []
        for rank_idx in ranked_order[:top_k]:
            orig_idx = candidate_indices[rank_idx]
            results.append((self.train_cases[orig_idx], float(sims[rank_idx])))

        return results
