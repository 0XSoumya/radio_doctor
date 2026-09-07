"""Transformation memory and clinical abbreviation/shorthand normalizer."""

import re
from typing import Dict, List, Set
from src.data import RadiologyCase
import src.templates as tm

# Comprehensive clinical shorthand & abbreviation expansions
SHORTHAND_EXPANSIONS = {
    r'\bdegen\b': 'degenerative',
    r'\bleuko\b': 'leukoaraiosis',
    r'\bcacified\b': 'calcified',
    r'\blymphnodes\b': 'lymph nodes',
    r'\beffusiom\b': 'effusion',
    r'\bspondy\b': 'spondylosis',
    r'\boa\b': 'osteoarthritis',
    r'\bddd\b': 'degenerative disc disease',
    r'\bw/\b': 'with',
    r'\bw/o\b': 'without',
    r'\brt\b': 'right',
    r'\blt\b': 'left',
    r'\bant\b': 'anterior',
    r'\bpost\b': 'posterior',
    r'\bsup\b': 'superior',
    r'\binf\b': 'inferior',
    r'\blat\b': 'lateral',
    r'\bmed\b': 'medial',
    r'\bbilat\b': 'bilateral',
    r'\bfx\b': 'fracture',
    r'\bsteatosis\b': 'hepatic steatosis',
}


def normalize_dictation_shorthand(dictation: str) -> str:
    """Expand shorthand, abbreviations, and common typos in dictation."""
    if not dictation:
        return ""
    text = dictation
    for pattern, replacement in SHORTHAND_EXPANSIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    # Fix double spaces
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


class TransformationMemory:
    """Mined transformation memory from training cases."""

    def __init__(self, train_cases: List[RadiologyCase]):
        self.field_vocabulary: Dict[str, Set[str]] = {}
        self.common_modifications: Dict[str, List[str]] = {}
        self._mine_patterns(train_cases)

    def _mine_patterns(self, cases: List[RadiologyCase]):
        for case in cases:
            if not case.report:
                continue
            tpl_parsed = tm.parse_template(case.template_content)
            rep_parsed = tm.parse_template(case.report)
            tpl_dict = tpl_parsed.get_field_dict()
            rep_dict = rep_parsed.get_field_dict()

            for fname, rep_text in rep_dict.items():
                tpl_text = tpl_dict.get(fname, "")
                if rep_text.strip() != tpl_text.strip():
                    # Extract words that appear in dictation and report
                    d_words = set(re.findall(r'[a-zA-Z]{4,}', case.dictation.lower()))
                    r_words = set(re.findall(r'[a-zA-Z]{4,}', rep_text.lower()))
                    overlap = d_words.intersection(r_words)
                    self.field_vocabulary.setdefault(fname, set()).update(overlap)

    def get_related_fields_for_text(self, text: str) -> List[str]:
        """Given a text/finding, rank matching template field names based on learned vocabulary."""
        words = set(re.findall(r'[a-zA-Z]{4,}', text.lower()))
        scores = {}
        for fname, vocab in self.field_vocabulary.items():
            intersection = words.intersection(vocab)
            if intersection:
                scores[fname] = len(intersection)
        return sorted(scores.keys(), key=lambda k: scores[k], reverse=True)
