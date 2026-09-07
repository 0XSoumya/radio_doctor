"""Official RES Metric Scorer for NATOE Radiology Reporting Harness.

Implements text normalization, token weight assignment, ordered word-level
weighted Levenshtein distance, field-aware Findings scoring (F), and Impression scoring (I).

Case score:
    RES_case = 0.65 * F + 0.35 * I
Leaderboard score:
    mean(RES_case)
"""

import re
import unicodedata
from typing import Dict, List, Tuple, Any

# Function words (weight 0.25)
FUNCTION_WORDS = {
    'the', 'and', 'of', 'with', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'from',
    'by', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has',
    'had', 'do', 'does', 'did', 'or', 'nor', 'but', 'so', 'if', 'that', 'this',
    'these', 'those', 'there', 'here', 'it', 'its', 'they', 'their', 'them',
    'also', 'than', 'then', 'into', 'about', 'above', 'below', 'between', 'under', 'over'
}

# Critical clinical words (weight 4.0)
CRITICAL_WORDS = {
    # Negation
    'no', 'not', 'none', 'without', 'negative', 'non', 'never', 'neither',
    'unremarkable', 'intact', 'normal', 'preserved', 'absent', 'denies', 'denied', 'free',
    # Laterality / orientation
    'right', 'left', 'bilateral', 'bilaterally', 'unilateral', 'unilaterally',
    'midline', 'anterior', 'posterior', 'superior', 'inferior', 'medial', 'lateral',
    'proximal', 'distal', 'dorsal', 'ventral', 'axial', 'coronal', 'sagittal',
    # Severity / degree
    'mild', 'mildly', 'moderate', 'moderately', 'severe', 'severely', 'trace',
    'minimal', 'minimally', 'gross', 'grossly', 'marked', 'markedly', 'extensive',
    'extensively', 'subtle', 'significant', 'insignificant', 'small', 'large', 'tiny',
    'massive', 'minor', 'major', 'advanced', 'pronounced',
    # Acuity
    'acute', 'acutely', 'chronic', 'chronically', 'subacute', 'subacutely',
    # Units
    'mm', 'cm', 'm', 'cc', 'ml', 'degrees', 'degree', 'percent',
    # Spelled out numbers
    'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'
}

NUMBER_PATTERN = re.compile(r'^[+-]?\d+(?:\.\d+)?$')


def is_number(token: str) -> bool:
    """Check if token is a numeric value."""
    return bool(NUMBER_PATTERN.match(token))


def get_token_weight(token: str) -> float:
    """Return token weight: 4.0 for critical, 0.25 for function words, 2.0 for other content words."""
    if is_number(token) or token in CRITICAL_WORDS:
        return 4.0
    if token in FUNCTION_WORDS:
        return 0.25
    return 2.0


def normalize_text(text: str) -> List[str]:
    """Perform text normalization according to official competition rules.
    
    1. Unicode NFKC normalization
    2. Lowercase
    3. Remove leading list markers on lines
    4. Remove hyphens between letters ('osteo-arthritis' -> 'osteoarthritis')
    5. Tokenize letter-number boundaries ('3mm' -> '3 mm')
    6. Standardize common unit spellings ('millimeters' -> 'mm', 'centimeters' -> 'cm')
    7. Punctuation ignored except leading '+' or '-' directly attached to numbers
    8. Preserve word order
    """
    if not text:
        return []
    
    # 1. Unicode normalization
    text = unicodedata.normalize('NFKC', text)
    # 2. Lowercase
    text = text.lower()
    
    # 3. Remove leading list markers on lines (e.g. '1.', '1)', '(1)', '-', '*', '•')
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        cleaned = re.sub(r'^\s*(?:\d+[\.\)]|\([0-9a-z]+\)|[a-z][\.\)]|[\-\*•])\s+', '', line)
        cleaned_lines.append(cleaned)
    text = '\n'.join(cleaned_lines)
    
    # Also strip inline numbered list markers following sentence boundaries
    text = re.sub(r'(?<=[.;]\s)\s*(?:\d+[\.\)]|\([0-9a-z]+\)|[a-z][\.\)]|[\-\*•])\s+', ' ', text)
    
    # 4. Remove hyphens between letters ('osteo-arthritis' -> 'osteoarthritis')
    text = re.sub(r'(?<=[a-z])-(?=[a-z])', '', text)
    
    # 5. Tokenize letter-number boundaries ('l1-l2' -> 'l 1 - l 2', '3mm' -> '3 mm')
    text = re.sub(r'(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])', ' ', text)
    
    # 6. Standardize common unit spellings
    text = re.sub(r'\b(millimeters?|millimetres?)\b', 'mm', text)
    text = re.sub(r'\b(centimeters?|centimetres?)\b', 'cm', text)
    
    # 7. Extract valid tokens: numbers with optional +/- or alphabetic words
    tokens = re.findall(r'[+-]?\d+(?:\.\d+)?|[a-z]+', text)
    return tokens


def weighted_levenshtein(tokens1: List[str], tokens2: List[str]) -> float:
    """Compute ordered word-level weighted Levenshtein distance normalized by max total weight."""
    n, m = len(tokens1), len(tokens2)
    if n == 0 and m == 0:
        return 0.0
    
    w1 = [get_token_weight(t) for t in tokens1]
    w2 = [get_token_weight(t) for t in tokens2]
    sum1 = sum(w1)
    sum2 = sum(w2)
    denom = max(sum1, sum2)
    if denom == 0:
        return 0.0
    if n == 0 or m == 0:
        return min(1.0, (sum1 + sum2) / denom)
        
    prev = [0.0] * (m + 1)
    curr = [0.0] * (m + 1)
    for j in range(1, m + 1):
        prev[j] = prev[j-1] + w2[j-1]
        
    for i in range(1, n + 1):
        curr[0] = prev[0] + w1[i-1]
        wi = w1[i-1]
        t1_i = tokens1[i-1]
        for j in range(1, m + 1):
            if t1_i == tokens2[j-1]:
                curr[j] = prev[j-1]
            else:
                curr[j] = min(
                    prev[j] + wi,                       # delete
                    curr[j-1] + w2[j-1],                 # insert
                    prev[j-1] + max(wi, w2[j-1])         # substitute
                )
        prev, curr = curr, prev
        
    return min(1.0, prev[m] / denom)


def parse_report_sections(report_text: str) -> Tuple[Dict[str, str], str]:
    """Parse report into findings fields dictionary and impression string."""
    parts = re.split(r'\n\s*IMPRESSION:\s*', report_text, maxsplit=1)
    findings_raw = parts[0]
    impression_raw = parts[1] if len(parts) > 1 else ''
    findings_raw = re.sub(r'^\s*FINDINGS:\s*', '', findings_raw)
    
    field_matches = list(re.finditer(r'(?:^|\n)([A-Za-z0-9 /,\-_&]+):\s*', findings_raw))
    if not field_matches:
        fields = {'_FREE_FORM_': findings_raw.strip()}
    else:
        fields = {}
        for i, match in enumerate(field_matches):
            name = match.group(1).strip().upper()
            start = match.end()
            end = field_matches[i+1].start() if i+1 < len(field_matches) else len(findings_raw)
            fields[name] = findings_raw[start:end].strip()
    return fields, impression_raw.strip()


def compute_findings_score(
    pred_fields: Dict[str, str],
    ref_fields: Dict[str, str],
    tpl_fields: Dict[str, str]
) -> float:
    """Compute field-aware Findings score F.
    
    Changed reference field: weight = 3
    Unchanged reference field: weight = 1
    Missing field in pred: penalized with score = 1.0
    Unexpected extra field in pred: penalized with weight = 1, score = 1.0
    """
    total_f_weight = 0.0
    weighted_f_score = 0.0
    
    for fname, ref_val in ref_fields.items():
        ref_tokens = normalize_text(ref_val)
        tpl_val = tpl_fields.get(fname, '')
        tpl_tokens = normalize_text(tpl_val)
        
        # Determine if field changed from template
        is_changed = (ref_tokens != tpl_tokens)
        fw = 3.0 if is_changed else 1.0
        
        if fname in pred_fields:
            pred_tokens = normalize_text(pred_fields[fname])
            fscore = weighted_levenshtein(pred_tokens, ref_tokens)
        else:
            fscore = 1.0  # missing expected field
            
        weighted_f_score += fw * fscore
        total_f_weight += fw
        
    # Penalize unexpected fields in prediction
    for fname in pred_fields:
        if fname not in ref_fields:
            fw = 1.0
            weighted_f_score += fw * 1.0
            total_f_weight += fw
            
    return (weighted_f_score / total_f_weight) if total_f_weight > 0 else 0.0


def compute_impression_score(pred_imp: str, ref_imp: str) -> float:
    """Compute Impression score I using weighted word edit distance."""
    pred_tokens = normalize_text(pred_imp)
    ref_tokens = normalize_text(ref_imp)
    return weighted_levenshtein(pred_tokens, ref_tokens)


def compute_res_case(
    pred_report: str,
    ref_report: str,
    template_content: str
) -> Dict[str, float]:
    """Compute case-level RES score.
    
    RES_case = 0.65 * F + 0.35 * I
    """
    pred_fields, pred_imp = parse_report_sections(pred_report)
    ref_fields, ref_imp = parse_report_sections(ref_report)
    tpl_fields, _ = parse_report_sections(template_content)
    
    F = compute_findings_score(pred_fields, ref_fields, tpl_fields)
    I = compute_impression_score(pred_imp, ref_imp)
    RES = 0.65 * F + 0.35 * I
    
    return {
        "res": RES,
        "findings_f": F,
        "impression_i": I
    }


def compute_res_corpus(
    predictions: List[str],
    references: List[str],
    templates: List[str]
) -> Dict[str, float]:
    """Compute mean RES across all cases."""
    assert len(predictions) == len(references) == len(templates)
    res_list = []
    f_list = []
    i_list = []
    
    for pred, ref, tpl in zip(predictions, references, templates):
        scores = compute_res_case(pred, ref, tpl)
        res_list.append(scores["res"])
        f_list.append(scores["findings_f"])
        i_list.append(scores["impression_i"])
        
    return {
        "mean_res": sum(res_list) / len(res_list),
        "mean_f": sum(f_list) / len(f_list),
        "mean_i": sum(i_list) / len(i_list),
        "count": len(res_list)
    }
