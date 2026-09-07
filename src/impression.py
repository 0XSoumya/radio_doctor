"""Impression synthesis, standardization, and post-processing."""

import re


def clean_impression_text(impression: str) -> str:
    """Clean and standardize impression formatting, numbering, and spacing."""
    if not impression:
        return ""

    text = impression.strip()
    # Normalize excessive spaces
    text = re.sub(r'[ \t]+', ' ', text)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if not lines:
        return ""

    # Check if lines have any numbered items
    has_any_number = any(re.match(r'^\d+[\.\)]\s*', l) for l in lines)

    if has_any_number:
        numbered_lines = []
        item_idx = 1
        for l in lines:
            # Check for Lung-RADS score line
            if re.match(r'^LUNG[- ]?RADS', l, re.IGNORECASE):
                numbered_lines.append(l)
                continue
            cleaned = re.sub(r'^\d+[\.\)]\s*', '', l).strip()
            if cleaned:
                numbered_lines.append(f"{item_idx}. {cleaned}")
                item_idx += 1
        return "\n".join(numbered_lines)

    # If multiple lines without numbers, keep clean lines
    return "\n".join(lines)
