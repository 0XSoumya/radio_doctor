"""Deterministic validation and repair for generated radiology reports."""

import re
from typing import Tuple, List
from src.data import RadiologyCase
import src.templates as tm
from src.impression import clean_impression_text


class ReportValidationError(Exception):
    pass


def validate_and_repair_report(
    report_text: str,
    case: RadiologyCase,
    strict: bool = False
) -> Tuple[bool, str, List[str]]:
    """Validate report structure and repair minor issues.
    
    Returns:
        (is_valid, repaired_report_text, list_of_warnings)
    """
    warnings = []
    text = report_text.strip()

    # 1. Structure checks
    if "FINDINGS:" not in text:
        msg = "Report missing FINDINGS section"
        if strict:
            raise ReportValidationError(msg)
        warnings.append(msg)
        text = f"FINDINGS:\n{text}"

    if "IMPRESSION:" not in text:
        msg = "Report missing IMPRESSION section"
        if strict:
            raise ReportValidationError(msg)
        warnings.append(msg)
        tpl_parsed = tm.parse_template(case.template_content)
        text = f"{text}\n\nIMPRESSION:\n{tpl_parsed.impression}"

    # 2. Check for unresolved bracket placeholders
    leftover_brackets = re.findall(r'\[(.*?)\]', text)
    if leftover_brackets:
        warnings.append(f"Unresolved placeholders found: {leftover_brackets}")
        parsed = tm.parse_template(text)
        parsed_resolved = tm.resolve_template_placeholders(parsed, case)
        text = tm.render_report(parsed_resolved)

    # 3. Clean and standardize impression numbering
    parsed = tm.parse_template(text)
    parsed.impression = clean_impression_text(parsed.impression)

    # 4. Check that field order and all template fields exist
    tpl_parsed = tm.parse_template(case.template_content)
    if tpl_parsed.is_colon_delimited:
        rep_field_dict = parsed.get_field_dict()

        reconstructed_fields = []
        for t_field in tpl_parsed.fields:
            if t_field.is_header:
                reconstructed_fields.append(tm.TemplateField(name=t_field.name, text="", is_header=True))
            else:
                updated_text = rep_field_dict.get(t_field.name.upper(), t_field.text)
                reconstructed_fields.append(tm.TemplateField(name=t_field.name, text=updated_text))

        # Check for any extra valid fields (e.g. OTHER FINDINGS)
        tpl_names_upper = {f.name.upper() for f in tpl_parsed.fields}
        for r_name, r_val in rep_field_dict.items():
            if r_name not in tpl_names_upper:
                if r_val.strip():
                    reconstructed_fields.append(tm.TemplateField(name=r_name, text=r_val))

        parsed.fields = reconstructed_fields
        text = tm.render_report(parsed)

    return (len(warnings) == 0, text, warnings)
