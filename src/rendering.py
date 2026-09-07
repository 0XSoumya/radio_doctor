"""Deterministic minimal-edit report rendering."""

from typing import Dict, Any, List
import src.templates as tm
from src.impression import clean_impression_text


def apply_edits_and_render(
    base_template: tm.ParsedTemplate,
    extraction_output: Dict[str, Any]
) -> str:
    """Apply structured edits onto base_template and render the final report."""
    rendered_tpl = base_template.clone()

    field_edits = extraction_output.get("field_edits", [])
    for edit in field_edits:
        fname = edit.get("field_name", "").strip()
        updated_text = edit.get("updated_text", "").strip()
        if not fname or not updated_text:
            continue

        existing_field = rendered_tpl.get_field(fname)
        if existing_field and not existing_field.is_header:
            existing_field.text = updated_text
        else:
            # Check if case-insensitive or fuzzy match
            matched = False
            for f in rendered_tpl.fields:
                if not f.is_header and (f.name.lower() == fname.lower() or fname.lower() in f.name.lower()):
                    f.text = updated_text
                    matched = True
                    break
            if not matched:
                # Add as extra field (e.g. OTHER FINDINGS)
                rendered_tpl.fields.append(tm.TemplateField(name=fname, text=updated_text))

    # Update impression
    extracted_imp = extraction_output.get("impression", "").strip()
    if extracted_imp:
        rendered_tpl.impression = clean_impression_text(extracted_imp)

    return tm.render_report(rendered_tpl)
