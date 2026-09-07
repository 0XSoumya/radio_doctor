"""Template parsing, representation, manipulation, and placeholder resolution."""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from src.data import RadiologyCase


@dataclass
class TemplateField:
    name: str
    text: str
    is_header: bool = False  # e.g., category grouping header without direct text


@dataclass
class ParsedTemplate:
    fields: List[TemplateField] = field(default_factory=list)
    impression: str = ""
    raw_text: str = ""
    is_colon_delimited: bool = True

    def get_field_dict(self) -> Dict[str, str]:
        """Return mapping of uppercase field name to content."""
        return {f.name.upper(): f.text for f in self.fields if not f.is_header}

    def get_field(self, name: str) -> Optional[TemplateField]:
        name_upper = name.strip().upper()
        for f in self.fields:
            if f.name.strip().upper() == name_upper:
                return f
        return None

    def clone(self) -> "ParsedTemplate":
        return ParsedTemplate(
            fields=[TemplateField(f.name, f.text, f.is_header) for f in self.fields],
            impression=self.impression,
            raw_text=self.raw_text,
            is_colon_delimited=self.is_colon_delimited
        )


def parse_template(raw_text: str) -> ParsedTemplate:
    """Parse a template or report into its structured representation."""
    if not raw_text:
        return ParsedTemplate()

    parts = re.split(r'\n\s*IMPRESSION:\s*', raw_text, maxsplit=1)
    findings_part = parts[0]
    impression_part = parts[1] if len(parts) > 1 else ""

    # Strip FINDINGS: header
    findings_part = re.sub(r'^\s*FINDINGS:\s*', '', findings_part)

    # Search for field headers
    # A field starts with NAME: on a line, not consuming subsequent newlines
    field_matches = list(re.finditer(r'(?:^|\n)\s*([A-Za-z0-9 /,\-_&]+):[ \t]*', findings_part))

    if not field_matches:
        # Colon-less template (e.g., cervical spine list of sentences)
        lines = [line.strip() for line in findings_part.split('\n') if line.strip()]
        fields = [TemplateField(name=f"LINE_{i+1}", text=line) for i, line in enumerate(lines)]
        return ParsedTemplate(
            fields=fields,
            impression=impression_part.strip(),
            raw_text=raw_text,
            is_colon_delimited=False
        )

    fields: List[TemplateField] = []
    for i, match in enumerate(field_matches):
        name = match.group(1).strip()
        start = match.end()
        end = field_matches[i + 1].start() if i + 1 < len(field_matches) else len(findings_part)
        content = findings_part[start:end].strip()
        is_header = (len(content) == 0 and i + 1 < len(field_matches))
        fields.append(TemplateField(name=name, text=content, is_header=is_header))

    return ParsedTemplate(
        fields=fields,
        impression=impression_part.strip(),
        raw_text=raw_text,
        is_colon_delimited=True
    )


def resolve_placeholders_in_text(text: str, case: RadiologyCase) -> str:
    """Resolve placeholders like [generic], [_laterality_], [left/right], etc."""
    if not text or "[" not in text:
        return text

    # 1. [generic] spine
    if "[generic]" in text:
        bp = (case.body_part or "").lower()
        sd = (case.study_description or "").lower()
        if "lumbar" in bp or "lsp" in sd:
            spine_type = "lumbar"
        elif "thoracic" in bp or "tsp" in sd:
            spine_type = "thoracic"
        elif "cervical" in bp or "csp" in sd:
            spine_type = "cervical"
        elif "lumbosacral" in bp:
            spine_type = "lumbosacral"
        else:
            spine_type = bp.replace(" spine", "").strip() or "spine"
        text = text.replace("[generic]", spine_type)

    # 2. Laterality placeholders
    combined = f"{case.study_description} {case.dictation}".lower()
    has_rt = bool(re.search(r'\b(right|rt)\b', combined))
    has_lt = bool(re.search(r'\b(left|lt)\b', combined))
    has_bilat = bool(re.search(r'\b(bilateral|bilat|blev)\b', combined))

    if has_bilat or (has_rt and has_lt):
        lat = "bilateral"
    elif has_lt:
        lat = "left"
    elif has_rt:
        lat = "right"
    else:
        lat = "right"

    if "[_laterality_]" in text:
        text = text.replace("[_laterality_]", lat)
    if "[left/right]" in text:
        text = text.replace("[left/right]", "left" if lat == "left" else "right")
    if "[left/right/bilateral]" in text:
        text = text.replace("[left/right/bilateral]", lat)

    # 3. [with or without]
    if "[with or without]" in text:
        sd = (case.study_description or "").upper()
        if "WO" in sd or "WITHOUT" in sd:
            text = text.replace("[with or without]", "without")
        elif "W" in sd or "WITH" in sd:
            text = text.replace("[with or without]", "with")
        else:
            text = text.replace("[with or without]", "without")

    # 4. Strip any remaining brackets around text
    text = re.sub(r'\[(.*?)\]', r'\1', text)
    return text


def resolve_template_placeholders(tpl: ParsedTemplate, case: RadiologyCase) -> ParsedTemplate:
    """Resolve placeholders across all fields and impression in a parsed template."""
    resolved = tpl.clone()
    for f in resolved.fields:
        f.text = resolve_placeholders_in_text(f.text, case)
    resolved.impression = resolve_placeholders_in_text(resolved.impression, case)
    return resolved


def render_report(parsed: ParsedTemplate) -> str:
    """Render a ParsedTemplate into standard report format."""
    sections = ["FINDINGS:"]
    if parsed.is_colon_delimited:
        for f in parsed.fields:
            if f.is_header:
                sections.append(f"{f.name}:")
            else:
                sections.append(f"{f.name}: {f.text}")
    else:
        # Free-form lines
        for f in parsed.fields:
            if f.text:
                sections.append(f.text)

    findings_str = "\n\n".join(sections)
    impression_str = f"IMPRESSION:\n{parsed.impression.strip()}"
    return f"{findings_str}\n\n{impression_str}"
