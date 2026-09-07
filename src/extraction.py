"""Structured LLM extraction and transformation engine with persistent disk caching."""

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from groq import Groq
import src.config as cfg
from src.data import RadiologyCase
import src.templates as tm
from src.transformations import normalize_dictation_shorthand


def split_dictation_explicit_header(dictation: str) -> Tuple[str, str]:
    """Split dictation into findings part and explicit impression part if an explicit header exists."""
    pat = re.compile(r'(?:^|\n)\s*(?:impression|conclusion|summary|overall|opinion)[:\s]*\n', re.IGNORECASE)
    m = pat.search(dictation)
    if m:
        findings = dictation[:m.start()].strip()
        impression = dictation[m.end():].strip()
        return findings, impression
    return dictation.strip(), ""


class ExtractionEngine:
    """Handles prompt construction, Groq LLM invocation, disk caching, and response parsing."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        reasoning_effort: str = "low",
        max_retries: int = 3
    ):
        self.model_name = model_name or cfg.MODEL_NAME
        self.cache_dir = cache_dir or (cfg.CACHE_DIR / "llm")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.reasoning_effort = reasoning_effort
        self.max_retries = max_retries
        self._client: Optional[Groq] = None

    @property
    def client(self) -> Groq:
        if self._client is None:
            if not cfg.GROQ_API_KEY:
                raise ValueError("GROQ_API_KEY is not set. Please set it in .env or .env.local")
            self._client = Groq(api_key=cfg.GROQ_API_KEY)
        return self._client

    def _get_cache_key(self, messages: List[Dict[str, str]]) -> str:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "reasoning_effort": self.reasoning_effort
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _call_llm_cached(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        cache_key = self._get_cache_key(messages)
        cache_file = self.cache_dir / f"{cache_key}.json"

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        # Call API with retry
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    response_format={"type": "json_object"},
                    reasoning_effort=self.reasoning_effort,
                    temperature=cfg.DEFAULT_TEMPERATURE,
                    max_tokens=2048
                )
                raw_text = resp.choices[0].message.content or "{}"
                data = json.loads(raw_text)

                # Save to cache
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)

                return data
            except Exception as e:
                last_err = e
                wait_time = (2 ** attempt) * 1.5
                time.sleep(wait_time)

        raise RuntimeError(f"Failed LLM call after {self.max_retries} attempts: {last_err}")

    def build_messages(
        self,
        case: RadiologyCase,
        target_template: tm.ParsedTemplate,
        demo_case: Optional[RadiologyCase] = None
    ) -> List[Dict[str, str]]:
        system_prompt = (
            "You are an expert radiologist and medical reporting assistant.\n"
            "Your task is to transform a standard normal radiology template into a precise final report based on the radiologist's dictation.\n\n"
            "CRITICAL REPORTING & LEVENSHTEIN-ACCURACY RULES:\n"
            "1. MINIMAL EDITS: ONLY modify fields where the dictation provides new, abnormal, or supplementary clinical facts.\n"
            "   All untouched normal template fields MUST remain completely untouched. Do NOT include unchanged fields in 'field_edits'.\n"
            "2. FIELD ORDER & WORDING IN MODIFIED FIELDS:\n"
            "   - State the abnormal/dictated finding FIRST, followed by the uncontradicted normal template statement (e.g. 'No acute fracture is seen.').\n"
            "   - Contradiction resolution: When an abnormality is present in a field (e.g. joint space narrowing), do NOT keep contradictory normal template statements like 'The joint spaces are normal.' Instead, replace with 'The remaining joint spaces are preserved.' or omit the contradicted normal sentence.\n"
            "3. ACCURATE FIELD ROUTING:\n"
            "   - BONES / OSSEOUS STRUCTURES: alignment, fractures, enthesophytes, bone spurs, trochanteric/tuberosity changes, cortical defects, marrow signal, spondylosis, lytic/blastic lesions.\n"
            "   - JOINTS: narrowing, effusion, cartilage, meniscus, labrum, subluxation, osteoarthrosis, arthritis.\n"
            "   - SOFT TISSUES: edema, swelling, hematoma, cystic lesions, subcutaneous findings, vascular calcifications.\n"
            "   - OTHER FINDINGS: foreign bodies, surgical clips, hardware, pacemakers, incidental extra-regional findings.\n"
            "4. PRESERVE DICTATION WORDING & ACCURACY:\n"
            "   - Preserve exact words, numbers, and medical acronyms from dictation (e.g., keep '1st MTP joint', 'L1-L2'; do NOT expand acronyms).\n"
            "   - Preserve exact measurements (e.g., '3 mm', '1.2 cm') and laterality ('right', 'left', 'bilateral').\n"
            "   - Strict Negation: If the dictation says 'no fracture' or 'no effusion', treat as negative. Never invent findings.\n"
            "   - Never copy facts, laterality, or measurements from reference demonstrations.\n"
            "5. IMPRESSION SYNTHESIS RULES:\n"
            "   - Summary Phrases Priority: If summary/conclusion statements are provided or appear in dictation, USE THOSE EXACT SUMMARY PHRASES for your numbered impression items!\n"
            "   - Each distinct dictated abnormality should be its own numbered item in the impression.\n"
            "   - For MRI and CT: Present findings directly; do NOT invent or append 'No acute fracture or dislocation' unless explicitly present in the dictation or template default impression.\n"
            "   - For X-RAYS with abnormalities where the template default impression was 'No acute osseous abnormality.', include 'No acute fracture or dislocation.' as the final numbered item.\n"
            "   - Target Organ Laterality: If the study is unilateral (e.g., RT Ankle, Left knee), ensure laterality is attached to the abnormal findings in the impression (e.g., 'right ankle cellulitis', 'right ankle joint effusion').\n"
            "   - Format: Format as a clean numbered list: '1. ...\\n2. ...'\n"
            "   - Lung-RADS: If Lung-RADS is present, include 'LUNG-RADS SCORE: Category [X] : [Description]'."
        )

        demo_str = ""
        if demo_case and demo_case.report:
            demo_tpl = tm.parse_template(demo_case.template_content)
            demo_rep = tm.parse_template(demo_case.report)
            demo_edits = []
            for f in demo_rep.fields:
                if not f.is_header:
                    orig = demo_tpl.get_field(f.name)
                    if not orig or orig.text.strip() != f.text.strip():
                        demo_edits.append({"field_name": f.name, "updated_text": f.text})

            demo_json = json.dumps({
                "field_edits": demo_edits,
                "impression": demo_rep.impression
            }, indent=2)

            demo_str = (
                f"--- REFERENCE DEMONSTRATION (For formatting style and routing only; do NOT copy facts) ---\n"
                f"Demonstration Dictation:\n{demo_case.dictation}\n\n"
                f"Demonstration Output JSON:\n{demo_json}\n"
                f"------------------------------------------------------------------------------------------\n\n"
            )

        field_list_str = "\n".join([f"- {f.name}: {f.text}" for f in target_template.fields if not f.is_header])
        normalized_dictation = normalize_dictation_shorthand(case.dictation)
        dict_findings, explicit_imp = split_dictation_explicit_header(normalized_dictation)

        study_context_str = f"Modality: {case.modality}\nBody Part: {case.body_part}\nStudy Description: {case.study_description}"

        explicit_imp_hint = ""
        if explicit_imp:
            explicit_imp_hint = f"\nDICTATED EXPLICIT IMPRESSION (Use this verbatim for impression):\n{explicit_imp}\n"

        user_content = (
            f"{demo_str}"
            f"TARGET CASE TO REPORT:\n"
            f"{study_context_str}\n\n"
            f"TEMPLATE FIELDS:\n{field_list_str}\n\n"
            f"TEMPLATE DEFAULT IMPRESSION:\n{target_template.impression}\n\n"
            f"DICTATION:\n{dict_findings}\n"
            f"{explicit_imp_hint}\n"
            f"Respond ONLY in valid JSON matching this exact structure:\n"
            f"{{\n"
            f'  "field_edits": [\n'
            f'    {{"field_name": "<EXACT_MATCHING_FIELD_NAME>", "updated_text": "<updated field text>"}}\n'
            f'  ],\n'
            f'  "impression": "<synthesized impression text>"\n'
            f"}}"
        )

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]

    def extract_and_transform(
        self,
        case: RadiologyCase,
        target_template: tm.ParsedTemplate,
        demo_case: Optional[RadiologyCase] = None
    ) -> Dict[str, Any]:
        """Perform structured extraction and return field edits and impression."""
        messages = self.build_messages(case, target_template, demo_case)
        return self._call_llm_cached(messages)
