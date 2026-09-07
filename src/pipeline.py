"""End-to-end Radiology Reporting Pipeline."""

import argparse
from pathlib import Path
from typing import List, Optional
import pandas as pd
from tqdm import tqdm

import src.config as cfg
from src.data import RadiologyCase, load_train_cases, load_test_cases
import src.templates as tm
from src.retrieval import CaseRetriever
from src.extraction import ExtractionEngine
from src.rendering import apply_edits_and_render
from src.validation import validate_and_repair_report

TRIVIAL_NORMAL_DICTATIONS = {
    'normal', 'normal study', 'unremarkable', 'normal examination',
    'no acute abnormality', 'no acute osseous abnormality',
    'no acute fracture or dislocation'
}


class RadiologyReportingPipeline:
    """End-to-end pipeline: template parsing -> retrieval -> structured extraction -> minimal edits -> validation."""

    def __init__(
        self,
        train_cases: Optional[List[RadiologyCase]] = None,
        extraction_engine: Optional[ExtractionEngine] = None
    ):
        self.train_cases = train_cases or load_train_cases()
        self.retriever = CaseRetriever(self.train_cases)
        self.extractor = extraction_engine or ExtractionEngine()

    def process_case(
        self,
        case: RadiologyCase,
        exclude_self_from_retrieval: bool = True
    ) -> str:
        """Process a single radiology case and return the final validated report."""
        # 1. Parse and resolve template placeholders
        raw_tpl = tm.parse_template(case.template_content)
        base_tpl = tm.resolve_template_placeholders(raw_tpl, case)

        # 2. Fast path for trivial normal dictations
        clean_dict = (case.dictation or "").strip().lower()
        if clean_dict in TRIVIAL_NORMAL_DICTATIONS:
            rendered_normal = tm.render_report(base_tpl)
            _, validated, _ = validate_and_repair_report(rendered_normal, case)
            return validated

        # 3. Retrieve best demonstration case
        exclude_id = case.case_id if exclude_self_from_retrieval else None
        demo_results = self.retriever.retrieve(case, top_k=1, exclude_case_id=exclude_id)
        demo_case = demo_results[0][0] if demo_results else None

        # 4. Structured extraction
        extraction_output = self.extractor.extract_and_transform(
            case=case,
            target_template=base_tpl,
            demo_case=demo_case
        )

        # 5. Apply minimal edits and render
        rendered_text = apply_edits_and_render(base_tpl, extraction_output)

        # 6. Validate and repair
        _, validated_text, _ = validate_and_repair_report(rendered_text, case)
        return validated_text

    def process_dataset(
        self,
        cases: List[RadiologyCase],
        show_progress: bool = True
    ) -> List[str]:
        """Process an entire list of cases."""
        reports = []
        iterator = tqdm(cases, desc="Processing cases") if show_progress else cases
        for case in iterator:
            report = self.process_case(case)
            reports.append(report)
        return reports


def main():
    parser = argparse.ArgumentParser(description="Run Radiology Reporting Pipeline")
    parser.add_argument("--input", type=str, default="data/test.csv", help="Path to input test CSV")
    parser.add_argument("--output", type=str, default="submission.csv", help="Path to output submission CSV")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"Loading input cases from {input_path}...")
    df = pd.read_csv(input_path)
    cases = [RadiologyCase.from_row(row) for _, row in df.iterrows()]

    print(f"Loaded {len(cases)} cases. Initializing pipeline...")
    pipeline = RadiologyReportingPipeline()

    print(f"Generating reports...")
    reports = pipeline.process_dataset(cases)

    sub_df = pd.DataFrame({
        "case_id": [c.case_id for c in cases],
        "report": reports
    })

    sub_df.to_csv(output_path, index=False)
    print(f"Saved submission to {output_path} with shape {sub_df.shape}")


if __name__ == "__main__":
    main()
