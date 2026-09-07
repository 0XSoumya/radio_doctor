"""Leakage-safe cross-validation runner for the Radiology Reporting Pipeline."""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from tqdm import tqdm

import src.config as cfg
from src.data import load_train_cases, get_cv_folds, RadiologyCase
from src.pipeline import RadiologyReportingPipeline
from src.scoring import compute_res_case, compute_res_corpus


def run_evaluation(
    fold_idx: int = 0,
    max_cases: int = 50,
    output_json: Optional[str] = None
):
    print(f"Loading training cases...")
    all_cases = load_train_cases()
    folds = get_cv_folds(all_cases, n_splits=5, random_state=42)
    train_indices, val_indices = folds[fold_idx]

    print(f"Fold {fold_idx}: {len(train_indices)} train cases, {len(val_indices)} val cases")
    train_cases = [all_cases[i] for i in train_indices]
    val_cases = [all_cases[i] for i in val_indices]

    if max_cases and max_cases < len(val_cases):
        print(f"Subsampling {max_cases} validation cases for evaluation budget...")
        val_cases = val_cases[:max_cases]

    print(f"Initializing pipeline with Fold {fold_idx} training pool only (leakage-safe)...")
    pipeline = RadiologyReportingPipeline(train_cases=train_cases)

    predictions = []
    references = []
    templates = []
    case_records = []

    for case in tqdm(val_cases, desc=f"Evaluating Fold {fold_idx}"):
        pred_report = pipeline.process_case(case, exclude_self_from_retrieval=True)
        predictions.append(pred_report)
        references.append(case.report)
        templates.append(case.template_content)

        scores = compute_res_case(pred_report, case.report, case.template_content)
        case_records.append({
            "case_id": case.case_id,
            "modality": case.modality,
            "body_part": case.body_part,
            "res": scores["res"],
            "findings_f": scores["findings_f"],
            "impression_i": scores["impression_i"]
        })

    corpus_scores = compute_res_corpus(predictions, references, templates)
    print("\n" + "=" * 50)
    print(f"CROSS-VALIDATION RESULTS (Fold {fold_idx}, {len(val_cases)} cases):")
    print(f"  Mean RES:          {corpus_scores['mean_res']:.5f}")
    print(f"  Mean Findings (F): {corpus_scores['mean_f']:.5f}")
    print(f"  Mean Impression (I): {corpus_scores['mean_i']:.5f}")
    print("=" * 50)

    # Breakdowns
    df_eval = pd.DataFrame(case_records)
    print("\nScore by Modality:")
    print(df_eval.groupby("modality")[["res", "findings_f", "impression_i"]].mean().to_string())

    if output_json:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({
                "fold": fold_idx,
                "summary": corpus_scores,
                "cases": case_records
            }, f, indent=2)
        print(f"\nSaved evaluation results to {out_path}")

    return corpus_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cross-validation")
    parser.add_argument("--fold", type=int, default=0, help="Fold index to evaluate (0-4)")
    parser.add_argument("--max-cases", type=int, default=30, help="Maximum validation cases to evaluate")
    parser.add_argument("--output", type=str, default="artifacts/candidates/cv_results_fold0.json", help="Output JSON path")
    args = parser.parse_args()

    run_evaluation(
        fold_idx=args.fold,
        max_cases=args.max_cases,
        output_json=args.output
    )
