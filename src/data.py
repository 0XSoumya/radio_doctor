"""Data loading, representation, and cross-validation splitting."""

from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold
import src.config as cfg


@dataclass
class RadiologyCase:
    case_id: str
    modality: str
    body_part: str
    study_description: str
    patient_age_band: str
    patient_sex: str
    template_content: str
    dictation: str
    report: Optional[str] = None

    @classmethod
    def from_row(cls, row: pd.Series) -> "RadiologyCase":
        return cls(
            case_id=str(row.get("case_id", "")),
            modality=str(row.get("modality", "")),
            body_part=str(row.get("body_part", "")),
            study_description=str(row.get("study_description", "")),
            patient_age_band=str(row.get("patient_age_band", "")),
            patient_sex=str(row.get("patient_sex", "")),
            template_content=str(row.get("template_content", "")),
            dictation=str(row.get("dictation", "")),
            report=str(row["report"]) if "report" in row and pd.notna(row["report"]) else None,
        )


def load_train_cases(data_path: Optional[Path] = None) -> List[RadiologyCase]:
    path = data_path or (cfg.DATA_DIR / "train.csv")
    df = pd.read_csv(path)
    return [RadiologyCase.from_row(row) for _, row in df.iterrows()]


def load_test_cases(data_path: Optional[Path] = None) -> List[RadiologyCase]:
    path = data_path or (cfg.DATA_DIR / "test.csv")
    df = pd.read_csv(path)
    return [RadiologyCase.from_row(row) for _, row in df.iterrows()]


def get_cv_folds(
    cases: Optional[List[RadiologyCase]] = None,
    n_splits: int = 5,
    random_state: int = 42
) -> List[Tuple[List[int], List[int]]]:
    """Generate 5-fold cross validation split indices stratified by modality."""
    if cases is None:
        cases = load_train_cases()
        
    modalities = [c.modality for c in cases]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []
    for train_idx, val_idx in skf.split(cases, modalities):
        folds.append((list(train_idx), list(val_idx)))
    return folds
