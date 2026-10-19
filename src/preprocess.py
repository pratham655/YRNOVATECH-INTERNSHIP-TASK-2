"""
Download and preprocess the UCI Student Performance dataset.

The workflow combines the mathematics and Portuguese course files as course
records (395 + 649 = 1,044 records in the source files). A record is a
student-course observation; rows across subjects should not be assumed to be
unique students.

Run from the repository root:
    python src/preprocess.py
"""

from __future__ import annotations

import io
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "reports"

DATASET_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/320/student+performance"
EXPECTED_FILES = ("student-mat.csv", "student-por.csv")
MINIMUM_ROWS = 1000
REQUIRED_COLUMNS = {"G1", "G2", "G3", "Medu", "Fedu"}


def download_course_files() -> dict[str, bytes]:
    """Download the official UCI ZIP and return the two course CSVs as bytes."""
    print(f"Downloading public dataset from: {SOURCE_PAGE}")
    try:
        response = requests.get(DATASET_URL, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            "Dataset download failed. Check your internet connection and retry."
        ) from exc

    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            names = set(archive.namelist())
            files: dict[str, bytes] = {}
            for expected in EXPECTED_FILES:
                match = next(
                    (name for name in names if name.endswith("/" + expected) or name == expected),
                    None,
                )
                if match is None:
                    raise RuntimeError(f"Expected file {expected!r} was not found in the downloaded ZIP.")
                files[expected] = archive.read(match)
            return files
    except zipfile.BadZipFile as exc:
        raise RuntimeError("The downloaded response was not a valid ZIP archive.") from exc


def load_course_csv(csv_bytes: bytes, course: str, filename: str) -> pd.DataFrame:
    """Parse one semicolon-delimited course file and mark its provenance."""
    try:
        frame = pd.read_csv(io.BytesIO(csv_bytes), sep=";")
    except Exception as exc:
        raise RuntimeError(f"Could not parse {filename}.") from exc

    frame.columns = [str(column).strip() for column in frame.columns]
    missing_columns = REQUIRED_COLUMNS - set(frame.columns)
    if missing_columns:
        raise ValueError(f"{filename} is missing expected columns: {sorted(missing_columns)}")

    frame["course"] = course
    frame["source_file"] = filename
    return frame


def clean_dataset(data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean missing values and duplicates and engineer documented features."""
    report: dict[str, Any] = {
        "rows_before_cleaning": int(len(data)),
        "columns_before_cleaning": int(data.shape[1]),
        "missing_values_before": {
            str(column): int(count)
            for column, count in data.isna().sum().items()
            if int(count) > 0
        },
        "exact_duplicate_rows_before": int(data.duplicated().sum()),
    }

    # Preserve source/course columns so each row remains traceable to its course file.
    before_duplicates = len(data)
    data = data.drop_duplicates().copy()
    report["duplicate_rows_removed"] = int(before_duplicates - len(data))

    # The source files are expected to be complete, but this makes the script
    # reproducible if future copies contain missing cells.
    numeric_columns = data.select_dtypes(include="number").columns.tolist()
    categorical_columns = data.select_dtypes(exclude="number").columns.tolist()
    imputation: dict[str, str] = {}

    for column in numeric_columns:
        missing_count = int(data[column].isna().sum())
        if missing_count:
            median = data[column].median()
            data[column] = data[column].fillna(median)
            imputation[column] = f"median ({median})"

    for column in categorical_columns:
        missing_count = int(data[column].isna().sum())
        if missing_count:
            modes = data[column].mode(dropna=True)
            fill_value = modes.iloc[0] if not modes.empty else "Unknown"
            data[column] = data[column].fillna(fill_value)
            imputation[column] = f"mode ({fill_value})"

    report["imputation_applied"] = imputation
    report["missing_values_after"] = int(data.isna().sum().sum())

    # Feature engineering based only on columns described in the UCI data.
    data["avg_parent_education"] = (data["Medu"] + data["Fedu"]) / 2
    data["early_grade_average"] = (data["G1"] + data["G2"]) / 2
    data["grade_change_g1_to_g3"] = data["G3"] - data["G1"]
    data["final_grade_band"] = pd.cut(
        data["G3"],
        bins=[-1, 9, 13, 16, 20],
        labels=["At risk", "Developing", "Proficient", "Advanced"],
        include_lowest=True,
    ).astype("string")

    report["rows_after_cleaning"] = int(len(data))
    report["columns_after_feature_engineering"] = int(data.shape[1])
    report["duplicate_rows_after"] = int(data.duplicated().sum())
    report["null_cells_after"] = int(data.isna().sum().sum())
    report["course_record_counts"] = {
        str(key): int(value)
        for key, value in data["course"].value_counts().sort_index().items()
    }
    report["final_grade_summary_G3"] = {
        "min": float(data["G3"].min()),
        "max": float(data["G3"].max()),
        "mean": round(float(data["G3"].mean()), 3),
        "median": float(data["G3"].median()),
    }
    report["final_grade_band_counts"] = {
        str(key): int(value)
        for key, value in data["final_grade_band"].value_counts(dropna=False).sort_index().items()
    }

    return data, report


def main() -> int:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        files = download_course_files()
        mathematics = load_course_csv(files["student-mat.csv"], "mathematics", "student-mat.csv")
        portuguese = load_course_csv(files["student-por.csv"], "portuguese", "student-por.csv")

        source_counts = {
            "student-mat.csv": int(len(mathematics)),
            "student-por.csv": int(len(portuguese)),
        }
        raw = pd.concat([mathematics, portuguese], ignore_index=True, sort=False)

        if len(raw) < MINIMUM_ROWS:
            raise ValueError(
                f"Dataset has {len(raw)} records; at least {MINIMUM_ROWS} are required."
            )

        cleaned, details = clean_dataset(raw)
        if cleaned.isna().any().any():
            raise ValueError("Validation failed: the cleaned dataset still contains missing cells.")
        if cleaned.duplicated().any():
            raise ValueError("Validation failed: duplicate rows remain after cleaning.")

        output_csv = PROCESSED_DIR / "cleaned_dataset.csv"
        output_json = REPORT_DIR / "preprocessing_report.json"
        cleaned.to_csv(output_csv, index=False)

        report = {
            "project": "YR NOVATECH AI & ML Internship - Task 2",
            "dataset": "UCI Student Performance",
            "dataset_source": SOURCE_PAGE,
            "download_url": DATASET_URL,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_file_row_counts": source_counts,
            "source_rows_combined": int(sum(source_counts.values())),
            "minimum_required_records": MINIMUM_ROWS,
            "minimum_requirement_met": bool(len(cleaned) >= MINIMUM_ROWS),
            "cleaned_csv": str(output_csv.relative_to(ROOT)).replace("\\", "/"),
            "cleaning_and_validation": details,
            "notes": [
                "Rows represent course records from separate mathematics and Portuguese files.",
                "Records from different course files are not assumed to be distinct people.",
                "Features are engineered for the preprocessing exercise, not proof of causal relationships.",
                "G1 and G2 are prior-period grades and are strongly related to G3 according to the UCI dataset documentation; consider this when using G3 as a prediction target.",
            ],
        }
        output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

        print("\nPreprocessing completed successfully.")
        print(f"Rows before cleaning: {details['rows_before_cleaning']}")
        print(f"Rows after cleaning:  {details['rows_after_cleaning']}")
        print(f"Columns after features: {details['columns_after_feature_engineering']}")
        print(f"Null cells after: {details['null_cells_after']}")
        print(f"Duplicate rows after: {details['duplicate_rows_after']}")
        print(f"Clean dataset: {output_csv}")
        print(f"Report: {output_json}")
        print("\nInspect the report and output CSV before submitting.")
        return 0

    except (RuntimeError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
