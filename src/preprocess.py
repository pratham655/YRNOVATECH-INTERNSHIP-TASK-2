
"""Download, clean, and prepare the UCI Student Performance dataset."""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import requests


ROOT_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
SCREENSHOTS_DIR = ROOT_DIR / "screenshots"

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/320/"
    "student+performance.zip"
)
SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/320/student-performance"

EXPECTED_FILES = ("student-mat.csv", "student-por.csv")

REQUIRED_COLUMNS = {"G1", "G2", "G3", "Medu", "Fedu"}


def download_course_files() -> dict[str, bytes]:
    """Download the UCI ZIP and extract both course CSV files."""
    print(f"Downloading public dataset from: {DATASET_URL}")

    try:
        response = requests.get(DATASET_URL, timeout=60)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            "Dataset download failed. Check your internet connection and retry."
        ) from exc

    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as outer_archive:
            inner_zip_bytes = outer_archive.read("student.zip")

        with zipfile.ZipFile(io.BytesIO(inner_zip_bytes)) as archive:
            archive_names = archive.namelist()
            files: dict[str, bytes] = {}

            for expected in EXPECTED_FILES:
                if expected not in archive_names:
                    raise RuntimeError(
                        f"Expected file {expected!r} was not found in student.zip."
                    )
                files[expected] = archive.read(expected)

            return files

    except (zipfile.BadZipFile, KeyError) as exc:
        raise RuntimeError(
            "The downloaded ZIP has an unexpected structure."
        ) from exc


def load_course_csv(
    filename: str,
    file_bytes: bytes,
    course_name: str,
) -> pd.DataFrame:
    """Read a course CSV and validate its required columns."""
    try:
        df = pd.read_csv(
            io.BytesIO(file_bytes),
            sep=";",
            encoding="utf-8",
        )
    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise RuntimeError(f"Could not read {filename}.") from exc

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise ValueError(
            f"{filename} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    df["course"] = course_name
    df["source_file"] = filename
    return df


def clean_and_engineer_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Clean records and create additional useful features."""
    original_rows = len(df)
    original_columns = len(df.columns)

    missing_before = int(df.isna().sum().sum())
    duplicate_rows_before = int(df.duplicated().sum())

    # Remove exact duplicate records.
    df = df.drop_duplicates().copy()
    duplicates_removed = original_rows - len(df)

    # Convert numeric-looking columns to numeric values where appropriate.
    for column in df.columns:
        if df[column].dtype == "object":
            converted = pd.to_numeric(df[column], errors="coerce")

            if converted.notna().sum() == df[column].notna().sum():
                df[column] = converted

    # Impute missing numeric values with the median and categorical values
    # with the most frequent value.
    numeric_columns = df.select_dtypes(include="number").columns
    categorical_columns = df.select_dtypes(
        exclude="number"
    ).columns

    imputation_counts = {}

    for column in numeric_columns:
        missing_count = int(df[column].isna().sum())

        if missing_count:
            median_value = df[column].median()
            df[column] = df[column].fillna(median_value)
            imputation_counts[column] = missing_count

    for column in categorical_columns:
        missing_count = int(df[column].isna().sum())

        if missing_count:
            modes = df[column].mode(dropna=True)

            if not modes.empty:
                df[column] = df[column].fillna(modes.iloc[0])
                imputation_counts[column] = missing_count
            else:
                df[column] = df[column].fillna("Unknown")
                imputation_counts[column] = missing_count

    # Feature engineering: average parent education level.
    df["avg_parent_education"] = (
        df["Medu"] + df["Fedu"]
    ) / 2

    # Early grade average: mean of the first two period grades.
    df["early_grade_average"] = (
        df["G1"] + df["G2"]
    ) / 2

    # Grade change between the first and final periods.
    df["grade_change_g1_to_g3"] = df["G3"] - df["G1"]

    # Categorize the final grade on the 0–20 scale.
    df["final_grade_band"] = pd.cut(
        df["G3"],
        bins=[-1, 9, 13, 16, 20],
        labels=["At risk", "Developing", "Good", "Excellent"],
        include_lowest=True,
    ).astype("string")

    # Ensure no missing values remain in the final dataset.
    remaining_missing = int(df.isna().sum().sum())
    if remaining_missing:
        raise ValueError(
            f"Cleaning left {remaining_missing} missing values."
        )

    remaining_duplicates = int(df.duplicated().sum())
    if remaining_duplicates:
        raise ValueError(
            f"Cleaning left {remaining_duplicates} duplicate rows."
        )

    if len(df) < 1000:
        raise ValueError(
            f"Expected at least 1,000 records, but found {len(df)}."
        )

    report = {
        "dataset_source": SOURCE_PAGE,
        "download_url": DATASET_URL,
        "original_rows": int(original_rows),
        "rows_after_duplicate_removal": int(len(df)),
        "duplicate_rows_before_cleaning": int(duplicate_rows_before),
        "duplicate_rows_removed": int(duplicates_removed),
        "missing_cells_before_cleaning": int(missing_before),
        "missing_values_imputed_by_column": imputation_counts,
        "missing_cells_after_cleaning": int(remaining_missing),
        "duplicate_rows_after_cleaning": int(remaining_duplicates),
        "original_column_count": int(original_columns),
        "final_column_count": int(len(df.columns)),
        "final_row_count": int(len(df)),
        "course_counts": {
            str(key): int(value)
            for key, value in df["course"].value_counts().items()
        },
        "feature_engineering": [
            "avg_parent_education",
            "early_grade_average",
            "grade_change_g1_to_g3",
            "final_grade_band",
        ],
        "final_grade_band_counts": {
            str(key): int(value)
            for key, value in df["final_grade_band"]
            .value_counts()
            .items()
        },
        "validation": {
            "at_least_1000_records": len(df) >= 1000,
            "no_missing_cells": remaining_missing == 0,
            "no_duplicate_rows": remaining_duplicates == 0,
        },
    }

    return df, report


def create_visualizations(df: pd.DataFrame) -> None:
    """Save simple plots documenting the processed dataset."""
    SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    # Distribution of final grades.
    plt.figure(figsize=(8, 5))
    df["G3"].plot(
        kind="hist",
        bins=21,
        edgecolor="black",
    )
    plt.title("Distribution of Final Student Grades")
    plt.xlabel("Final grade (G3)")
    plt.ylabel("Number of records")
    plt.tight_layout()
    plt.savefig(
        SCREENSHOTS_DIR / "final_grade_distribution.png",
        dpi=150,
    )
    plt.close()

    # Number of records per course.
    plt.figure(figsize=(7, 5))
    df["course"].value_counts().plot(kind="bar")
    plt.title("Records by Course")
    plt.xlabel("Course")
    plt.ylabel("Number of records")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(
        SCREENSHOTS_DIR / "course_record_counts.png",
        dpi=150,
    )
    plt.close()


def main() -> None:
    """Run the complete preprocessing workflow."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    try:
        course_files = download_course_files()

        math_df = load_course_csv(
            "student-mat.csv",
            course_files["student-mat.csv"],
            "Math",
        )
        portuguese_df = load_course_csv(
            "student-por.csv",
            course_files["student-por.csv"],
            "Portuguese",
        )

        combined_df = pd.concat(
            [math_df, portuguese_df],
            ignore_index=True,
        )

        cleaned_df, report = clean_and_engineer_features(combined_df)

        output_csv = PROCESSED_DIR / "cleaned_dataset.csv"
        output_json = REPORTS_DIR / "preprocessing_report.json"

        cleaned_df.to_csv(output_csv, index=False)

        with output_json.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)

        create_visualizations(cleaned_df)

        print("\nPREPROCESSING COMPLETED SUCCESSFULLY")
        print(f"Math course records: {len(math_df)}")
        print(f"Portuguese course records: {len(portuguese_df)}")
        print(f"Total cleaned records: {len(cleaned_df)}")
        print(f"Final columns: {len(cleaned_df.columns)}")
        print(f"Missing cells remaining: {cleaned_df.isna().sum().sum()}")
        print(f"Duplicate rows remaining: {cleaned_df.duplicated().sum()}")
        print(f"Cleaned dataset: {output_csv}")
        print(f"Preprocessing report: {output_json}")
        print(f"Visualizations: {SCREENSHOTS_DIR}")

    except (
        RuntimeError,
        ValueError,
        OSError,
        pd.errors.ParserError,
        requests.RequestException,
    ) as exc:
        raise SystemExit(f"ERROR: {exc}") from exc


if __name__ == "__main__":
    main()
