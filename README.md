# Data Collection and Preprocessing

**Internship:** YR NOVATECH AI & ML Internship — Task 2  
**Dataset:** UCI Student Performance  
**Status:** Initial workflow committed; run the script locally to generate and validate the cleaned dataset and report.

This project creates a reproducible data collection and preprocessing workflow using two public UCI course files. The source provides a Mathematics file with 395 records and a Portuguese-language file with 649 records, for **1,044 course records in total**. These are course records, not a guarantee of 1,044 unique individuals.

## Live Dashboard

Explore the deployed Streamlit dashboard here: **[Student Performance Data Explorer](https://yrnovatech-internship-task-2-sybjkzrr85xrpikmbdk2t2.streamlit.app/)**.

The dashboard provides dataset overview metrics, a preview and course filter, grade/course charts, preprocessing report details, and a cleaned CSV download.

## Dataset Source

- [UCI Student Performance dataset](https://archive.ics.uci.edu/dataset/320/student+performance)
- [Official dataset ZIP](https://archive.ics.uci.edu/static/public/320/student+performance.zip)

The script downloads the official ZIP and reads the two semicolon-delimited CSV files directly. It does not save the raw files in the repository.

## Repository Contents

- `src/preprocess.py` — reproducible collection, cleaning, feature engineering, and validation script
- `requirements.txt` — Python dependencies
- `data/raw/README.md` — raw source information
- `data/processed/cleaned_dataset.csv` — generated cleaned dataset (created when the script runs)
- `reports/preprocessing_report.json` — generated row counts, cleaning decisions, and validation results
- `screenshots/course_record_counts.png` and `screenshots/final_grade_distribution.png` — generated visualizations
- `demo/Task2.mp4` — code walkthrough video
- `app.py` — Streamlit dashboard for exploring the cleaned dataset

## What the Script Does

1. Downloads the public dataset from the official UCI source.
2. Reads `student-mat.csv` and `student-por.csv` using the correct semicolon separator.
3. Adds `course` and `source_file` provenance columns before combining the files.
4. Checks required grade and parent-education columns and verifies that the combined data has at least 1,000 records.
5. Reports missing values and removes exact duplicate rows.
6. Fills missing numeric values with each column's median and missing categorical values with the mode, only where needed.
7. Engineers `avg_parent_education`, `early_grade_average`, `grade_change_g1_to_g3`, and `final_grade_band`.
8. Saves a cleaned CSV and a JSON report with before/after checks.

Cleaning findings are generated from the downloaded data when the script runs; this README does not assume a particular missing-value or duplicate count.

## Setup (Windows PowerShell)

Use Python 3.10 or newer.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/preprocess.py
```

If PowerShell blocks activation in the current terminal session:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Output Files

After a successful run, check:

- `data/processed/cleaned_dataset.csv`
- `reports/preprocessing_report.json`

The script prints row counts, column counts, missing-cell and duplicate validation, and the output paths. Review the generated outputs before submission.

## Feature Notes and Limitations

- A row represents a subject/course record from one of the source files. Do not assume rows from different files represent different people.
- The grade-band feature is a descriptive category derived from final grade `G3`; category boundaries are documented in the script.
- The UCI documentation notes that `G1` and `G2` are strongly correlated with final grade `G3`. If using this dataset for prediction, consider target leakage and the intended prediction time.
- Engineered associations are descriptive; this preprocessing workflow does not establish causation.

## Submission Checklist

- [ ] Script runs successfully on your laptop
- [ ] At least 1,000 records verified in the generated report
- [ ] Cleaned CSV generated and inspected
- [ ] Report values checked against the actual run
- [ ] Screenshots captured from the real workflow
- [ ] Files pushed to GitHub
- [ ] Live dashboard checked
- [ ] Internship portal submission completed

## Internship

Created for the YR NOVATECH AI & ML Internship — Task 2.
