# Data Collection and Preprocessing

**Internship:** YR NOVATECH AI & ML Internship — Task 2  
**Status:** Project setup started.

This project demonstrates a reproducible data preparation workflow using a public dataset with at least 1,000 records.

## Objectives

- Collect a suitable public dataset and document its source and license/usage terms.
- Inspect the dataset structure, column types, missing values, duplicate records, and basic statistics.
- Clean data with documented, justified decisions.
- Engineer useful features when appropriate.
- Save the cleaned dataset and a reproducible notebook or Python script.
- Capture evidence of the workflow and results.

## Planned Deliverables

- `notebooks/data_preprocessing.ipynb` — walkthrough of the analysis and preprocessing
- `data/raw/` — original downloaded dataset (only if redistribution terms allow it; otherwise follow the source's download instructions)
- `data/processed/cleaned_dataset.csv` — cleaned output
- `src/preprocess.py` — reproducible preprocessing script
- `reports/` — preprocessing summary and validation results
- `screenshots/` — evidence for the internship portal

## Workflow

1. Identify and cite the public dataset source.
2. Load the dataset and verify it has at least 1,000 samples.
3. Explore data types, summary statistics, missing values, and duplicates.
4. Handle data quality issues using documented rules.
5. Engineer and validate appropriate features.
6. Export the cleaned data and report the before/after checks.

## Reproducibility

Python 3.10+ is recommended. Create a virtual environment and install dependencies from `requirements.txt`.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python src/preprocess.py
```

## Data Integrity

Dataset source, data dictionary, license/terms, row counts, and all cleaning decisions will be recorded. No results or data-quality findings should be claimed until they have been measured from the chosen dataset.

## Submission Checklist

- [ ] Public dataset source documented
- [ ] At least 1,000 samples verified
- [ ] Raw data handling respects source license/terms
- [ ] Cleaning and feature engineering documented
- [ ] Notebook/script runs successfully
- [ ] Cleaned dataset produced
- [ ] Screenshots captured
- [ ] README and final outputs reviewed
