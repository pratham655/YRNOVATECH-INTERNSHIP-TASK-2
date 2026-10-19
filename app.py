
import json
from pathlib import Path

import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "processed" / "cleaned_dataset.csv"
REPORT_FILE = BASE_DIR / "reports" / "preprocessing_report.json"

st.set_page_config(
    page_title="Student Performance Data Explorer",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Student Performance Data Explorer")
st.write(
    "An interactive dashboard for exploring the cleaned UCI "
    "Student Performance dataset and its preprocessing results."
)

if not DATA_FILE.exists():
    st.error("Cleaned dataset not found. Please check the repository files.")
    st.stop()

df = pd.read_csv(DATA_FILE)

st.subheader("Dataset Overview")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Records", f"{len(df):,}")
col2.metric("Total Features", len(df.columns))
col3.metric("Missing Cells", int(df.isna().sum().sum()))
col4.metric("Duplicate Rows", int(df.duplicated().sum()))

if REPORT_FILE.exists():
    with open(REPORT_FILE, encoding="utf-8") as file:
        report = json.load(file)

    with st.expander("Preprocessing Report"):
        st.json(report)

st.subheader("Explore the Cleaned Dataset")

search = st.text_input("Search column names")
columns = [
    column for column in df.columns
    if search.lower() in column.lower()
]
selected_columns = st.multiselect(
    "Choose columns to display",
    options=columns,
    default=columns[:min(6, len(columns))],
)

if selected_columns:
    st.dataframe(df[selected_columns], use_container_width=True)
else:
    st.info("Select at least one column to display.")

st.subheader("Filter Records")

if "course" in df.columns:
    courses = sorted(df["course"].dropna().unique().tolist())
    selected_courses = st.multiselect(
        "Select course",
        courses,
        default=courses,
    )
    filtered_df = df[df["course"].isin(selected_courses)]
else:
    filtered_df = df.copy()

st.write(f"Showing **{len(filtered_df):,} records**")
st.dataframe(filtered_df.head(100), use_container_width=True)

st.subheader("Final Grade Distribution")

if "G3" in filtered_df.columns:
    grade_counts = (
        filtered_df["G3"]
        .value_counts()
        .sort_index()
    )
    st.bar_chart(grade_counts)

st.subheader("Course Record Counts")

if "course" in df.columns:
    st.bar_chart(df["course"].value_counts())

st.subheader("Download Data")

st.download_button(
    label="Download cleaned dataset (CSV)",
    data=df.to_csv(index=False).encode("utf-8"),
    file_name="cleaned_dataset.csv",
    mime="text/csv",
)

st.caption("Dataset source: UCI Machine Learning Repository — Student Performance.")
