"""
Cohort summary statistics for the Results section (basic sample description).

Usage:
    python3 cohort_summary.py file.xlsx
"""

import sys
import pandas as pd
import numpy as np

def describe_numeric(series, name, sentinel_values):
    s = series.copy()
    n_total = len(s)
    n_null = s.isna().sum()

    sentinel_mask = s.isin(sentinel_values)
    n_sentinel = sentinel_mask.sum()

    s_clean = s[~sentinel_mask].dropna()

    print(f"\n--- {name} ---")
    print(f"n total rows:        {n_total}")
    print(f"n blank/NaN:         {n_null}")
    print(f"n sentinel-coded (in {sentinel_values}): {n_sentinel}")
    if n_sentinel > 0:
        print(f"  sentinel value counts: {s[sentinel_mask].value_counts().to_dict()}")
    print(f"n used for stats:    {len(s_clean)}")
    if len(s_clean) > 0:
        print(f"mean:   {s_clean.mean():.2f}")
        print(f"SD:     {s_clean.std():.2f}")
        print(f"median: {s_clean.median():.2f}")
        print(f"min:    {s_clean.min():.2f}")
        print(f"max:    {s_clean.max():.2f}")
    else:
        print("No valid numeric values found, check column name/codes.")


def describe_categorical(series, name):
    print(f"\n--- {name} ---")
    print("Raw value counts (check against your codebook for what each code means):")
    print(series.value_counts(dropna=False).to_string())


def main(path):
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, sheet_name="Raw Values")
    else:
        df = pd.read_csv(path)
    print(f"Loaded {path} (sheet: Raw Values)")
    print(f"Total rows (should be 161): {len(df)}")
    print(f"Columns found: {list(df.columns)}")

    if "NPSEX" in df.columns:
        describe_categorical(df["NPSEX"], "NPSEX (sex)")
    else:
        print("\nWARNING: NPSEX column not found.")

    if "NACCDAGE" in df.columns:
        # 15-120 valid; 888 = Not applicable; 999 = Unknown
        describe_numeric(df["NACCDAGE"], "NACCDAGE (age at death, years)", sentinel_values=[888, 999])
    else:
        print("\nWARNING: NACCDAGE column not found.")

    if "NPPMIH" in df.columns:
        # 0.0-98.9 valid; 99.9 = Unknown; -4 = Not available
        describe_numeric(df["NPPMIH"], "NPPMIH (postmortem interval, hours)", sentinel_values=[99.9, -4])
    else:
        print("\nWARNING: NPPMIH column not found.")

    if "NPWBRWT" in df.columns:
        # 100-2500 valid; 9999 = unknown; -4 = Not available
        describe_numeric(df["NPWBRWT"], "NPWBRWT (whole brain weight, grams)", sentinel_values=[9999, -4])
    else:
        print("\nWARNING: NPWBRWT column not found.")

    print("\nDone.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 cohort_summary.py file.xlsx")
        sys.exit(1)
    main(sys.argv[1])