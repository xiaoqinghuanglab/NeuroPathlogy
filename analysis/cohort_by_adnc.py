"""
Cohort characteristics stratified by ADNC classification group
(Not/Low vs Intermediate/High)

Usage:
    python3 cohort_by_adnc.py file.xlsx
"""

import sys
import pandas as pd
import numpy as np
from scipy import stats

NACCDAGE_SENTINELS = [888, 999]
NPPMIH_SENTINELS = [99.9, -4]
NPWBRWT_SENTINELS = [9999, -4]


def clean(series, sentinels):
    return series[~series.isin(sentinels)].dropna()


def group_stats(df, group_col, value_col, sentinels, label, is_categorical=False):
    g0 = df[df[group_col] == 0]
    g1 = df[df[group_col] == 1]

    print(f"\n--- {label} ---")

    if is_categorical:
        print("Not/Low group:")
        print(g0[value_col].value_counts().to_string())
        print("Intermediate/High group:")
        print(g1[value_col].value_counts().to_string())
        # chi-square test
        try:
            ct = pd.crosstab(df[group_col], df[value_col])
            chi2, p, _, _ = stats.chi2_contingency(ct)
            print(f"Chi-square p-value: {p:.4f}")
        except Exception as e:
            print(f"Chi-square test failed: {e}")
    else:
        s0 = clean(g0[value_col], sentinels)
        s1 = clean(g1[value_col], sentinels)
        print(f"Not/Low group:          n={len(s0)}, mean={s0.mean():.2f}, SD={s0.std():.2f}, median={s0.median():.2f}, range={s0.min():.2f}-{s0.max():.2f}")
        print(f"Intermediate/High group: n={len(s1)}, mean={s1.mean():.2f}, SD={s1.std():.2f}, median={s1.median():.2f}, range={s1.min():.2f}-{s1.max():.2f}")
        try:
            t_stat, p_val = stats.ttest_ind(s0, s1, equal_var=False)
            print(f"Welch t-test p-value: {p_val:.4f}")
        except Exception as e:
            print(f"t-test failed: {e}")


def main(path):
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path, sheet_name="Raw Values")
    else:
        df = pd.read_csv(path)

    print(f"Loaded {path}")
    print(f"Total rows: {len(df)}")

    if "NPADNC" not in df.columns:
        print("ERROR: NPADNC column not found. Cannot stratify without it.")
        return

    # Collapse NPADNC (0-3) to binary Not/Low (0) vs Intermediate/High (1)
    df["ADNC_binary"] = df["NPADNC"].apply(
        lambda x: 0 if x in [0, 1] else (1 if x in [2, 3] else np.nan)
    )
    print("\nADNC group sizes:")
    print(df["ADNC_binary"].value_counts(dropna=False).to_string())
    print("(0 = Not/Low, 1 = Intermediate/High, NaN = unscorable/missing NPADNC)")

    df_valid = df.dropna(subset=["ADNC_binary"])

    if "NPSEX" in df_valid.columns:
        group_stats(df_valid, "ADNC_binary", "NPSEX", None, "NPSEX (sex)", is_categorical=True)

    if "NACCDAGE" in df_valid.columns:
        group_stats(df_valid, "ADNC_binary", "NACCDAGE", NACCDAGE_SENTINELS, "NACCDAGE (age at death)")

    if "NPPMIH" in df_valid.columns:
        group_stats(df_valid, "ADNC_binary", "NPPMIH", NPPMIH_SENTINELS, "NPPMIH (postmortem interval, hours)")

    if "NPWBRWT" in df_valid.columns:
        group_stats(df_valid, "ADNC_binary", "NPWBRWT", NPWBRWT_SENTINELS, "NPWBRWT (whole brain weight, grams)")

    print("\nDone.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 cohort_by_adnc.py file.xlsx")
        sys.exit(1)
    main(sys.argv[1])