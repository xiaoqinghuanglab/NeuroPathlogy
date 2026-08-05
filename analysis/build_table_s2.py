"""
Builds Supplementary Table S2 (variable prevalence summary) from the two
variable_matrix xlsx files, converting the "blank row = section header"
format into a fully tidy, sortable/filterable table with a Schema_Block
column instead of divider rows, plus a Source column (Primary/Residual).

Also applies light per-row color shading by Category (Always Present /
Present / Rarely Present / Never Present) so the visual grouping survives
sorting and filtering, unlike a blank-row divider.

Edit the CONFIG section below if your filenames or sheet names differ.
Run in the same folder as the two source files:

    python3 build_table_s2.py
"""

import pandas as pd
import openpyxl
from openpyxl.styles import PatternFill, Font

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PRIMARY_FILE = "results/variable_matrix_oss-20b_primary.xlsx"
RESIDUAL_FILE = "results/variable_matrix_oss-20b_residual.xlsx"
SHEET_NAME = 0  # 0 = first/active sheet; change to a sheet name string if needed
OUTPUT_FILE = "Table_S2.xlsx"

CATEGORY_COLORS = {
    "Always Present":  "C6EFCE",  # light green
    "Present":         "FFEB9C",  # light yellow
    "Rarely Present":  "FFD9B3",  # light orange
    "Never Present":   "FFC7CE",  # light red/pink
}


def parse_sheet(path, sheet_name, source_label):
    """
    Reads a variable_matrix sheet where section headers appear as a row
    with only the Variable column filled and the rest blank/NaN.
    Forward-fills that section name into every subsequent data row until
    the next header, and drops the header rows themselves from the output.
    """
    df = pd.read_excel(path, sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]

    records = []
    current_block = None

    for _, row in df.iterrows():
        variable = row.get("Variable")
        n_present = row.get("N Present")
        n_absent = row.get("N Absent")
        pct_present = row.get("% Present")
        category = row.get("Category")

        if pd.isna(variable):
            continue  # fully blank row, skip

        is_header = pd.isna(n_present) and pd.isna(n_absent) and pd.isna(pct_present) and pd.isna(category)

        if is_header:
            current_block = str(variable).strip()
            continue

        records.append({
            "Variable": variable,
            "Schema_Block": current_block,
            "Source": source_label,
            "N_Present": n_present,
            "N_Absent": n_absent,
            "Pct_Present": pct_present,
            "Category": category,
        })

    return pd.DataFrame(records)


def main():
    primary_df = parse_sheet(PRIMARY_FILE, SHEET_NAME, "Primary")
    residual_df = parse_sheet(RESIDUAL_FILE, SHEET_NAME, "Residual")

    combined = pd.concat([primary_df, residual_df], ignore_index=True)

    print(f"Primary rows parsed:  {len(primary_df)}")
    print(f"Residual rows parsed: {len(residual_df)}")
    print(f"Total combined rows:  {len(combined)}")
    print(f"Schema blocks found:  {sorted(combined['Schema_Block'].dropna().unique().tolist())}")

    # --- Write to xlsx with per-row category shading ---
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Table S2"

    headers = ["Variable", "Schema_Block", "Source", "N_Present", "N_Absent", "Pct_Present", "Category"]
    ws.append(headers)
    for cell in ws[1]:
        cell.font = Font(bold=True)

    for _, r in combined.iterrows():
        ws.append([
            r["Variable"], r["Schema_Block"], r["Source"],
            r["N_Present"], r["N_Absent"], r["Pct_Present"], r["Category"],
        ])
        row_idx = ws.max_row
        color = CATEGORY_COLORS.get(str(r["Category"]).strip())
        if color:
            fill = PatternFill(start_color=color, end_color=color, fill_type="solid")
            for col in range(1, len(headers) + 1):
                ws.cell(row=row_idx, column=col).fill = fill

    # Reasonable column widths
    widths = [34, 20, 10, 11, 10, 12, 16]
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = w

    ws.freeze_panes = "A2"  # keep header visible when scrolling 252 rows

    wb.save(OUTPUT_FILE)
    print(f"\nSaved {OUTPUT_FILE}")


if __name__ == "__main__":
    main()