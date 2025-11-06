#!/usr/bin/env python3
import argparse
import os
import sqlite3
from typing import Optional

import pandas as pd


def coerce_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert columns that are fully numeric (ignoring empty/nulls) to numeric dtypes.
    - For each object-like column, try to convert to numeric with errors='coerce'.
    - If all originally non-null values become non-null numerics after conversion, adopt it.
    - Leaves mixed or non-numeric columns untouched.
    """
    df = df.copy()
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s):
            non_null = s.notna() & (s.astype(str).str.len() > 0)
            if non_null.any():
                converted = pd.to_numeric(s, errors="coerce")
                # If every originally non-null value converted to a number (not NaN), accept conversion
                if converted[non_null].notna().all():
                    df[col] = converted
    # Let pandas refine dtypes (nullable integers/floats/strings/booleans)
    try:
        df = df.convert_dtypes()
    except Exception:
        pass
    return df


def load_csv(csv_path: str) -> pd.DataFrame:
    # Read CSV with reasonable defaults; low_memory=False to avoid mixed-type inference
    df = pd.read_csv(csv_path, low_memory=False)
    df = coerce_numeric_columns(df)
    return df


def write_sqlite(df: pd.DataFrame, db_path: str, table_name: str = "plays", if_exists: str = "replace") -> None:
    # Use sqlite3 DB-API connection; pandas will map numeric dtypes to INTEGER/REAL
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Import a CSV into SQLite as a single table named 'plays'.")
    parser.add_argument("--csv", dest="csv_path", default="team_data_combined/plays.csv", help="Path to plays.csv")
    parser.add_argument("--db", dest="db_path", default="plays.db", help="Output SQLite database path")
    parser.add_argument("--table", dest="table_name", default="plays", help="SQLite table name (default: plays)")
    args = parser.parse_args(argv)

    if not os.path.exists(args.csv_path):
        raise SystemExit(f"CSV not found: {args.csv_path}")

    print(f"Reading CSV: {args.csv_path}")
    df = load_csv(args.csv_path)

    print(f"Rows: {len(df):,} | Columns: {len(df.columns)}")
    print("Inferred dtypes:")
    # Show a compact dtype summary
    for name, dtype in df.dtypes.items():
        print(f"  - {name}: {dtype}")

    print(f"Writing SQLite DB: {args.db_path} (table: {args.table_name})")
    write_sqlite(df, args.db_path, args.table_name)

    # Quick verification: show schema and a count
    with sqlite3.connect(args.db_path) as conn:
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({args.table_name});")
        cols = cur.fetchall()
        print("\nTable schema (name, type):")
        for _, name, coltype, _, _, _ in cols:
            print(f"  - {name}: {coltype}")
        cur.execute(f"SELECT COUNT(*) FROM {args.table_name};")
        count = cur.fetchone()[0]
        print(f"\nRow count in table '{args.table_name}': {count:,}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
