import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "gen_review.db"

def main() -> None:
    print("DB path:", DB_PATH)
    print("Exists:", DB_PATH.exists())

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # List tables
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [t[0] for t in cur.fetchall()]
    print("\nTables:")
    for t in tables:
        print(" -", t)

    # For each table, print a few columns (schema peek)
    print("\nSchema preview:")
    for t in tables:
        cur.execute(f"PRAGMA table_info({t});")
        cols = cur.fetchall()
        col_names = [c[1] for c in cols]
        print(f"\n{t} ({len(col_names)} cols):")
        print("  ", ", ".join(col_names[:20]), ("..." if len(col_names) > 20 else ""))

    conn.close()

if __name__ == "__main__":
    main()
