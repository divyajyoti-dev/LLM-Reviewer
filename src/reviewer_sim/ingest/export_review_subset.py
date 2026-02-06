"""
Export a clean subset of papers with reviews from the Gen-Review SQLite database.

Example usage:
    python -m reviewer_sim.ingest.export_review_subset \\
        --db-path data/gen_review.db \\
        --out-path outputs/review_subset.jsonl \\
        --n 200 \\
        --seed 42 \\
        --min-year 2021 \\
        --min-review-chars 50

Filters applied:
    - Only papers with non-empty main_review
    - Only papers from >= min-year (based on when_submitted or detected year column)
    - Only reviews with >= min-review-chars after whitespace normalization
    - Drops rows with missing paper_id

Output schema per line:
    {
        "paper_id": "<str>",
        "title": "<str>",
        "abstract": "<str>",
        "primary_area": "<str>",
        "year": <int or null>,
        "review": {"main_review": "<str>"},
        "meta": {
            "source": "gen_review_sqlite",
            "filters": {"min_year": ..., "min_review_chars": ...},
            "exported_at": "<ISO8601>"
        }
    }
"""

import argparse
import json
import random
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Optional


# Priority order for year/date columns
YEAR_COLUMN_PRIORITY = [
    ("year", "int"),
    ("submission_year", "int"),
    ("when_submitted", "int"),  # In this DB, it's actually a year int
    ("created_at", "timestamp"),
    ("submitted_at", "timestamp"),
    ("decision_date", "timestamp"),
]


@dataclass
class DropStats:
    null_or_empty_review: int = 0
    pre_min_year: int = 0
    missing_paper_id: int = 0
    too_short: int = 0
    bad_date: int = 0
    total_candidates: int = 0
    kept: int = 0
    review_lengths: list = field(default_factory=list)
    years: list = field(default_factory=list)


def detect_year_column(cursor: sqlite3.Cursor) -> Optional[tuple[str, str]]:
    """Detect which year/date column exists in SUBMISSION table."""
    cursor.execute("PRAGMA table_info(SUBMISSION)")
    columns = {row[1].lower() for row in cursor.fetchall()}

    for col_name, col_type in YEAR_COLUMN_PRIORITY:
        if col_name.lower() in columns:
            return (col_name, col_type)
    return None


def parse_year_from_value(value, col_type: str) -> Optional[int]:
    """Parse year from column value based on column type."""
    if value is None:
        return None

    if col_type == "int":
        try:
            year = int(value)
            if 1900 <= year <= 2100:
                return year
        except (ValueError, TypeError):
            pass
        return None

    # Timestamp/string parsing
    if col_type == "timestamp":
        val_str = str(value)
        # Try ISO8601 or YYYY-MM-DD style
        match = re.match(r"(\d{4})", val_str)
        if match:
            year = int(match.group(1))
            if 1900 <= year <= 2100:
                return year
    return None


def normalize_text(text: Optional[str]) -> str:
    """Strip and collapse multiple whitespace/newlines."""
    if text is None:
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def export_review_subset(
    db_path: Path,
    out_path: Path,
    n: int = 200,
    seed: int = 42,
    min_year: Optional[int] = None,
    min_review_chars: int = 50,
    allow_missing_year_filter: bool = False,
    keep_bad_dates: bool = False,
) -> DropStats:
    """Export a clean subset of papers with reviews."""
    stats = DropStats()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Detect year column
    year_col_info = detect_year_column(cursor)
    year_col = None
    year_col_type = None

    if year_col_info:
        year_col, year_col_type = year_col_info
        print(f"Using year column: {year_col} (type: {year_col_type})")
    else:
        if min_year is not None and not allow_missing_year_filter:
            raise ValueError(
                "No year/date column found in SUBMISSION table. "
                "Pass --allow-missing-year-filter to export without year filtering."
            )
        print("Warning: No year column found. Year filtering disabled.")

    # Build query
    year_select = f", s.{year_col}" if year_col else ""
    query = f"""
        SELECT 
            s.id as paper_id,
            s.title,
            s.abstract,
            s.primary_area,
            r.main_review
            {year_select}
        FROM REVIEW r
        JOIN SUBMISSION s ON r.paper_id = s.id
        WHERE r.main_review IS NOT NULL 
          AND TRIM(r.main_review) != ''
    """

    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    stats.total_candidates = len(rows)
    print(f"Total joined rows (pre-filters): {stats.total_candidates}")

    # Process and filter
    candidates = []
    for row in rows:
        if year_col:
            paper_id, title, abstract, primary_area, main_review, year_val = row
        else:
            paper_id, title, abstract, primary_area, main_review = row
            year_val = None

        # Check paper_id
        if not paper_id or not str(paper_id).strip():
            stats.missing_paper_id += 1
            continue

        # Parse year
        year_int = None
        if year_col:
            year_int = parse_year_from_value(year_val, year_col_type)
            if year_int is None and year_val is not None:
                if not keep_bad_dates:
                    stats.bad_date += 1
                    continue

        # Year filter
        if min_year is not None and year_int is not None:
            if year_int < min_year:
                stats.pre_min_year += 1
                continue

        # Normalize review text
        clean_review = normalize_text(main_review)
        if len(clean_review) < min_review_chars:
            stats.too_short += 1
            continue

        # Clean other fields
        clean_title = normalize_text(title)
        clean_abstract = normalize_text(abstract)
        clean_primary_area = normalize_text(primary_area) or "general"

        candidates.append({
            "paper_id": str(paper_id).strip(),
            "title": clean_title,
            "abstract": clean_abstract,
            "primary_area": clean_primary_area,
            "year": year_int,
            "main_review": clean_review,
        })

    print(f"Candidates after filtering: {len(candidates)}")

    # Deterministic shuffle and sample
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:n]

    stats.kept = len(selected)

    # Build output
    exported_at = datetime.now(timezone.utc).isoformat()
    meta_filters = {
        "min_year": min_year,
        "min_review_chars": min_review_chars,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        for rec in selected:
            stats.review_lengths.append(len(rec["main_review"]))
            if rec["year"] is not None:
                stats.years.append(rec["year"])

            output_record = {
                "paper_id": rec["paper_id"],
                "title": rec["title"],
                "abstract": rec["abstract"],
                "primary_area": rec["primary_area"],
                "year": rec["year"],
                "review": {"main_review": rec["main_review"]},
                "meta": {
                    "source": "gen_review_sqlite",
                    "filters": meta_filters,
                    "exported_at": exported_at,
                },
            }
            f.write(json.dumps(output_record, ensure_ascii=False) + "\n")

    return stats


def print_summary(stats: DropStats) -> None:
    """Print export summary statistics."""
    print("\n" + "=" * 50)
    print("EXPORT SUMMARY")
    print("=" * 50)
    print(f"Total joined rows (pre-filters): {stats.total_candidates}")
    print(f"Kept: {stats.kept}")
    print("\nDropped by reason:")
    print(f"  - null_or_empty_review: {stats.null_or_empty_review}")
    print(f"  - pre_min_year: {stats.pre_min_year}")
    print(f"  - missing_paper_id: {stats.missing_paper_id}")
    print(f"  - too_short: {stats.too_short}")
    print(f"  - bad_date: {stats.bad_date}")

    if stats.review_lengths:
        sorted_lens = sorted(stats.review_lengths)
        p95_idx = int(len(sorted_lens) * 0.95)
        print(f"\nReview length stats (kept):")
        print(f"  - min: {min(sorted_lens)}")
        print(f"  - median: {median(sorted_lens):.0f}")
        print(f"  - p95: {sorted_lens[p95_idx]}")

    if stats.years:
        from collections import Counter
        year_counts = Counter(stats.years)
        print(f"\nYear distribution (kept):")
        print(f"  - min: {min(stats.years)}, max: {max(stats.years)}")
        print("  - top counts:")
        for year, count in year_counts.most_common(5):
            print(f"      {year}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export a clean subset of papers with reviews from Gen-Review SQLite."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/gen_review.db"),
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=Path("outputs/review_subset.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=200,
        help="Number of samples to export",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=None,
        help="Minimum year filter (e.g., 2021)",
    )
    parser.add_argument(
        "--min-review-chars",
        type=int,
        default=50,
        help="Minimum review length in characters after normalization",
    )
    parser.add_argument(
        "--allow-missing-year-filter",
        action="store_true",
        help="Allow export without year filtering if no year column exists",
    )
    parser.add_argument(
        "--keep-bad-dates",
        action="store_true",
        help="Keep rows with unparseable dates (year will be null)",
    )

    args = parser.parse_args()

    if not args.db_path.exists():
        raise FileNotFoundError(f"Database not found: {args.db_path}")

    print(f"Exporting from: {args.db_path}")
    print(f"Output to: {args.out_path}")
    print(f"Samples: {args.n}, Seed: {args.seed}")
    if args.min_year:
        print(f"Min year: {args.min_year}")
    print(f"Min review chars: {args.min_review_chars}")

    stats = export_review_subset(
        db_path=args.db_path,
        out_path=args.out_path,
        n=args.n,
        seed=args.seed,
        min_year=args.min_year,
        min_review_chars=args.min_review_chars,
        allow_missing_year_filter=args.allow_missing_year_filter,
        keep_bad_dates=args.keep_bad_dates,
    )

    print_summary(stats)
    print(f"\nExported {stats.kept} records to: {args.out_path}")


if __name__ == "__main__":
    main()
