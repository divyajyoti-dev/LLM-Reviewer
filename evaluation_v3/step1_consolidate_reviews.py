#!/usr/bin/env python3
"""
Step 1: Consolidate all human reviews per paper.

Reads from SQLite DB, groups ALL reviews by paper_id, and outputs
a JSONL file with one record per paper containing all merged reviews.

This ensures we use the full breadth of human reviewer feedback
rather than a single review per paper.

Usage:
    python evaluation_v3/step1_consolidate_reviews.py \
        --db-path data/gen_review.db \
        --output evaluation_v3/outputs/consolidated_reviews.jsonl \
        --n 200 --seed 42 --min-year 2021
"""

import argparse
import json
import random
import re
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Optional


def normalize_text(text: Optional[str]) -> str:
    """Strip and collapse whitespace."""
    if text is None:
        return ""
    text = str(text).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def consolidate_reviews(
    db_path: Path,
    out_path: Path,
    n: int = 200,
    seed: int = 42,
    min_year: Optional[int] = 2021,
    min_review_chars: int = 50,
) -> dict:
    """Export papers with ALL reviews consolidated per paper."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all paper-review pairs (no grouping in SQL — we group in Python
    # to avoid SQLite GROUP_CONCAT truncation on long reviews)
    query = """
        SELECT
            s.id AS paper_id,
            s.title,
            s.abstract,
            s.primary_area,
            s.when_submitted AS year,
            r.main_review
        FROM SUBMISSION s
        JOIN REVIEW r ON r.paper_id = s.id
        WHERE r.main_review IS NOT NULL
          AND TRIM(r.main_review) != ''
    """
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    print(f"Total paper-review rows from DB: {len(rows)}")

    # Group reviews by paper_id
    papers: dict[str, dict] = {}
    for paper_id, title, abstract, primary_area, year_val, review in rows:
        if not paper_id:
            continue
        pid = str(paper_id).strip()

        if pid not in papers:
            # Parse year
            year_int = None
            try:
                y = int(year_val)
                if 1900 <= y <= 2100:
                    year_int = y
            except (ValueError, TypeError):
                pass

            papers[pid] = {
                "title": normalize_text(title),
                "abstract": normalize_text(abstract),
                "primary_area": normalize_text(primary_area) or "general",
                "year": year_int,
                "reviews": [],
            }

        clean_review = normalize_text(review)
        if len(clean_review) >= min_review_chars:
            papers[pid]["reviews"].append(clean_review)

    print(f"Unique papers with valid reviews: {sum(1 for p in papers.values() if p['reviews'])}")

    # Filter by year and non-empty reviews
    candidates = []
    skipped_year = 0
    skipped_no_reviews = 0

    for pid, data in papers.items():
        if not data["reviews"]:
            skipped_no_reviews += 1
            continue
        if min_year and data["year"] and data["year"] < min_year:
            skipped_year += 1
            continue
        candidates.append((pid, data))

    print(f"Candidates after filtering: {len(candidates)}")
    print(f"  Skipped (year < {min_year}): {skipped_year}")
    print(f"  Skipped (no valid reviews): {skipped_no_reviews}")

    # Deterministic sample
    rng = random.Random(seed)
    rng.shuffle(candidates)
    selected = candidates[:n]

    # Write output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    exported_at = datetime.now(timezone.utc).isoformat()

    review_counts = []
    with open(out_path, "w", encoding="utf-8") as f:
        for pid, data in selected:
            # Combine all reviews with clear separators
            combined = "\n\n---\n\n".join(
                f"[Review {i + 1}]\n{r}" for i, r in enumerate(data["reviews"])
            )
            review_counts.append(len(data["reviews"]))

            record = {
                "paper_id": pid,
                "title": data["title"],
                "abstract": data["abstract"],
                "primary_area": data["primary_area"],
                "year": data["year"],
                "review_count": len(data["reviews"]),
                "individual_reviews": data["reviews"],
                "combined_human_review": combined,
                "meta": {
                    "source": "gen_review_sqlite",
                    "consolidated": True,
                    "min_review_chars": min_review_chars,
                    "min_year": min_year,
                    "exported_at": exported_at,
                },
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    stats = {
        "total_papers_in_db": len(papers),
        "candidates_after_filter": len(candidates),
        "selected": len(selected),
        "avg_reviews_per_paper": round(mean(review_counts), 2) if review_counts else 0,
        "median_reviews_per_paper": round(median(review_counts), 1) if review_counts else 0,
        "max_reviews_per_paper": max(review_counts) if review_counts else 0,
        "min_reviews_per_paper": min(review_counts) if review_counts else 0,
    }
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Consolidate all human reviews per paper from SQLite."
    )
    parser.add_argument(
        "--db-path", type=Path, default=Path("data/gen_review.db"),
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--output", "-o", type=Path,
        default=Path("evaluation_v3/outputs/consolidated_reviews.jsonl"),
        help="Output JSONL path",
    )
    parser.add_argument("--n", type=int, default=200, help="Number of papers to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--min-year", type=int, default=2021, help="Minimum year filter")
    parser.add_argument(
        "--min-review-chars", type=int, default=50,
        help="Minimum review length in characters",
    )
    args = parser.parse_args()

    if not args.db_path.exists():
        raise FileNotFoundError(f"Database not found: {args.db_path}")

    print("=" * 60)
    print("STEP 1: Consolidate Human Reviews")
    print("=" * 60)
    print(f"Database: {args.db_path}")
    print(f"Output: {args.output}")
    print(f"Sample size: {args.n}, Seed: {args.seed}")
    print(f"Min year: {args.min_year}, Min review chars: {args.min_review_chars}")
    print()

    stats = consolidate_reviews(
        db_path=args.db_path,
        out_path=args.output,
        n=args.n,
        seed=args.seed,
        min_year=args.min_year,
        min_review_chars=args.min_review_chars,
    )

    print()
    print("=" * 60)
    print("STEP 1 COMPLETE")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"\nOutput: {args.output}")


if __name__ == "__main__":
    main()
