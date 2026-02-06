"""
Tests for the export_review_subset module.

Run with: pytest tests/test_export_review_subset.py -v
"""

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from reviewer_sim.ingest.export_review_subset import export_review_subset, DropStats


def create_test_db(db_path: Path) -> None:
    """Create a minimal test database with SUBMISSION and REVIEW tables."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables matching the real schema
    cursor.execute("""
        CREATE TABLE SUBMISSION (
            id TEXT PRIMARY KEY,
            paper_number INTEGER,
            title TEXT,
            abstract TEXT,
            tldr TEXT,
            primary_area TEXT,
            code_of_ethics TEXT,
            pdf TEXT,
            keywords TEXT,
            decision TEXT,
            when_submitted INTEGER,
            source_id TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE REVIEW (
            paper_id TEXT,
            reviewer_id TEXT,
            summary TEXT,
            soundness TEXT,
            presentation TEXT,
            contribution TEXT,
            strength TEXT,
            weaknesses TEXT,
            questions TEXT,
            flag_for_ethics_review TEXT,
            rating TEXT,
            confidence TEXT,
            correctness TEXT,
            technical_novelty_and_significance TEXT,
            empirical_novelty_and_significance TEXT,
            main_review TEXT,
            summary_of_the_review TEXT,
            binocular_score REAL,
            PRIMARY KEY (paper_id, reviewer_id)
        )
    """)

    # Insert test data
    submissions = [
        ("paper_2020_a", 1, "Old Paper", "Abstract A", 2020, "ML"),
        ("paper_2021_b", 2, "Paper 2021 B", "Abstract B", 2021, "NLP"),
        ("paper_2022_c", 3, "Paper 2022 C", "Abstract C", 2022, "CV"),
        ("paper_empty_review", 4, "Paper Empty", "Abstract Empty", 2021, "ML"),
    ]
    for pid, num, title, abstract, year, area in submissions:
        cursor.execute(
            "INSERT INTO SUBMISSION (id, paper_number, title, abstract, when_submitted, primary_area) VALUES (?, ?, ?, ?, ?, ?)",
            (pid, num, title, abstract, year, area),
        )

    reviews = [
        ("paper_2020_a", "r1", "This is a valid review for the old paper with enough characters to pass the filter."),
        ("paper_2021_b", "r2", "This is a valid review for paper B from 2021 with sufficient length for testing."),
        ("paper_2022_c", "r3", "This is a valid review for paper C from 2022 with enough content to be exported."),
        ("paper_empty_review", "r4", ""),  # Empty review - should be dropped
    ]
    for pid, rid, review in reviews:
        cursor.execute(
            "INSERT INTO REVIEW (paper_id, reviewer_id, main_review) VALUES (?, ?, ?)",
            (pid, rid, review),
        )

    conn.commit()
    conn.close()


class TestExportReviewSubset:
    """Test suite for export_review_subset functionality."""

    def test_filters_old_papers_by_min_year(self, tmp_path: Path) -> None:
        """Papers before min_year should be dropped."""
        db_path = tmp_path / "test.db"
        out_path = tmp_path / "output.jsonl"
        create_test_db(db_path)

        stats = export_review_subset(
            db_path=db_path,
            out_path=out_path,
            n=10,
            seed=42,
            min_year=2021,
            min_review_chars=10,
        )

        # paper_2020_a should be dropped (year 2020 < 2021)
        # paper_empty_review should be dropped (empty review, filtered by SQL)
        # paper_2021_b and paper_2022_c should remain
        assert stats.pre_min_year == 1  # paper_2020_a
        assert stats.kept == 2  # paper_2021_b and paper_2022_c

        # Verify output
        with open(out_path) as f:
            records = [json.loads(line) for line in f]

        assert len(records) == 2
        paper_ids = {r["paper_id"] for r in records}
        assert "paper_2020_a" not in paper_ids
        assert "paper_2021_b" in paper_ids or "paper_2022_c" in paper_ids

    def test_deterministic_sampling(self, tmp_path: Path) -> None:
        """Same seed should produce identical output."""
        db_path = tmp_path / "test.db"
        out_path_1 = tmp_path / "output1.jsonl"
        out_path_2 = tmp_path / "output2.jsonl"
        create_test_db(db_path)

        # Run twice with same seed
        export_review_subset(
            db_path=db_path,
            out_path=out_path_1,
            n=1,
            seed=42,
            min_year=2021,
            min_review_chars=10,
        )

        export_review_subset(
            db_path=db_path,
            out_path=out_path_2,
            n=1,
            seed=42,
            min_year=2021,
            min_review_chars=10,
        )

        # Compare outputs (ignoring exported_at timestamp)
        with open(out_path_1) as f:
            rec1 = json.loads(f.readline())
        with open(out_path_2) as f:
            rec2 = json.loads(f.readline())

        assert rec1["paper_id"] == rec2["paper_id"]
        assert rec1["title"] == rec2["title"]
        assert rec1["review"] == rec2["review"]

    def test_drops_empty_reviews(self, tmp_path: Path) -> None:
        """Empty reviews should be filtered by SQL query."""
        db_path = tmp_path / "test.db"
        out_path = tmp_path / "output.jsonl"
        create_test_db(db_path)

        stats = export_review_subset(
            db_path=db_path,
            out_path=out_path,
            n=10,
            seed=42,
            min_year=None,  # No year filter
            min_review_chars=10,
        )

        # paper_empty_review has empty main_review, filtered by SQL
        with open(out_path) as f:
            records = [json.loads(line) for line in f]

        paper_ids = {r["paper_id"] for r in records}
        assert "paper_empty_review" not in paper_ids

    def test_filters_short_reviews(self, tmp_path: Path) -> None:
        """Reviews shorter than min_review_chars should be dropped."""
        db_path = tmp_path / "test.db"
        out_path = tmp_path / "output.jsonl"
        create_test_db(db_path)

        stats = export_review_subset(
            db_path=db_path,
            out_path=out_path,
            n=10,
            seed=42,
            min_year=None,
            min_review_chars=1000,  # Very high threshold
        )

        # All reviews are < 1000 chars, so all should be dropped
        assert stats.too_short >= 3
        assert stats.kept == 0

    def test_output_schema(self, tmp_path: Path) -> None:
        """Verify output JSONL matches expected schema."""
        db_path = tmp_path / "test.db"
        out_path = tmp_path / "output.jsonl"
        create_test_db(db_path)

        export_review_subset(
            db_path=db_path,
            out_path=out_path,
            n=1,
            seed=42,
            min_year=2021,
            min_review_chars=10,
        )

        with open(out_path) as f:
            record = json.loads(f.readline())

        # Verify required fields
        assert "paper_id" in record
        assert "title" in record
        assert "abstract" in record
        assert "primary_area" in record
        assert "year" in record
        assert "review" in record
        assert "main_review" in record["review"]
        assert "meta" in record
        assert record["meta"]["source"] == "gen_review_sqlite"
        assert "filters" in record["meta"]
        assert "exported_at" in record["meta"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
