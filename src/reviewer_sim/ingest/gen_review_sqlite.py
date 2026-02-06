import argparse
import hashlib
import json
import random
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def export_gen_review_jsonl(
    db_path: Path,
    out_path: Path,
    n_papers: int = 15,
    seed: int = 42,
    include_genai: bool = False,
) -> None:
    """
    Export a sample of papers + (human) reviews from the Gen-Review SQLite database into JSONL.

    Output schema per line:
    {
      "paper_id": ...,
      "title": ...,
      "abstract": ...,
      "metadata": {...},
      "reviewer_profile": {"expertise": "...", "tone": "...", "seniority": "..."},
      "human_review": {"text": "...", "rating": ..., "confidence": ...},
      "genai_reviews": [ {"type": "...", "text": "...", "rating": ...}, ... ]   # optional
    }
    """
    random.seed(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # 1) get candidate paper ids that have at least one human review with main_review text
    cur.execute("""
        SELECT DISTINCT paper_id
        FROM REVIEW
        WHERE main_review IS NOT NULL AND TRIM(main_review) != ''
    """)
    paper_ids = [row[0] for row in cur.fetchall()]
    if not paper_ids:
        raise RuntimeError("No paper_ids found with non-empty main_review in REVIEW table.")

    # sample
    sampled = random.sample(paper_ids, k=min(n_papers, len(paper_ids)))

    def deterministic_choice(key: str, options: List[str]) -> str:
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        idx = int(digest, 16) % len(options)
        return options[idx]

    # helper: fetch submission row
    def get_submission(paper_id: str) -> Optional[Tuple]:
        cur.execute("""
            SELECT id, paper_number, title, abstract, tldr, primary_area, keywords, decision, when_submitted, source_id
            FROM SUBMISSION
            WHERE id = ?
        """, (paper_id,))
        return cur.fetchone()

    # helper: fetch one human review (you can later export all)
    def get_one_human_review(paper_id: str) -> Optional[Dict]:
        cur.execute("""
            SELECT reviewer_id, summary, strength, weaknesses, questions, rating, confidence, main_review, binocular_score
            FROM REVIEW
            WHERE paper_id = ?
              AND main_review IS NOT NULL AND TRIM(main_review) != ''
            LIMIT 1
        """, (paper_id,))
        r = cur.fetchone()
        if not r:
            return None
        reviewer_id, summary, strengths, weaknesses, questions, rating, confidence, main_review, binocular_score = r
        return {
            "reviewer_id": reviewer_id,
            "summary": summary,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "questions": questions,
            "rating": rating,
            "confidence": confidence,
            "text": main_review,
            "binocular_score": binocular_score,
        }

    # helper: fetch genai reviews
    def get_genai_reviews(paper_id: str) -> List[Dict]:
        cur.execute("""
            SELECT type, generated, rating, binocular_score
            FROM GENAI_REVIEW
            WHERE paper_id = ?
              AND generated IS NOT NULL AND TRIM(generated) != ''
        """, (paper_id,))
        out = []
        for t, generated, rating, binocular_score in cur.fetchall():
            out.append({
                "type": t,
                "text": generated,
                "rating": rating,
                "binocular_score": binocular_score,
            })
        return out

    rows_written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for pid in sampled:
            sub = get_submission(pid)
            if not sub:
                continue

            human = get_one_human_review(pid)
            if not human:
                continue

            (
                sid, paper_number, title, abstract, tldr, primary_area,
                keywords, decision, when_submitted, source_id
            ) = sub

            expertise = primary_area or "general"
            tone = deterministic_choice(f"{sid}:tone", ["neutral", "critical", "positive"])
            seniority = deterministic_choice(f"{sid}:seniority", ["junior", "senior"])

            record = {
                "paper_id": sid,
                "title": title,
                "abstract": abstract,
                "metadata": {
                    "paper_number": paper_number,
                    "tldr": tldr,
                    "primary_area": primary_area,
                    "keywords": keywords,
                    "decision": decision,
                    "when_submitted": when_submitted,
                    "source_id": source_id,
                },
                "reviewer_profile": {
                    "expertise": expertise,
                    "tone": tone,
                    "seniority": seniority,
                },
                "human_review": human,
            }

            if include_genai:
                record["genai_reviews"] = get_genai_reviews(pid)

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            rows_written += 1

    conn.close()
    print(f"Wrote {rows_written} examples to: {out_path}")


def default_paths() -> Tuple[Path, Path]:
    repo_root = Path(__file__).resolve().parents[3]  # src/reviewer_sim/ingest/ -> repo root
    db_path = repo_root / "data" / "gen_review.db"
    out_path = repo_root / "data" / "processed" / "gen_review_sample.jsonl"
    return db_path, out_path


if __name__ == "__main__":
    db_path_default, out_path_default = default_paths()

    parser = argparse.ArgumentParser(description="Export Gen-Review SQLite to JSONL.")
    parser.add_argument("--db-path", type=Path, default=db_path_default)
    parser.add_argument("--out-path", type=Path, default=out_path_default)
    parser.add_argument("--n-papers", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-genai", action="store_true")
    args = parser.parse_args()

    export_gen_review_jsonl(
        db_path=args.db_path,
        out_path=args.out_path,
        n_papers=args.n_papers,
        seed=args.seed,
        include_genai=args.include_genai,
    )
