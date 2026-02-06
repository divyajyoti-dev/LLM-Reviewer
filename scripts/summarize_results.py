import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import List, Tuple


def _read_jsonl(path: Path) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _fmt_stats(values: List[float]) -> str:
    if not values:
        return "mean=n/a median=n/a"
    return f"mean={mean(values):.4f} median={median(values):.4f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize reviewer_sim results JSONL.")
    parser.add_argument("path", type=Path, help="Path to results JSONL")
    args = parser.parse_args()

    rows = _read_jsonl(args.path)
    tfidf_vals: List[float] = []
    jaccard_vals: List[float] = []
    score_diff_non_null = 0
    total = 0
    scored_examples: List[Tuple[float, str]] = []

    for row in rows:
        total += 1
        metrics = row.get("metrics", {}) or {}
        tfidf = metrics.get("tfidf_cosine", 0.0)
        jaccard = metrics.get("keyword_jaccard", 0.0)
        score_diff = metrics.get("score_abs_diff", None)
        paper_id = row.get("paper_id", "unknown")

        if tfidf is not None:
            tfidf_vals.append(float(tfidf))
            scored_examples.append((float(tfidf), str(paper_id)))
        if jaccard is not None:
            jaccard_vals.append(float(jaccard))
        if score_diff is not None:
            score_diff_non_null += 1

    percent_non_null = (score_diff_non_null / total * 100) if total else 0.0

    print("tfidf_cosine:", _fmt_stats(tfidf_vals))
    print("keyword_jaccard:", _fmt_stats(jaccard_vals))
    print(f"score_abs_diff non-null: {percent_non_null:.2f}% ({score_diff_non_null}/{total})")

    scored_examples.sort(key=lambda x: x[0])
    bottom = scored_examples[:3]
    top = scored_examples[-3:][::-1]

    print("top_3_tfidf_cosine:")
    for score, pid in top:
        print(f"{pid}\t{score:.4f}")

    print("bottom_3_tfidf_cosine:")
    for score, pid in bottom:
        print(f"{pid}\t{score:.4f}")


if __name__ == "__main__":
    main()
