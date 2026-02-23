#!/usr/bin/env python3
"""
Step 3: Semantic keyword evaluation.

For each paper:
  1. Extract top 10 keywords from consolidated human review (via LLM)
  2. Extract top 10 keywords from LLM-generated review (via LLM)
  3. Compute semantic similarity between keyword sets (via embeddings)
  4. Map sentences between human and LLM reviews
  5. Calculate coverage score: matched/10 = X%

Usage:
    export TOGETHER_API_KEY="your-key"
    python evaluation_v3/step3_semantic_evaluation.py \
        --human-reviews evaluation_v3/outputs/consolidated_reviews.jsonl \
        --llm-reviews evaluation_v3/outputs/llm_reviews.jsonl \
        --output-dir evaluation_v3/outputs
"""

import argparse
import asyncio
import json
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Optional

import httpx
import numpy as np

TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

# ── Keyword extraction prompt ──────────────────────────────────────────

KEYWORD_SYSTEM_PROMPT = """You are an expert at analyzing academic peer reviews.

Extract exactly 10 key technical keywords or phrases from the given review text.

Guidelines:
- Each keyword/phrase should be 1-5 words
- Focus on: technical methods, experimental concerns, evaluation metrics,
  theoretical concepts, specific criticisms, praised innovations
- Use standard ML/AI terminology
- Lowercase only
- Order by importance (most important first)

Output format: Return ONLY a JSON array of exactly 10 strings.
Example: ["gradient descent", "overfitting", "ablation study", "convergence rate",
          "baseline comparison", "sample efficiency", "generalization", "attention mechanism",
          "training stability", "computational cost"]

IMPORTANT: Return EXACTLY 10 keywords. No other text."""

KEYWORD_USER_PROMPT = """Extract 10 keywords from this review:

{review_text}

Return ONLY: ["keyword1", ..., "keyword10"]"""


# ── Data classes ───────────────────────────────────────────────────────

@dataclass
class KeywordMatch:
    human_keyword: str
    best_llm_keyword: str
    similarity: float
    matched: bool


@dataclass
class SentenceMapping:
    human_sentence: str
    llm_sentence: str
    similarity: float


@dataclass
class PaperCoverage:
    paper_id: str
    title: str
    model: str
    human_keywords: list[str]
    llm_keywords: list[str]
    keyword_matches: list[dict]
    missed_keywords: list[str]
    coverage_score: float
    matched_count: int
    sentence_mappings: list[dict]


# ── Keyword extraction via LLM ────────────────────────────────────────

def parse_keywords_response(text: str) -> list[str]:
    """Parse keyword list from LLM response."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"```(?:json)?\n?", "", text).strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    # Try direct parse
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [str(k).lower().strip() for k in result if k][:10]
    except json.JSONDecodeError:
        pass

    # Find array in text
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return [str(k).lower().strip() for k in result if k][:10]
        except json.JSONDecodeError:
            pass

    return []


async def extract_keywords_single(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    review_text: str,
    max_retries: int = 3,
) -> list[str]:
    """Extract top 10 keywords from a review text via LLM."""
    truncated = review_text[:4000]

    for attempt in range(max_retries):
        try:
            resp = await client.post(
                TOGETHER_API_URL,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": KEYWORD_SYSTEM_PROMPT},
                        {"role": "user", "content": KEYWORD_USER_PROMPT.format(
                            review_text=truncated
                        )},
                    ],
                    "temperature": 0,
                    "max_tokens": 200,
                },
                timeout=60.0,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                await asyncio.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            keywords = parse_keywords_response(content)
            if keywords:
                return keywords
        except Exception:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
    return []


async def extract_keywords_batch(
    texts: list[tuple[str, str]],  # [(id, text), ...]
    api_key: str,
    model: str,
    max_concurrency: int = 5,
    label: str = "",
) -> dict[str, list[str]]:
    """Extract keywords for a batch of texts. Returns {id: [keywords]}."""
    sem = asyncio.Semaphore(max_concurrency)
    results: dict[str, list[str]] = {}

    async def bounded(client, item_id, text):
        async with sem:
            kws = await extract_keywords_single(client, api_key, model, text)
            return item_id, kws

    async with httpx.AsyncClient() as client:
        tasks = [bounded(client, item_id, text) for item_id, text in texts]
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            item_id, kws = await coro
            results[item_id] = kws
            if (i + 1) % 20 == 0 or (i + 1) == len(texts):
                print(f"  {label} {i + 1}/{len(texts)}")

    return results


# ── Semantic matching via embeddings ───────────────────────────────────

def load_embedder():
    """Load sentence-transformers model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("ERROR: sentence-transformers is required.")
        print("Install with: pip install sentence-transformers")
        sys.exit(1)
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    return SentenceTransformer("all-MiniLM-L6-v2")


def cosine_sim_matrix(embeddings_a: np.ndarray, embeddings_b: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix between two sets of embeddings."""
    # Normalize
    a_norm = embeddings_a / (np.linalg.norm(embeddings_a, axis=1, keepdims=True) + 1e-9)
    b_norm = embeddings_b / (np.linalg.norm(embeddings_b, axis=1, keepdims=True) + 1e-9)
    return a_norm @ b_norm.T


def compute_keyword_coverage(
    human_keywords: list[str],
    llm_keywords: list[str],
    embedder,
    threshold: float = 0.55,
) -> tuple[list[KeywordMatch], list[str], int]:
    """
    Match human keywords against LLM keywords semantically.
    Returns (matches, missed_keywords, matched_count).
    """
    if not human_keywords or not llm_keywords:
        return [], human_keywords, 0

    h_emb = embedder.encode(human_keywords)
    l_emb = embedder.encode(llm_keywords)
    sim_matrix = cosine_sim_matrix(h_emb, l_emb)

    matches = []
    missed = []
    matched_count = 0

    for i, hk in enumerate(human_keywords):
        best_j = int(np.argmax(sim_matrix[i]))
        best_sim = float(sim_matrix[i, best_j])
        is_match = best_sim >= threshold

        matches.append(KeywordMatch(
            human_keyword=hk,
            best_llm_keyword=llm_keywords[best_j],
            similarity=round(best_sim, 4),
            matched=is_match,
        ))

        if is_match:
            matched_count += 1
        else:
            missed.append(hk)

    return matches, missed, matched_count


# ── Sentence mapping ──────────────────────────────────────────────────

def split_sentences(text: str) -> list[str]:
    """Split review text into sentences."""
    # Normalize numbered/bulleted items
    text = re.sub(r"\n\s*(\d+[.)]\s)", r". \1", text)
    text = re.sub(r"\n\s*[-•]\s", r". ", text)
    text = re.sub(r"\n+", " ", text)

    sentences = re.split(r"(?<=[.!?])\s+", text)
    # Keep only substantial sentences
    sentences = [s.strip() for s in sentences if len(s.strip()) > 30]
    return sentences


def compute_sentence_mappings(
    human_text: str,
    llm_text: str,
    embedder,
    threshold: float = 0.50,
    top_k: int = 10,
) -> list[SentenceMapping]:
    """Find sentence-level correspondences between human and LLM reviews."""
    h_sents = split_sentences(human_text)
    l_sents = split_sentences(llm_text)

    if not h_sents or not l_sents:
        return []

    h_emb = embedder.encode(h_sents)
    l_emb = embedder.encode(l_sents)
    sim_matrix = cosine_sim_matrix(h_emb, l_emb)

    mappings = []
    for i, hs in enumerate(h_sents):
        best_j = int(np.argmax(sim_matrix[i]))
        best_sim = float(sim_matrix[i, best_j])
        if best_sim >= threshold:
            mappings.append(SentenceMapping(
                human_sentence=hs,
                llm_sentence=l_sents[best_j],
                similarity=round(best_sim, 4),
            ))

    # Return top-k by similarity
    mappings.sort(key=lambda m: m.similarity, reverse=True)
    return mappings[:top_k]


# ── Report generation ─────────────────────────────────────────────────

def build_llm_review_text(review: dict) -> str:
    """Combine LLM review fields into a single text block."""
    parts = [review.get("summary", "")]
    for s in review.get("strengths", []):
        parts.append(f"Strength: {s}")
    for w in review.get("weaknesses", []):
        parts.append(f"Weakness: {w}")
    for q in review.get("questions", []):
        parts.append(f"Question: {q}")
    raw = review.get("raw_review", "")
    if raw:
        parts.append(raw)
    return " ".join(p for p in parts if p)


def write_report(results: list[PaperCoverage], output_dir: Path, threshold: float):
    """Write markdown coverage report."""
    report_path = output_dir / "coverage_report.md"

    by_model = defaultdict(list)
    for r in results:
        by_model[r.model].append(r)

    models = sorted(by_model.keys())
    all_coverage = [r.coverage_score for r in results]

    with open(report_path, "w") as f:
        f.write("# Semantic Coverage Analysis Report\n\n")
        f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n")

        f.write("## Methodology\n\n")
        f.write("- **Human reviews**: All reviews per paper consolidated into one\n")
        f.write("- **Keywords**: Top 10 extracted from each review using LLM\n")
        f.write("- **Matching**: Sentence-transformers (`all-MiniLM-L6-v2`) cosine similarity\n")
        f.write(f"- **Match threshold**: {threshold} cosine similarity\n")
        f.write("- **Coverage**: matched_human_keywords / 10\n\n")

        f.write("## Overall Results\n\n")
        f.write(f"- Papers evaluated: {len(set(r.paper_id for r in results))}\n")
        f.write(f"- Models compared: {len(models)}\n")
        f.write(f"- Total comparisons: {len(results)}\n")
        f.write(f"- **Average coverage: {mean(all_coverage):.1%}**\n")
        f.write(f"- Median coverage: {median(all_coverage):.1%}\n\n")

        f.write("## Per-Model Coverage\n\n")
        f.write("| Model | Avg Coverage | Median | Papers |\n")
        f.write("|-------|-------------|--------|--------|\n")

        model_scores = []
        for m in models:
            scores = [r.coverage_score for r in by_model[m]]
            avg = mean(scores)
            med = median(scores)
            model_scores.append((m, avg))
            short = m.split("/")[-1][:35]
            f.write(f"| {short} | {avg:.1%} | {med:.1%} | {len(scores)} |\n")

        model_scores.sort(key=lambda x: x[1], reverse=True)

        f.write("\n## Model Rankings\n\n")
        for i, (m, score) in enumerate(model_scores, 1):
            short = m.split("/")[-1]
            f.write(f"{i}. **{short}**: {score:.1%}\n")

        # Most missed keywords
        missed_counts = defaultdict(int)
        for r in results:
            for kw in r.missed_keywords:
                missed_counts[kw] += 1

        top_missed = sorted(missed_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        if top_missed:
            f.write("\n## Most Frequently Missed Keywords\n\n")
            f.write("| Keyword | Times Missed | % of Comparisons |\n")
            f.write("|---------|-------------|------------------|\n")
            for kw, count in top_missed:
                pct = count / len(results) * 100
                f.write(f"| {kw} | {count} | {pct:.1f}% |\n")

        # Most commonly matched keywords
        matched_counts = defaultdict(list)
        for r in results:
            for km in r.keyword_matches:
                if km["matched"]:
                    matched_counts[km["human_keyword"]].append(km["similarity"])

        top_matched = sorted(
            matched_counts.items(), key=lambda x: len(x[1]), reverse=True
        )[:15]
        if top_matched:
            f.write("\n## Most Commonly Matched Keywords\n\n")
            f.write("| Human Keyword | Times Matched | Avg Similarity |\n")
            f.write("|--------------|---------------|----------------|\n")
            for kw, sims in top_matched:
                f.write(f"| {kw} | {len(sims)} | {mean(sims):.3f} |\n")

        # Example sentence mappings (from best-covered paper)
        best_paper = max(results, key=lambda r: r.coverage_score)
        if best_paper.sentence_mappings:
            f.write("\n## Example Sentence Mappings\n\n")
            f.write(f"Paper: *{best_paper.title[:80]}* ")
            f.write(f"(coverage: {best_paper.coverage_score:.0%})\n\n")
            f.write("| Human Sentence | LLM Sentence | Similarity |\n")
            f.write("|---------------|-------------|------------|\n")
            for sm in best_paper.sentence_mappings[:5]:
                hs = sm["human_sentence"][:80].replace("|", "/")
                ls = sm["llm_sentence"][:80].replace("|", "/")
                f.write(f"| {hs}... | {ls}... | {sm['similarity']:.3f} |\n")

        # Key findings
        best_model = model_scores[0]
        worst_model = model_scores[-1]
        f.write("\n## Key Findings\n\n")
        f.write(f"- LLMs cover **{mean(all_coverage):.0%}** of human reviewer keywords on average\n")
        f.write(f"- **Best model**: {best_model[0].split('/')[-1]} ({best_model[1]:.1%})\n")
        f.write(f"- **Worst model**: {worst_model[0].split('/')[-1]} ({worst_model[1]:.1%})\n")
        f.write(f"- Performance gap: {best_model[1] - worst_model[1]:.1%}\n")

    print(f"Wrote report: {report_path}")


# ── Main pipeline ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 3: Semantic keyword evaluation"
    )
    parser.add_argument(
        "--human-reviews", "-hr", required=True,
        help="JSONL with consolidated human reviews (from step 1)",
    )
    parser.add_argument(
        "--llm-reviews", "-lr", required=True,
        help="JSONL with LLM reviews (from step 2)",
    )
    parser.add_argument(
        "--output-dir", "-o", default="evaluation_v3/outputs",
        help="Output directory",
    )
    parser.add_argument(
        "--extraction-model", default="deepseek-ai/DeepSeek-V3",
        help="Model to use for keyword extraction (default: DeepSeek-V3)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.55,
        help="Cosine similarity threshold for keyword match (default: 0.55)",
    )
    parser.add_argument(
        "--max-concurrency", type=int, default=5,
        help="Max concurrent API requests for keyword extraction",
    )
    args = parser.parse_args()

    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        print("Error: TOGETHER_API_KEY required")
        sys.exit(1)

    # Load data
    print("=" * 60)
    print("STEP 3: Semantic Keyword Evaluation")
    print("=" * 60)

    human_data = {}
    with open(args.human_reviews) as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                human_data[rec["paper_id"]] = rec

    llm_reviews = []
    with open(args.llm_reviews) as f:
        for line in f:
            if line.strip():
                llm_reviews.append(json.loads(line))

    # Group LLM reviews by paper_id
    llm_by_paper = defaultdict(list)
    for r in llm_reviews:
        if r.get("paper_id") in human_data and not r.get("error"):
            llm_by_paper[r["paper_id"]].append(r)

    common_papers = set(human_data.keys()) & set(llm_by_paper.keys())
    print(f"Human reviews: {len(human_data)} papers")
    print(f"LLM reviews: {len(llm_reviews)} total")
    print(f"Common papers: {len(common_papers)}")
    print(f"Extraction model: {args.extraction_model}")
    print(f"Similarity threshold: {args.threshold}")
    print()

    # ── Phase A: Extract keywords from human reviews ───────────────────
    print("Phase A: Extracting keywords from human reviews...")
    human_texts = [
        (pid, human_data[pid]["combined_human_review"])
        for pid in common_papers
    ]

    human_keywords = asyncio.run(extract_keywords_batch(
        human_texts, api_key, args.extraction_model,
        max_concurrency=args.max_concurrency, label="Human keywords",
    ))

    ok = sum(1 for kws in human_keywords.values() if kws)
    print(f"  Extracted keywords for {ok}/{len(human_texts)} papers")
    print()

    # ── Phase B: Extract keywords from LLM reviews ─────────────────────
    print("Phase B: Extracting keywords from LLM reviews...")
    llm_texts = []
    llm_text_keys = []  # (paper_id, model)
    for pid in common_papers:
        for review in llm_by_paper[pid]:
            key = f"{pid}||{review['model']}"
            text = build_llm_review_text(review)
            llm_texts.append((key, text))
            llm_text_keys.append((pid, review["model"]))

    llm_keywords_raw = asyncio.run(extract_keywords_batch(
        llm_texts, api_key, args.extraction_model,
        max_concurrency=args.max_concurrency, label="LLM keywords",
    ))

    # Restructure: {(paper_id, model): [keywords]}
    llm_keywords = {}
    for (pid, model), key_str in zip(llm_text_keys, [f"{p}||{m}" for p, m in llm_text_keys]):
        llm_keywords[(pid, model)] = llm_keywords_raw.get(key_str, [])

    ok = sum(1 for kws in llm_keywords.values() if kws)
    print(f"  Extracted keywords for {ok}/{len(llm_texts)} reviews")
    print()

    # ── Phase C: Semantic coverage ─────────────────────────────────────
    print("Phase C: Computing semantic coverage...")
    embedder = load_embedder()

    results: list[PaperCoverage] = []
    processed = 0

    for pid in common_papers:
        hk = human_keywords.get(pid, [])
        if not hk:
            continue

        for review in llm_by_paper[pid]:
            model_name = review["model"]
            lk = llm_keywords.get((pid, model_name), [])

            # Keyword coverage
            matches, missed, matched_count = compute_keyword_coverage(
                hk, lk, embedder, threshold=args.threshold,
            )
            coverage = matched_count / max(len(hk), 1)

            # Sentence mapping
            human_text = human_data[pid]["combined_human_review"]
            llm_text = build_llm_review_text(review)
            sent_maps = compute_sentence_mappings(
                human_text, llm_text, embedder, threshold=0.50, top_k=10,
            )

            results.append(PaperCoverage(
                paper_id=pid,
                title=human_data[pid].get("title", ""),
                model=model_name,
                human_keywords=hk,
                llm_keywords=lk,
                keyword_matches=[asdict(m) for m in matches],
                missed_keywords=missed,
                coverage_score=round(coverage, 4),
                matched_count=matched_count,
                sentence_mappings=[asdict(sm) for sm in sent_maps],
            ))

        processed += 1
        if processed % 50 == 0:
            print(f"  Processed {processed}/{len(common_papers)} papers")

    print(f"  Processed {processed}/{len(common_papers)} papers total")
    print()

    # ── Phase D: Write results ─────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detailed results
    results_path = output_dir / "semantic_coverage_results.jsonl"
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")
    print(f"Wrote detailed results: {results_path}")

    # Summary JSON
    all_cov = [r.coverage_score for r in results]
    by_model = defaultdict(list)
    for r in results:
        by_model[r.model].append(r.coverage_score)

    summary = {
        "papers_evaluated": len(common_papers),
        "total_comparisons": len(results),
        "overall_mean_coverage": round(mean(all_cov), 4) if all_cov else 0,
        "overall_median_coverage": round(median(all_cov), 4) if all_cov else 0,
        "per_model": {
            m: {
                "mean_coverage": round(mean(scores), 4),
                "median_coverage": round(median(scores), 4),
                "count": len(scores),
            }
            for m, scores in sorted(by_model.items())
        },
        "threshold": args.threshold,
        "extraction_model": args.extraction_model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    summary_path = output_dir / "coverage_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote summary: {summary_path}")

    # Markdown report
    write_report(results, output_dir, args.threshold)

    # Console summary
    print()
    print("=" * 60)
    print("STEP 3 COMPLETE")
    print("=" * 60)
    print(f"Papers evaluated: {len(common_papers)}")
    print(f"Total comparisons: {len(results)}")
    print(f"Overall mean coverage: {mean(all_cov):.1%}")
    print()
    print("Per-model coverage:")
    model_ranking = sorted(by_model.items(), key=lambda x: mean(x[1]), reverse=True)
    for i, (m, scores) in enumerate(model_ranking, 1):
        short = m.split("/")[-1]
        print(f"  {i}. {short}: {mean(scores):.1%}")


if __name__ == "__main__":
    main()
