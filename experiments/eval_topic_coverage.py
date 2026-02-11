"""
Evaluate topic coverage between model-generated reviews.

Extracts topics from each review using LLM or keyword extraction,
then computes overlap/Jaccard metrics for model pairs.

Environment variables:
  TOGETHER_API_KEY: API key for LLM-based topic extraction (optional)

Example usage:
  # Using LLM topic extraction (recommended)
  export TOGETHER_API_KEY="your-key"
  python experiments/eval_topic_coverage.py \
    --input experiments/outputs/run_001/reviews.jsonl \
    --output-dir experiments/outputs/run_001 \
    --extractor llm

  # Using keyword extraction (no API needed)
  python experiments/eval_topic_coverage.py \
    --input experiments/outputs/run_001/reviews.jsonl \
    --output-dir experiments/outputs/run_001 \
    --extractor keywords

  # Compare specific model pair only
  python experiments/eval_topic_coverage.py \
    --input experiments/outputs/run_001/reviews.jsonl \
    --output-dir experiments/outputs/run_001 \
    --model-a mock \
    --model-b "mistralai/Mixtral-8x7B-Instruct-v0.1"

Output files:
  - evaluation_report.json: Consolidated report with all data
  - topic_eval.csv: Per-paper-pair metrics
  - topic_eval_summary.md: Aggregated summary with statistics
  - topics_extracted.jsonl: Raw extracted topics per review
  - charts/: Directory with visualization charts
    - jaccard_distribution.png: Histogram of Jaccard scores
    - jaccard_by_model_pair.png: Bar chart comparing model pairs
    - topic_frequency.png: Top topics by frequency
    - topics_per_review.png: Distribution of topics per review
"""

import argparse
import asyncio
import csv
import json
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from statistics import mean, median
from typing import Optional

import httpx


# LLM prompt for topic extraction
TOPIC_SYSTEM_PROMPT = """You are a research paper review analyst. Given a peer review, extract the main topics/themes discussed.

Topics should be:
- Concise (1-4 words each)
- Lowercase
- Focused on technical/methodological aspects discussed in the review
- Deduplicated and normalized

Examples of good topics:
- "experimental design", "baseline comparison", "ablation study"
- "scalability", "computational cost", "theoretical analysis"
- "novelty", "reproducibility", "writing clarity"

Respond with ONLY a JSON object:
{"topics": ["topic1", "topic2", "topic3", ...]}

Extract 5-15 topics that capture what the review discusses."""

TOPIC_USER_PROMPT = """Extract topics from this review:

{review_text}

Respond with JSON: {{"topics": [...]}}"""


# Keywords for fallback extraction (grouped by category)
KEYWORD_CATEGORIES = {
    "methodology": [
        "method", "approach", "technique", "algorithm", "framework",
        "architecture", "model", "system", "pipeline", "workflow",
    ],
    "evaluation": [
        "experiment", "evaluation", "benchmark", "baseline", "comparison",
        "ablation", "analysis", "metric", "result", "performance",
    ],
    "quality": [
        "novelty", "contribution", "significance", "impact", "originality",
        "clarity", "presentation", "writing", "organization", "reproducibility",
    ],
    "technical": [
        "training", "optimization", "loss", "gradient", "convergence",
        "scalability", "efficiency", "complexity", "computation", "memory",
    ],
    "data": [
        "dataset", "data", "sample", "distribution", "annotation",
        "augmentation", "preprocessing", "feature", "representation", "embedding",
    ],
    "theory": [
        "theorem", "proof", "bound", "guarantee", "assumption",
        "analysis", "theoretical", "mathematical", "formulation", "derivation",
    ],
    "concerns": [
        "limitation", "weakness", "concern", "issue", "problem",
        "missing", "unclear", "insufficient", "question", "clarification",
    ],
}


@dataclass
class TopicResult:
    """Topics extracted from a single review."""
    paper_id: str
    model: str
    topics: list[str]
    extractor: str  # "llm" or "keywords"


def extract_keywords(text: str) -> list[str]:
    """Extract topics using keyword matching."""
    text_lower = text.lower()
    found_topics = set()
    
    # Match keywords
    for category, keywords in KEYWORD_CATEGORIES.items():
        for kw in keywords:
            if kw in text_lower:
                found_topics.add(kw)
    
    # Also extract noun phrases that appear multiple times
    # Simple heuristic: 2-3 word sequences that appear 2+ times
    words = re.findall(r'\b[a-z]+\b', text_lower)
    bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
    bigram_counts = Counter(bigrams)
    
    for bigram, count in bigram_counts.items():
        if count >= 2 and len(bigram) > 6:
            # Filter out common stopword bigrams
            if not any(sw in bigram for sw in ["the ", " the", "this ", " is", " a ", " of ", " to "]):
                found_topics.add(bigram)
    
    return sorted(found_topics)[:15]  # Cap at 15 topics


async def extract_topics_llm(
    client: httpx.AsyncClient,
    api_key: str,
    review_text: str,
    model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1",
    max_retries: int = 3,
) -> list[str]:
    """Extract topics using LLM."""
    if not review_text or len(review_text) < 50:
        return []
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    user_prompt = TOPIC_USER_PROMPT.format(review_text=review_text[:4000])
    
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": TOPIC_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": 200,
    }
    
    for attempt in range(max_retries):
        try:
            response = await client.post(
                "https://api.together.xyz/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30.0,
            )
            
            if response.status_code == 429:
                await asyncio.sleep(2 ** attempt)
                continue
            
            if response.status_code >= 500:
                await asyncio.sleep(2 ** attempt)
                continue
            
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"].strip()
            
            # Parse JSON
            try:
                result = json.loads(content)
                topics = result.get("topics", [])
            except json.JSONDecodeError:
                # Try to find JSON
                match = re.search(r'\{[^{}]*"topics"[^{}]*\}', content)
                if match:
                    result = json.loads(match.group())
                    topics = result.get("topics", [])
                else:
                    topics = []
            
            # Normalize topics
            topics = [t.lower().strip() for t in topics if isinstance(t, str)]
            topics = [t for t in topics if 2 <= len(t) <= 50]
            return topics[:15]
        
        except (httpx.HTTPError, KeyError, json.JSONDecodeError):
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            return []
    
    return []


async def extract_all_topics_llm(
    reviews: list[dict],
    api_key: str,
    max_concurrency: int = 8,
) -> list[TopicResult]:
    """Extract topics from all reviews using LLM."""
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def process_one(client: httpx.AsyncClient, review: dict) -> TopicResult:
        async with semaphore:
            topics = await extract_topics_llm(
                client,
                api_key,
                review.get("review_text", ""),
            )
            return TopicResult(
                paper_id=review.get("paper_id", ""),
                model=review.get("model", ""),
                topics=topics,
                extractor="llm",
            )
    
    async with httpx.AsyncClient() as client:
        tasks = [process_one(client, r) for r in reviews]
        return await asyncio.gather(*tasks)


def extract_all_topics_keywords(reviews: list[dict]) -> list[TopicResult]:
    """Extract topics from all reviews using keyword matching."""
    results = []
    for review in reviews:
        topics = extract_keywords(review.get("review_text", ""))
        results.append(TopicResult(
            paper_id=review.get("paper_id", ""),
            model=review.get("model", ""),
            topics=topics,
            extractor="keywords",
        ))
    return results


def compute_jaccard(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set_a and not set_b:
        return 1.0  # Both empty = identical
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


@dataclass
class PairMetrics:
    """Metrics for comparing two reviews on the same paper."""
    paper_id: str
    model_a: str
    model_b: str
    topics_a: list[str]
    topics_b: list[str]
    overlap_count: int
    union_count: int
    jaccard: float
    topics_only_a: list[str]
    topics_only_b: list[str]


def compute_pairwise_metrics(
    topic_results: list[TopicResult],
    model_a: Optional[str] = None,
    model_b: Optional[str] = None,
) -> list[PairMetrics]:
    """Compute pairwise topic metrics.
    
    If model_a and model_b are specified, only compare that pair.
    Otherwise, compare all model pairs.
    """
    # Group by paper_id
    by_paper: dict[str, dict[str, TopicResult]] = defaultdict(dict)
    for tr in topic_results:
        by_paper[tr.paper_id][tr.model] = tr
    
    # Get all models
    all_models = sorted(set(tr.model for tr in topic_results))
    
    # Determine pairs to compare
    if model_a and model_b:
        pairs = [(model_a, model_b)]
    else:
        pairs = list(combinations(all_models, 2))
    
    metrics = []
    
    for paper_id, model_topics in by_paper.items():
        for ma, mb in pairs:
            if ma not in model_topics or mb not in model_topics:
                continue
            
            topics_a = set(model_topics[ma].topics)
            topics_b = set(model_topics[mb].topics)
            
            overlap = topics_a & topics_b
            union = topics_a | topics_b
            only_a = topics_a - topics_b
            only_b = topics_b - topics_a
            
            metrics.append(PairMetrics(
                paper_id=paper_id,
                model_a=ma,
                model_b=mb,
                topics_a=sorted(topics_a),
                topics_b=sorted(topics_b),
                overlap_count=len(overlap),
                union_count=len(union),
                jaccard=compute_jaccard(topics_a, topics_b),
                topics_only_a=sorted(only_a),
                topics_only_b=sorted(only_b),
            ))
    
    return metrics


def write_csv(metrics: list[PairMetrics], output_path: Path) -> None:
    """Write metrics to CSV file."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "paper_id", "model_a", "model_b",
            "overlap_count", "union_count", "jaccard",
            "topics_only_a", "topics_only_b",
        ])
        
        for m in metrics:
            writer.writerow([
                m.paper_id,
                m.model_a,
                m.model_b,
                m.overlap_count,
                m.union_count,
                f"{m.jaccard:.4f}",
                "; ".join(m.topics_only_a),
                "; ".join(m.topics_only_b),
            ])


def write_summary(
    metrics: list[PairMetrics],
    topic_results: list[TopicResult],
    output_path: Path,
    extractor: str,
) -> None:
    """Write summary markdown file."""
    # Group metrics by model pair
    by_pair: dict[tuple[str, str], list[PairMetrics]] = defaultdict(list)
    for m in metrics:
        by_pair[(m.model_a, m.model_b)].append(m)
    
    # Compute aggregate stats
    lines = [
        "# Topic Coverage Evaluation",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Extractor: {extractor}",
        f"Total reviews: {len(topic_results)}",
        f"Total paper-pairs evaluated: {len(metrics)}",
        "",
        "## Overall Statistics",
        "",
    ]
    
    if metrics:
        all_jaccards = [m.jaccard for m in metrics]
        all_overlaps = [m.overlap_count for m in metrics]
        all_unions = [m.union_count for m in metrics]
        
        lines.extend([
            f"| Metric | Mean | Median | Min | Max |",
            f"|--------|------|--------|-----|-----|",
            f"| Jaccard | {mean(all_jaccards):.4f} | {median(all_jaccards):.4f} | {min(all_jaccards):.4f} | {max(all_jaccards):.4f} |",
            f"| Overlap count | {mean(all_overlaps):.1f} | {median(all_overlaps):.1f} | {min(all_overlaps)} | {max(all_overlaps)} |",
            f"| Union count | {mean(all_unions):.1f} | {median(all_unions):.1f} | {min(all_unions)} | {max(all_unions)} |",
            "",
        ])
    
    lines.append("## Per-Model-Pair Statistics")
    lines.append("")
    
    for (model_a, model_b), pair_metrics in sorted(by_pair.items()):
        jaccards = [m.jaccard for m in pair_metrics]
        overlaps = [m.overlap_count for m in pair_metrics]
        
        lines.extend([
            f"### {model_a} vs {model_b}",
            "",
            f"- Papers compared: {len(pair_metrics)}",
            f"- Mean Jaccard: {mean(jaccards):.4f}",
            f"- Median Jaccard: {median(jaccards):.4f}",
            f"- Mean overlap: {mean(overlaps):.1f} topics",
            "",
        ])
        
        # Top topics unique to each model
        all_only_a: Counter = Counter()
        all_only_b: Counter = Counter()
        for m in pair_metrics:
            all_only_a.update(m.topics_only_a)
            all_only_b.update(m.topics_only_b)
        
        if all_only_a:
            top_a = all_only_a.most_common(5)
            lines.append(f"**Topics more common in {model_a}:**")
            for topic, count in top_a:
                lines.append(f"- {topic} ({count})")
            lines.append("")
        
        if all_only_b:
            top_b = all_only_b.most_common(5)
            lines.append(f"**Topics more common in {model_b}:**")
            for topic, count in top_b:
                lines.append(f"- {topic} ({count})")
            lines.append("")
    
    # Example comparisons
    lines.append("## Example Comparisons")
    lines.append("")
    
    if metrics:
        # Highest Jaccard
        best = max(metrics, key=lambda m: m.jaccard)
        lines.extend([
            f"### Highest overlap (Jaccard={best.jaccard:.4f})",
            f"- Paper: {best.paper_id}",
            f"- {best.model_a}: {', '.join(best.topics_a[:10])}",
            f"- {best.model_b}: {', '.join(best.topics_b[:10])}",
            "",
        ])
        
        # Lowest Jaccard
        worst = min(metrics, key=lambda m: m.jaccard)
        lines.extend([
            f"### Lowest overlap (Jaccard={worst.jaccard:.4f})",
            f"- Paper: {worst.paper_id}",
            f"- {worst.model_a}: {', '.join(worst.topics_a[:10])}",
            f"- {worst.model_b}: {', '.join(worst.topics_b[:10])}",
            "",
        ])
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_charts(
    metrics: list[PairMetrics],
    topic_results: list[TopicResult],
    reviews: list[dict],
    output_dir: Path,
) -> list[str]:
    """Generate visualization charts and return list of created files."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
    except ImportError:
        print("Warning: matplotlib not installed, skipping charts")
        return []
    
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    created_files = []
    
    # Set style
    plt.style.use('seaborn-v0_8-whitegrid') if 'seaborn-v0_8-whitegrid' in plt.style.available else None
    
    # 1. Jaccard Distribution Histogram
    if metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        jaccards = [m.jaccard for m in metrics]
        ax.hist(jaccards, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(mean(jaccards), color='red', linestyle='--', linewidth=2, label=f'Mean: {mean(jaccards):.3f}')
        ax.axvline(median(jaccards), color='orange', linestyle='--', linewidth=2, label=f'Median: {median(jaccards):.3f}')
        ax.set_xlabel('Jaccard Similarity', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Distribution of Topic Overlap (Jaccard Similarity)', fontsize=14)
        ax.legend()
        ax.set_xlim(0, 1)
        
        chart_path = charts_dir / "jaccard_distribution.png"
        plt.tight_layout()
        plt.savefig(chart_path, dpi=150)
        plt.close()
        created_files.append(str(chart_path))
    
    # 2. Jaccard by Model Pair (Bar Chart)
    if metrics:
        by_pair: dict[tuple[str, str], list[float]] = defaultdict(list)
        for m in metrics:
            by_pair[(m.model_a, m.model_b)].append(m.jaccard)
        
        if by_pair:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            pair_labels = []
            means = []
            stds = []
            
            for (ma, mb), jacs in sorted(by_pair.items()):
                # Shorten model names for display
                label_a = ma.split('/')[-1][:20]
                label_b = mb.split('/')[-1][:20]
                pair_labels.append(f"{label_a}\nvs\n{label_b}")
                means.append(mean(jacs))
                stds.append((max(jacs) - min(jacs)) / 2 if len(jacs) > 1 else 0)
            
            x = range(len(pair_labels))
            bars = ax.bar(x, means, yerr=stds, capsize=5, color='steelblue', edgecolor='black', alpha=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(pair_labels, fontsize=9)
            ax.set_ylabel('Mean Jaccard Similarity', fontsize=12)
            ax.set_title('Topic Overlap by Model Pair', fontsize=14)
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, val in zip(bars, means):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                       f'{val:.3f}', ha='center', va='bottom', fontsize=10)
            
            chart_path = charts_dir / "jaccard_by_model_pair.png"
            plt.tight_layout()
            plt.savefig(chart_path, dpi=150)
            plt.close()
            created_files.append(str(chart_path))
    
    # 3. Topic Frequency (Top 20 topics)
    if topic_results:
        all_topics: Counter = Counter()
        for tr in topic_results:
            all_topics.update(tr.topics)
        
        if all_topics:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            top_topics = all_topics.most_common(20)
            topics, counts = zip(*top_topics)
            
            y_pos = range(len(topics))
            ax.barh(y_pos, counts, color='teal', edgecolor='black', alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(topics, fontsize=10)
            ax.invert_yaxis()
            ax.set_xlabel('Frequency', fontsize=12)
            ax.set_title('Top 20 Most Frequent Topics', fontsize=14)
            
            # Add count labels
            for i, count in enumerate(counts):
                ax.text(count + 0.5, i, str(count), va='center', fontsize=9)
            
            chart_path = charts_dir / "topic_frequency.png"
            plt.tight_layout()
            plt.savefig(chart_path, dpi=150)
            plt.close()
            created_files.append(str(chart_path))
    
    # 4. Topics per Review by Model
    if topic_results:
        by_model: dict[str, list[int]] = defaultdict(list)
        for tr in topic_results:
            by_model[tr.model].append(len(tr.topics))
        
        if by_model:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            model_names = []
            data = []
            for model, counts in sorted(by_model.items()):
                model_names.append(model.split('/')[-1][:25])
                data.append(counts)
            
            bp = ax.boxplot(data, tick_labels=model_names, patch_artist=True)
            
            colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightpink']
            for patch, color in zip(bp['boxes'], colors * 10):
                patch.set_facecolor(color)
            
            ax.set_ylabel('Topics per Review', fontsize=12)
            ax.set_xlabel('Model', fontsize=12)
            ax.set_title('Distribution of Topics Extracted per Review', fontsize=14)
            plt.xticks(rotation=15, ha='right')
            
            chart_path = charts_dir / "topics_per_review.png"
            plt.tight_layout()
            plt.savefig(chart_path, dpi=150)
            plt.close()
            created_files.append(str(chart_path))
    
    # 5. Score Distribution by Model (if scores available)
    scores_by_model: dict[str, list[int]] = defaultdict(list)
    for r in reviews:
        score = r.get("score")
        if score is not None:
            scores_by_model[r.get("model", "unknown")].append(score)
    
    if any(scores_by_model.values()):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        model_names = []
        data = []
        for model, scores in sorted(scores_by_model.items()):
            if scores:
                model_names.append(model.split('/')[-1][:25])
                data.append(scores)
        
        if data:
            bp = ax.boxplot(data, tick_labels=model_names, patch_artist=True)
            
            colors = ['steelblue', 'coral', 'seagreen', 'orchid', 'goldenrod']
            for patch, color in zip(bp['boxes'], colors * 10):
                patch.set_facecolor(color)
            
            ax.set_ylabel('Review Score (1-10)', fontsize=12)
            ax.set_xlabel('Model', fontsize=12)
            ax.set_title('Score Distribution by Model', fontsize=14)
            ax.set_ylim(0, 11)
            plt.xticks(rotation=15, ha='right')
            
            chart_path = charts_dir / "score_distribution.png"
            plt.tight_layout()
            plt.savefig(chart_path, dpi=150)
            plt.close()
            created_files.append(str(chart_path))
    
    # 6. Latency Distribution by Model (if available)
    latency_by_model: dict[str, list[float]] = defaultdict(list)
    for r in reviews:
        latency = r.get("meta", {}).get("latency_ms")
        if latency is not None and latency > 0:
            latency_by_model[r.get("model", "unknown")].append(latency / 1000)  # Convert to seconds
    
    if any(latency_by_model.values()):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        model_names = []
        data = []
        for model, latencies in sorted(latency_by_model.items()):
            if latencies:
                model_names.append(model.split('/')[-1][:25])
                data.append(latencies)
        
        if data:
            bp = ax.boxplot(data, tick_labels=model_names, patch_artist=True)
            
            colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors * 10):
                patch.set_facecolor(color)
            
            ax.set_ylabel('Latency (seconds)', fontsize=12)
            ax.set_xlabel('Model', fontsize=12)
            ax.set_title('Generation Latency by Model', fontsize=14)
            plt.xticks(rotation=15, ha='right')
            
            chart_path = charts_dir / "latency_distribution.png"
            plt.tight_layout()
            plt.savefig(chart_path, dpi=150)
            plt.close()
            created_files.append(str(chart_path))
    
    return created_files


def write_consolidated_report(
    metrics: list[PairMetrics],
    topic_results: list[TopicResult],
    reviews: list[dict],
    output_path: Path,
    extractor: str,
    chart_files: list[str],
) -> None:
    """Write a consolidated JSON report with all data."""
    # Group metrics by model pair
    by_pair: dict[str, list[dict]] = defaultdict(list)
    for m in metrics:
        pair_key = f"{m.model_a} vs {m.model_b}"
        by_pair[pair_key].append({
            "paper_id": m.paper_id,
            "jaccard": round(m.jaccard, 4),
            "overlap_count": m.overlap_count,
            "union_count": m.union_count,
            "topics_a": m.topics_a,
            "topics_b": m.topics_b,
            "topics_only_a": m.topics_only_a,
            "topics_only_b": m.topics_only_b,
        })
    
    # Compute aggregate stats per pair
    pair_summaries = {}
    for pair_key, pair_metrics in by_pair.items():
        jaccards = [m["jaccard"] for m in pair_metrics]
        overlaps = [m["overlap_count"] for m in pair_metrics]
        unions = [m["union_count"] for m in pair_metrics]
        
        # Count unique topics per model
        all_only_a: Counter = Counter()
        all_only_b: Counter = Counter()
        for m in pair_metrics:
            all_only_a.update(m["topics_only_a"])
            all_only_b.update(m["topics_only_b"])
        
        pair_summaries[pair_key] = {
            "num_papers": len(pair_metrics),
            "jaccard": {
                "mean": round(mean(jaccards), 4),
                "median": round(median(jaccards), 4),
                "min": round(min(jaccards), 4),
                "max": round(max(jaccards), 4),
            },
            "overlap": {
                "mean": round(mean(overlaps), 2),
                "median": round(median(overlaps), 2),
            },
            "union": {
                "mean": round(mean(unions), 2),
                "median": round(median(unions), 2),
            },
            "top_unique_topics_a": dict(all_only_a.most_common(10)),
            "top_unique_topics_b": dict(all_only_b.most_common(10)),
        }
    
    # Model-level stats
    model_stats = {}
    by_model: dict[str, list] = defaultdict(list)
    for tr in topic_results:
        by_model[tr.model].append(tr)
    
    for model, trs in by_model.items():
        topic_counts = [len(tr.topics) for tr in trs]
        all_topics: Counter = Counter()
        for tr in trs:
            all_topics.update(tr.topics)
        
        # Get scores and latency from reviews
        model_reviews = [r for r in reviews if r.get("model") == model]
        scores = [r.get("score") for r in model_reviews if r.get("score") is not None]
        latencies = [r.get("meta", {}).get("latency_ms") for r in model_reviews 
                     if r.get("meta", {}).get("latency_ms") is not None]
        
        model_stats[model] = {
            "num_reviews": len(trs),
            "topics_per_review": {
                "mean": round(mean(topic_counts), 2) if topic_counts else 0,
                "median": round(median(topic_counts), 2) if topic_counts else 0,
            },
            "top_topics": dict(all_topics.most_common(15)),
            "unique_topics_count": len(all_topics),
            "scores": {
                "mean": round(mean(scores), 2) if scores else None,
                "median": round(median(scores), 2) if scores else None,
                "count": len(scores),
            },
            "latency_ms": {
                "mean": round(mean(latencies), 0) if latencies else None,
                "median": round(median(latencies), 0) if latencies else None,
            },
        }
    
    # Overall stats
    all_jaccards = [m.jaccard for m in metrics]
    all_overlaps = [m.overlap_count for m in metrics]
    
    report = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "extractor": extractor,
            "total_reviews": len(reviews),
            "total_papers": len(set(r.get("paper_id") for r in reviews)),
            "models": list(by_model.keys()),
            "total_pairwise_comparisons": len(metrics),
        },
        "overall_statistics": {
            "jaccard": {
                "mean": round(mean(all_jaccards), 4) if all_jaccards else None,
                "median": round(median(all_jaccards), 4) if all_jaccards else None,
                "min": round(min(all_jaccards), 4) if all_jaccards else None,
                "max": round(max(all_jaccards), 4) if all_jaccards else None,
            },
            "overlap": {
                "mean": round(mean(all_overlaps), 2) if all_overlaps else None,
                "median": round(median(all_overlaps), 2) if all_overlaps else None,
            },
        },
        "model_statistics": model_stats,
        "pairwise_summaries": pair_summaries,
        "pairwise_details": dict(by_pair),
        "charts": chart_files,
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate topic coverage between model-generated reviews."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input reviews.jsonl from run_multi_model_reviews.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--extractor",
        choices=["llm", "keywords"],
        default="keywords",
        help="Topic extraction method (default: keywords)",
    )
    parser.add_argument(
        "--model-a",
        type=str,
        default=None,
        help="First model to compare (optional, compares all pairs if not specified)",
    )
    parser.add_argument(
        "--model-b",
        type=str,
        default=None,
        help="Second model to compare (optional)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="Max concurrent API requests for LLM extraction",
    )
    
    args = parser.parse_args()
    
    # Validate
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    if (args.model_a and not args.model_b) or (args.model_b and not args.model_a):
        raise ValueError("Both --model-a and --model-b must be specified, or neither")
    
    # Load reviews
    reviews = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                reviews.append(json.loads(line))
    
    print(f"Loaded {len(reviews)} reviews from {args.input}")
    
    # Get unique models
    models = sorted(set(r.get("model", "") for r in reviews))
    print(f"Models found: {models}")
    
    # Extract topics
    print(f"\nExtracting topics using: {args.extractor}")
    start_time = time.time()
    
    if args.extractor == "llm":
        api_key = os.environ.get("TOGETHER_API_KEY", "")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY required for LLM extraction")
        
        topic_results = asyncio.run(
            extract_all_topics_llm(reviews, api_key, args.max_concurrency)
        )
    else:
        topic_results = extract_all_topics_keywords(reviews)
    
    elapsed = time.time() - start_time
    print(f"Extracted topics from {len(topic_results)} reviews in {elapsed:.1f}s")
    
    # Compute stats on extraction
    topic_counts = [len(tr.topics) for tr in topic_results]
    if topic_counts:
        print(f"Topics per review: mean={mean(topic_counts):.1f}, median={median(topic_counts):.1f}")
    
    # Save extracted topics
    args.output_dir.mkdir(parents=True, exist_ok=True)
    topics_file = args.output_dir / "topics_extracted.jsonl"
    with open(topics_file, "w", encoding="utf-8") as f:
        for tr in topic_results:
            f.write(json.dumps({
                "paper_id": tr.paper_id,
                "model": tr.model,
                "topics": tr.topics,
                "extractor": tr.extractor,
            }, ensure_ascii=False) + "\n")
    print(f"Saved extracted topics to {topics_file}")
    
    # Compute pairwise metrics
    print("\nComputing pairwise metrics...")
    metrics = compute_pairwise_metrics(
        topic_results,
        model_a=args.model_a,
        model_b=args.model_b,
    )
    
    if not metrics:
        print("No valid pairs to compare. Check that reviews have matching paper_ids.")
        return
    
    print(f"Computed {len(metrics)} paper-pair comparisons")
    
    # Write outputs
    csv_file = args.output_dir / "topic_eval.csv"
    write_csv(metrics, csv_file)
    print(f"Saved CSV: {csv_file}")
    
    summary_file = args.output_dir / "topic_eval_summary.md"
    write_summary(metrics, topic_results, summary_file, args.extractor)
    print(f"Saved summary: {summary_file}")
    
    # Generate charts
    print("\nGenerating charts...")
    chart_files = generate_charts(metrics, topic_results, reviews, args.output_dir)
    if chart_files:
        print(f"Generated {len(chart_files)} charts in {args.output_dir / 'charts'}")
        for cf in chart_files:
            print(f"  - {Path(cf).name}")
    
    # Write consolidated report
    report_file = args.output_dir / "evaluation_report.json"
    write_consolidated_report(
        metrics, topic_results, reviews, report_file, args.extractor, chart_files
    )
    print(f"Saved consolidated report: {report_file}")
    
    # Print quick summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    
    jaccards = [m.jaccard for m in metrics]
    overlaps = [m.overlap_count for m in metrics]
    
    print(f"Mean Jaccard: {mean(jaccards):.4f}")
    print(f"Median Jaccard: {median(jaccards):.4f}")
    print(f"Mean topic overlap: {mean(overlaps):.1f}")
    
    print(f"\nAll results saved to: {args.output_dir}")
    print("Key files:")
    print(f"  - evaluation_report.json  (consolidated data)")
    print(f"  - topic_eval_summary.md   (human-readable summary)")
    print(f"  - charts/                 (visualizations)")


if __name__ == "__main__":
    main()
