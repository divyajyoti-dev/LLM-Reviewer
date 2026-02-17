#!/usr/bin/env python3
"""
Phase 3: Compare LLM reviews to human-review keywords.

Measures how well LLM-generated reviews cover the salient points
that human reviewers emphasized.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median, stdev
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class CoverageResult:
    paper_id: str
    model: str
    human_keywords: list[str]
    llm_keywords: list[str]
    exact_matches: list[str]
    fuzzy_matches: list[str]
    missed_keywords: list[str]
    extra_keywords: list[str]
    exact_recall: float
    fuzzy_recall: float
    precision: float
    f1_score: float


def normalize_keyword(kw: str) -> str:
    """Normalize a keyword for matching."""
    kw = kw.lower().strip()
    # Remove common suffixes
    kw = re.sub(r'(ing|ed|s|tion|ment)$', '', kw)
    # Remove non-alphanumeric
    kw = re.sub(r'[^a-z0-9\s]', '', kw)
    return kw.strip()


def tokenize(text: str) -> set[str]:
    """Tokenize text into normalized words."""
    words = re.findall(r'\b[a-z]+\b', text.lower())
    return set(normalize_keyword(w) for w in words if len(w) > 2)


def fuzzy_match(kw: str, text_tokens: set[str]) -> bool:
    """Check if keyword appears (fuzzily) in text."""
    kw_norm = normalize_keyword(kw)
    kw_tokens = set(kw_norm.split())
    
    # Direct match
    if kw_norm in text_tokens:
        return True
    
    # All tokens of keyword present
    if kw_tokens and kw_tokens.issubset(text_tokens):
        return True
    
    # Partial match (at least half of keyword tokens)
    if kw_tokens:
        matches = sum(1 for t in kw_tokens if t in text_tokens)
        if matches >= len(kw_tokens) * 0.5:
            return True
    
    return False


def compute_coverage(
    human_keywords: list[str],
    llm_review: dict,
) -> CoverageResult:
    """Compute coverage metrics for a single paper-model pair."""
    
    # Combine all LLM review text
    llm_text_parts = [
        llm_review.get("summary", ""),
        " ".join(llm_review.get("strengths", [])),
        " ".join(llm_review.get("weaknesses", [])),
        " ".join(llm_review.get("questions", [])),
    ]
    llm_full_text = " ".join(llm_text_parts).lower()
    llm_tokens = tokenize(llm_full_text)
    
    # LLM keywords
    llm_keywords = llm_review.get("technical_keywords", [])
    llm_keyword_set = set(normalize_keyword(k) for k in llm_keywords)
    
    # Compute matches
    exact_matches = []
    fuzzy_matches = []
    missed = []
    
    human_normalized = set()
    for hk in human_keywords:
        hk_norm = normalize_keyword(hk)
        human_normalized.add(hk_norm)
        
        # Check exact match in LLM keywords
        if hk_norm in llm_keyword_set:
            exact_matches.append(hk)
        # Check fuzzy match in full review text
        elif fuzzy_match(hk, llm_tokens):
            fuzzy_matches.append(hk)
        else:
            missed.append(hk)
    
    # Extra keywords (in LLM but not in human)
    extra = [k for k in llm_keywords if normalize_keyword(k) not in human_normalized]
    
    # Compute metrics
    total_human = len(human_keywords)
    n_exact = len(exact_matches)
    n_fuzzy = len(fuzzy_matches)
    n_total_found = n_exact + n_fuzzy
    
    exact_recall = n_exact / total_human if total_human > 0 else 0
    fuzzy_recall = n_total_found / total_human if total_human > 0 else 0
    
    # Precision: how many LLM keywords are relevant (matched human keywords)
    n_llm = len(llm_keywords)
    matched_llm = len([k for k in llm_keywords 
                       if normalize_keyword(k) in human_normalized 
                       or fuzzy_match(k, tokenize(" ".join(human_keywords)))])
    precision = matched_llm / n_llm if n_llm > 0 else 0
    
    # F1
    f1 = 2 * precision * fuzzy_recall / (precision + fuzzy_recall) if (precision + fuzzy_recall) > 0 else 0
    
    return CoverageResult(
        paper_id=llm_review.get("paper_id", ""),
        model=llm_review.get("model", ""),
        human_keywords=human_keywords,
        llm_keywords=llm_keywords,
        exact_matches=exact_matches,
        fuzzy_matches=fuzzy_matches,
        missed_keywords=missed,
        extra_keywords=extra,
        exact_recall=exact_recall,
        fuzzy_recall=fuzzy_recall,
        precision=precision,
        f1_score=f1,
    )


def generate_charts(results: list[CoverageResult], output_dir: Path):
    """Generate visualization charts."""
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(exist_ok=True)
    
    # Group by model
    by_model = defaultdict(list)
    for r in results:
        by_model[r.model].append(r)
    
    models = sorted(by_model.keys())
    short_names = {m: m.split("/")[-1][:20] for m in models}
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    # 1. Recall by Model (boxplot)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    recall_data = [[r.fuzzy_recall for r in by_model[m]] for m in models]
    bp = ax1.boxplot(recall_data, tick_labels=[short_names[m] for m in models], patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax1.set_ylabel("Keyword Recall")
    ax1.set_title("Human Keyword Coverage by Model")
    ax1.tick_params(axis="x", rotation=45)
    ax1.set_ylim(0, 1)
    ax1.grid(axis="y", alpha=0.3)
    
    ax2 = axes[1]
    f1_data = [[r.f1_score for r in by_model[m]] for m in models]
    bp = ax2.boxplot(f1_data, tick_labels=[short_names[m] for m in models], patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax2.set_ylabel("F1 Score")
    ax2.set_title("F1 Score (Precision-Recall Balance)")
    ax2.tick_params(axis="x", rotation=45)
    ax2.set_ylim(0, 1)
    ax2.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(charts_dir / "coverage_by_model.png", dpi=150)
    plt.close()
    
    # 2. Mean metrics bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(models))
    width = 0.25
    
    exact_means = [mean([r.exact_recall for r in by_model[m]]) for m in models]
    fuzzy_means = [mean([r.fuzzy_recall for r in by_model[m]]) for m in models]
    precision_means = [mean([r.precision for r in by_model[m]]) for m in models]
    
    bars1 = ax.bar(x - width, exact_means, width, label="Exact Recall", color="steelblue")
    bars2 = ax.bar(x, fuzzy_means, width, label="Fuzzy Recall", color="forestgreen")
    bars3 = ax.bar(x + width, precision_means, width, label="Precision", color="coral")
    
    ax.set_ylabel("Score")
    ax.set_title("Coverage Metrics by Model")
    ax.set_xticks(x)
    ax.set_xticklabels([short_names[m] for m in models], rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(charts_dir / "metrics_comparison.png", dpi=150)
    plt.close()
    
    # 3. Most missed keywords (across all models)
    missed_counts = defaultdict(int)
    for r in results:
        for kw in r.missed_keywords:
            missed_counts[normalize_keyword(kw)] += 1
    
    top_missed = sorted(missed_counts.items(), key=lambda x: x[1], reverse=True)[:15]
    
    if top_missed:
        fig, ax = plt.subplots(figsize=(10, 6))
        keywords, counts = zip(*top_missed)
        ax.barh(range(len(keywords)), counts, color="tomato")
        ax.set_yticks(range(len(keywords)))
        ax.set_yticklabels(keywords)
        ax.set_xlabel("Times Missed")
        ax.set_title("Most Frequently Missed Human Keywords")
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig(charts_dir / "missed_keywords.png", dpi=150)
        plt.close()
    
    # 4. Model ranking
    fig, ax = plt.subplots(figsize=(10, 6))
    
    model_f1 = [(m, mean([r.f1_score for r in by_model[m]])) for m in models]
    model_f1.sort(key=lambda x: x[1], reverse=True)
    
    models_sorted, f1_sorted = zip(*model_f1)
    colors_sorted = [colors[models.index(m)] for m in models_sorted]
    
    bars = ax.barh(range(len(models_sorted)), f1_sorted, color=colors_sorted)
    ax.set_yticks(range(len(models_sorted)))
    ax.set_yticklabels([short_names[m] for m in models_sorted])
    ax.set_xlabel("Mean F1 Score")
    ax.set_title("Model Ranking by Coverage Performance")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    
    for bar, score in zip(bars, f1_sorted):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
               f"{score:.3f}", va="center")
    
    plt.tight_layout()
    plt.savefig(charts_dir / "model_ranking.png", dpi=150)
    plt.close()
    
    print(f"Generated charts in {charts_dir}")


def write_report(results: list[CoverageResult], output_dir: Path):
    """Write markdown summary report."""
    report_path = output_dir / "coverage_report.md"
    
    by_model = defaultdict(list)
    for r in results:
        by_model[r.model].append(r)
    
    models = sorted(by_model.keys())
    
    with open(report_path, "w") as f:
        f.write("# Phase 3: LLM Review Coverage Analysis\n\n")
        f.write(f"Generated: {datetime.now(timezone.utc).isoformat()}\n\n")
        f.write("## Overview\n\n")
        f.write("This report measures how well LLM-generated reviews cover the key points\n")
        f.write("that human reviewers emphasized in their reviews.\n\n")
        
        # Overall stats
        all_recall = [r.fuzzy_recall for r in results]
        all_f1 = [r.f1_score for r in results]
        
        f.write("## Overall Statistics\n\n")
        f.write(f"- Total papers evaluated: {len(set(r.paper_id for r in results))}\n")
        f.write(f"- Total model-paper comparisons: {len(results)}\n")
        f.write(f"- Mean keyword recall: {mean(all_recall):.3f}\n")
        f.write(f"- Mean F1 score: {mean(all_f1):.3f}\n\n")
        
        # Per-model stats
        f.write("## Per-Model Performance\n\n")
        f.write("| Model | N | Exact Recall | Fuzzy Recall | Precision | F1 |\n")
        f.write("|-------|---|--------------|--------------|-----------|----|\n")
        
        model_scores = []
        for m in models:
            mr = by_model[m]
            exact = mean([r.exact_recall for r in mr])
            fuzzy = mean([r.fuzzy_recall for r in mr])
            prec = mean([r.precision for r in mr])
            f1 = mean([r.f1_score for r in mr])
            model_scores.append((m, f1))
            
            short = m.split("/")[-1][:30]
            f.write(f"| {short} | {len(mr)} | {exact:.3f} | {fuzzy:.3f} | {prec:.3f} | {f1:.3f} |\n")
        
        # Rankings
        model_scores.sort(key=lambda x: x[1], reverse=True)
        f.write("\n## Model Rankings (by F1 Score)\n\n")
        for i, (m, score) in enumerate(model_scores, 1):
            short = m.split("/")[-1]
            f.write(f"{i}. **{short}**: F1 = {score:.3f}\n")
        
        # Most missed keywords
        missed_counts = defaultdict(int)
        for r in results:
            for kw in r.missed_keywords:
                missed_counts[normalize_keyword(kw)] += 1
        
        top_missed = sorted(missed_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        if top_missed:
            f.write("\n## Most Frequently Missed Keywords\n\n")
            f.write("These human-reviewer keywords were most often NOT covered by LLMs:\n\n")
            for kw, count in top_missed:
                f.write(f"- **{kw}**: missed {count} times\n")
        
        # Key findings
        f.write("\n## Key Findings\n\n")
        best = model_scores[0]
        worst = model_scores[-1]
        f.write(f"- **Best performing model**: {best[0].split('/')[-1]} (F1={best[1]:.3f})\n")
        f.write(f"- **Lowest performing model**: {worst[0].split('/')[-1]} (F1={worst[1]:.3f})\n")
        f.write(f"- **Performance gap**: {best[1] - worst[1]:.3f}\n")
    
    print(f"Wrote report: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3: Compare LLM reviews to human keywords"
    )
    parser.add_argument(
        "--human-keywords", "-k",
        required=True,
        help="JSONL file with human review keywords (from Phase 1)"
    )
    parser.add_argument(
        "--llm-reviews", "-r",
        required=True,
        help="JSONL file with LLM reviews (from Phase 2)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        required=True,
        help="Output directory for coverage analysis"
    )
    
    args = parser.parse_args()
    
    # Load human keywords
    human_kw_path = Path(args.human_keywords)
    if not human_kw_path.exists():
        print(f"Error: Human keywords file not found: {human_kw_path}")
        sys.exit(1)
    
    human_keywords = {}
    with open(human_kw_path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                paper_id = data["paper_id"]
                # Combine keywords, strengths, weaknesses as keyword set
                kws = data.get("keywords", [])
                human_keywords[paper_id] = kws
    
    print(f"Loaded human keywords for {len(human_keywords)} papers")
    
    # Load LLM reviews
    llm_path = Path(args.llm_reviews)
    if not llm_path.exists():
        print(f"Error: LLM reviews file not found: {llm_path}")
        sys.exit(1)
    
    llm_reviews = []
    with open(llm_path) as f:
        for line in f:
            if line.strip():
                llm_reviews.append(json.loads(line))
    
    print(f"Loaded {len(llm_reviews)} LLM reviews")
    
    # Compute coverage
    results = []
    for review in llm_reviews:
        paper_id = review["paper_id"]
        if paper_id in human_keywords:
            result = compute_coverage(human_keywords[paper_id], review)
            results.append(result)
    
    print(f"Computed coverage for {len(results)} paper-model pairs")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Write detailed results
    results_path = output_dir / "coverage_results.jsonl"
    with open(results_path, "w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")
    
    # Generate charts and report
    generate_charts(results, output_dir)
    write_report(results, output_dir)
    
    # Write metadata
    meta_path = output_dir / "phase3_meta.json"
    all_recall = [r.fuzzy_recall for r in results]
    all_f1 = [r.f1_score for r in results]
    
    with open(meta_path, "w") as f:
        json.dump({
            "phase": 3,
            "description": "Compare LLM reviews to human keywords",
            "papers_compared": len(set(r.paper_id for r in results)),
            "total_comparisons": len(results),
            "mean_fuzzy_recall": mean(all_recall),
            "mean_f1_score": mean(all_f1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)
    
    print()
    print("=" * 60)
    print("PHASE 3 COMPLETE")
    print("=" * 60)
    print(f"Comparisons: {len(results)}")
    print(f"Mean keyword recall: {mean(all_recall):.3f}")
    print(f"Mean F1 score: {mean(all_f1):.3f}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
