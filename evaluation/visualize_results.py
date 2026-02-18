#!/usr/bin/env python3
"""
Generate comprehensive visualizations for the 3-phase evaluation.
"""

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt
import numpy as np


def short_name(model: str) -> str:
    """Get shortened model name."""
    name = model.split("/")[-1]
    for suffix in ["-Instruct-Turbo", "-Instruct", "-v0.1"]:
        name = name.replace(suffix, "")
    return name[:18]


def load_jsonl(path: Path) -> list[dict]:
    """Load JSONL file."""
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def create_dashboard(coverage_results: list[dict], human_keywords: list[dict], output_dir: Path):
    """Create main evaluation dashboard."""
    
    # Group by model
    by_model = defaultdict(list)
    for r in coverage_results:
        by_model[r["model"]].append(r)
    
    models = sorted(by_model.keys())
    short_names = [short_name(m) for m in models]
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("LLM Review Coverage Evaluation Dashboard", fontsize=16, fontweight="bold", y=0.98)
    
    # 1. F1 Score by Model (top left)
    ax1 = fig.add_subplot(2, 3, 1)
    f1_data = [[r["f1_score"] for r in by_model[m]] for m in models]
    bp = ax1.boxplot(f1_data, tick_labels=short_names, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
    ax1.set_ylabel("F1 Score")
    ax1.set_title("F1 Score Distribution by Model")
    ax1.tick_params(axis="x", rotation=45)
    ax1.set_ylim(0, 1)
    ax1.grid(axis="y", alpha=0.3)
    
    # 2. Mean Metrics Comparison (top middle)
    ax2 = fig.add_subplot(2, 3, 2)
    x = np.arange(len(models))
    width = 0.25
    
    exact_means = [mean([r["exact_recall"] for r in by_model[m]]) for m in models]
    fuzzy_means = [mean([r["fuzzy_recall"] for r in by_model[m]]) for m in models]
    precision_means = [mean([r["precision"] for r in by_model[m]]) for m in models]
    
    ax2.bar(x - width, exact_means, width, label="Exact Recall", color="steelblue")
    ax2.bar(x, fuzzy_means, width, label="Fuzzy Recall", color="forestgreen")
    ax2.bar(x + width, precision_means, width, label="Precision", color="coral")
    
    ax2.set_ylabel("Score")
    ax2.set_title("Coverage Metrics Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, rotation=45, ha="right")
    ax2.legend(loc="upper right")
    ax2.set_ylim(0, 0.8)
    ax2.grid(axis="y", alpha=0.3)
    
    # 3. Model Ranking (top right)
    ax3 = fig.add_subplot(2, 3, 3)
    model_f1 = [(m, mean([r["f1_score"] for r in by_model[m]])) for m in models]
    model_f1.sort(key=lambda x: x[1], reverse=True)
    
    models_sorted, f1_sorted = zip(*model_f1)
    colors_sorted = [colors[models.index(m)] for m in models_sorted]
    
    bars = ax3.barh(range(len(models_sorted)), f1_sorted, color=colors_sorted)
    ax3.set_yticks(range(len(models_sorted)))
    ax3.set_yticklabels([short_name(m) for m in models_sorted])
    ax3.set_xlabel("Mean F1 Score")
    ax3.set_title("Model Ranking")
    ax3.set_xlim(0, 0.6)
    ax3.invert_yaxis()
    
    for bar, score in zip(bars, f1_sorted):
        ax3.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f"{score:.3f}", va="center", fontsize=10)
    ax3.grid(axis="x", alpha=0.3)
    
    # 4. Recall Distribution (bottom left)
    ax4 = fig.add_subplot(2, 3, 4)
    all_recall = [r["fuzzy_recall"] for r in coverage_results]
    ax4.hist(all_recall, bins=20, color="forestgreen", edgecolor="black", alpha=0.7)
    ax4.axvline(mean(all_recall), color="red", linestyle="--", linewidth=2,
               label=f"Mean: {mean(all_recall):.3f}")
    ax4.axvline(median(all_recall), color="orange", linestyle="--", linewidth=2,
               label=f"Median: {median(all_recall):.3f}")
    ax4.set_xlabel("Keyword Recall")
    ax4.set_ylabel("Count")
    ax4.set_title("Overall Recall Distribution")
    ax4.legend()
    ax4.grid(axis="y", alpha=0.3)
    
    # 5. Most Missed Keywords (bottom middle)
    ax5 = fig.add_subplot(2, 3, 5)
    missed_counts = defaultdict(int)
    for r in coverage_results:
        for kw in r.get("missed_keywords", []):
            # Normalize
            kw_norm = kw.lower().strip()[:25]
            missed_counts[kw_norm] += 1
    
    top_missed = sorted(missed_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    if top_missed:
        keywords, counts = zip(*top_missed)
        ax5.barh(range(len(keywords)), counts, color="tomato")
        ax5.set_yticks(range(len(keywords)))
        ax5.set_yticklabels(keywords, fontsize=9)
        ax5.set_xlabel("Times Missed")
        ax5.set_title("Top Missed Keywords")
        ax5.invert_yaxis()
        ax5.grid(axis="x", alpha=0.3)
    
    # 6. Human Keywords per Paper (bottom right)
    ax6 = fig.add_subplot(2, 3, 6)
    kw_counts = [len(h.get("keywords", [])) for h in human_keywords]
    ax6.hist(kw_counts, bins=15, color="purple", edgecolor="black", alpha=0.7)
    ax6.axvline(mean(kw_counts), color="red", linestyle="--", linewidth=2,
               label=f"Mean: {mean(kw_counts):.1f}")
    ax6.set_xlabel("Keywords per Paper")
    ax6.set_ylabel("Count")
    ax6.set_title("Human Keywords Distribution")
    ax6.legend()
    ax6.grid(axis="y", alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "evaluation_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'evaluation_dashboard.png'}")


def create_model_comparison(coverage_results: list[dict], output_dir: Path):
    """Create per-model detailed comparison."""
    
    by_model = defaultdict(list)
    for r in coverage_results:
        by_model[r["model"]].append(r)
    
    models = sorted(by_model.keys())
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle("Per-Model Recall Distribution", fontsize=14, fontweight="bold")
    
    axes_flat = axes.flatten()
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    
    for idx, model in enumerate(models):
        if idx >= 6:
            break
        ax = axes_flat[idx]
        recalls = [r["fuzzy_recall"] for r in by_model[model]]
        
        ax.hist(recalls, bins=15, color=colors[idx], edgecolor="black", alpha=0.7)
        ax.axvline(mean(recalls), color="red", linestyle="--", linewidth=2)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Count")
        ax.set_title(f"{short_name(model)}\n(μ={mean(recalls):.3f}, n={len(recalls)})")
        ax.set_xlim(0, 1)
        ax.grid(axis="y", alpha=0.3)
    
    for idx in range(len(models), 6):
        axes_flat[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_dir / "model_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_dir / 'model_distributions.png'}")


def create_summary_infographic(coverage_results: list[dict], human_keywords: list[dict], output_dir: Path):
    """Create summary infographic."""
    
    by_model = defaultdict(list)
    for r in coverage_results:
        by_model[r["model"]].append(r)
    
    models = sorted(by_model.keys())
    
    # Compute stats
    all_recall = [r["fuzzy_recall"] for r in coverage_results]
    all_f1 = [r["f1_score"] for r in coverage_results]
    all_precision = [r["precision"] for r in coverage_results]
    
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")
    
    # Title
    ax.text(7, 9.5, "LLM Review Coverage Evaluation", fontsize=22, fontweight="bold", 
           ha="center", va="top")
    ax.text(7, 8.9, "Can LLMs identify the same salient points as human reviewers?", 
           fontsize=12, ha="center", va="top", style="italic", color="gray")
    
    # Key metrics
    box_props = dict(boxstyle="round,pad=0.4", facecolor="lightblue", alpha=0.9)
    
    ax.text(2, 7.5, f"Papers\nEvaluated\n{len(human_keywords)}", fontsize=14, ha="center",
           va="center", bbox=box_props, fontweight="bold")
    ax.text(5, 7.5, f"LLM Reviews\nGenerated\n{len(coverage_results)}", fontsize=14, ha="center",
           va="center", bbox=box_props, fontweight="bold")
    ax.text(8, 7.5, f"Models\nCompared\n{len(models)}", fontsize=14, ha="center",
           va="center", bbox=box_props, fontweight="bold")
    ax.text(11, 7.5, f"Mean\nRecall\n{mean(all_recall):.1%}", fontsize=14, ha="center",
           va="center", bbox=box_props, fontweight="bold")
    
    # Model rankings
    ax.text(2, 5.8, "Model Rankings (F1 Score)", fontsize=14, fontweight="bold")
    
    model_f1 = [(m, mean([r["f1_score"] for r in by_model[m]])) for m in models]
    model_f1.sort(key=lambda x: x[1], reverse=True)
    
    colors = ["#FFD700", "#C0C0C0", "#CD7F32", "#A0A0A0", "#808080"]
    for i, (m, score) in enumerate(model_f1):
        y_pos = 5.2 - i * 0.55
        ax.scatter([1.5], [y_pos], s=150, c=colors[i], edgecolors="black", zorder=5)
        ax.text(2, y_pos, f"{i+1}. {short_name(m)}", fontsize=12, va="center")
        ax.text(5.5, y_pos, f"F1 = {score:.3f}", fontsize=12, va="center", 
               fontweight="bold" if i == 0 else "normal")
    
    # Key finding box
    finding_props = dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9)
    ax.text(10.5, 5, 
           f"Key Finding:\n\nLLMs cover only {mean(all_recall):.0%}\nof human reviewer\nconcerns on average",
           fontsize=13, ha="center", va="center", bbox=finding_props)
    
    # Most missed
    ax.text(2, 2.2, "Most Frequently Missed Keywords:", fontsize=12, fontweight="bold")
    
    missed_counts = defaultdict(int)
    for r in coverage_results:
        for kw in r.get("missed_keywords", []):
            missed_counts[kw.lower().strip()[:20]] += 1
    
    top_missed = sorted(missed_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    for i, (kw, count) in enumerate(top_missed):
        ax.text(2.5, 1.7 - i * 0.35, f"• {kw} ({count}x)", fontsize=10, va="center")
    
    # Performance gap
    best = model_f1[0]
    worst = model_f1[-1]
    gap = best[1] - worst[1]
    
    ax.text(8, 2.2, "Performance Insights:", fontsize=12, fontweight="bold")
    ax.text(8.5, 1.7, f"• Best: {short_name(best[0])} (F1={best[1]:.3f})", fontsize=10)
    ax.text(8.5, 1.35, f"• Worst: {short_name(worst[0])} (F1={worst[1]:.3f})", fontsize=10)
    ax.text(8.5, 1.0, f"• Performance gap: {gap:.3f} ({gap/best[1]:.0%})", fontsize=10)
    ax.text(8.5, 0.65, f"• Larger models ≠ better coverage", fontsize=10, style="italic")
    
    # Footer
    ax.text(7, 0.2, "3-Phase Evaluation: Human Keywords → LLM Reviews → Coverage Analysis", 
           fontsize=10, ha="center", color="gray")
    
    plt.savefig(output_dir / "summary_infographic.png", dpi=150, bbox_inches="tight", 
               facecolor="white")
    plt.close()
    print(f"Saved: {output_dir / 'summary_infographic.png'}")


def main():
    output_dir = Path("evaluation/outputs")
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(exist_ok=True)
    
    print("Loading data...")
    coverage_results = load_jsonl(output_dir / "coverage_results.jsonl")
    human_keywords = load_jsonl(output_dir / "human_keywords.jsonl")
    
    print(f"Loaded {len(coverage_results)} coverage results")
    print(f"Loaded {len(human_keywords)} human keyword sets")
    
    print("\nGenerating visualizations...")
    create_dashboard(coverage_results, human_keywords, charts_dir)
    create_model_comparison(coverage_results, charts_dir)
    create_summary_infographic(coverage_results, human_keywords, charts_dir)
    
    print(f"\nAll visualizations saved to: {charts_dir}")


if __name__ == "__main__":
    main()
