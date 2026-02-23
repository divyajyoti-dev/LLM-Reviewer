#!/usr/bin/env python3
"""
Visualizations for the semantic coverage evaluation results.

Usage:
    python evaluation_v3/visualize_results.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


# ── Helpers ────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def short_name(model: str) -> str:
    name = model.split("/")[-1]
    for sfx in ["-Instruct-Turbo", "-Instruct", "-v0.1"]:
        name = name.replace(sfx, "")
    return name[:20]


# ── Charts ─────────────────────────────────────────────────────────────

def plot_coverage_by_model(results, ax, colors):
    """Box plot of coverage scores per model."""
    by_model = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r["coverage_score"] * 100)

    models = sorted(by_model.keys(), key=lambda m: mean(by_model[m]), reverse=True)
    data = [by_model[m] for m in models]
    names = [short_name(m) for m in models]

    bp = ax.boxplot(data, tick_labels=names, patch_artist=True, widths=0.6)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.8)
    for median_line in bp["medians"]:
        median_line.set_color("black")
        median_line.set_linewidth(2)

    ax.set_ylabel("Coverage (%)", fontsize=11)
    ax.set_title("Keyword Coverage by Model", fontsize=13, fontweight="bold")
    ax.tick_params(axis="x", rotation=30)
    ax.set_ylim(0, 105)
    ax.axhline(y=mean([r["coverage_score"] * 100 for r in results]),
               color="red", linestyle="--", alpha=0.6, label="Overall mean")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)


def plot_model_ranking(results, ax, colors):
    """Horizontal bar chart ranking models by mean coverage."""
    by_model = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r["coverage_score"])

    ranking = sorted(by_model.items(), key=lambda x: mean(x[1]), reverse=True)
    models_sorted = [m for m, _ in ranking]
    means = [mean(s) * 100 for _, s in ranking]

    bars = ax.barh(range(len(models_sorted)), means,
                   color=[colors[i] for i in range(len(models_sorted))],
                   edgecolor="black", linewidth=0.5)
    ax.set_yticks(range(len(models_sorted)))
    ax.set_yticklabels([short_name(m) for m in models_sorted], fontsize=10)
    ax.set_xlabel("Mean Coverage (%)", fontsize=11)
    ax.set_title("Model Ranking", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 55)
    ax.invert_yaxis()

    for bar, val in zip(bars, means):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=11, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)


def plot_coverage_distribution(results, ax):
    """Histogram of coverage scores across all comparisons."""
    scores = [r["coverage_score"] * 100 for r in results]
    ax.hist(scores, bins=20, color="steelblue", edgecolor="black", alpha=0.75)
    ax.axvline(mean(scores), color="red", linestyle="--", linewidth=2,
               label=f"Mean: {mean(scores):.1f}%")
    ax.axvline(median(scores), color="orange", linestyle="--", linewidth=2,
               label=f"Median: {median(scores):.1f}%")
    ax.set_xlabel("Coverage (%)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Overall Coverage Distribution", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)


def plot_missed_keywords(results, ax):
    """Bar chart of most frequently missed keywords."""
    missed = Counter()
    for r in results:
        for kw in r.get("missed_keywords", []):
            missed[kw.lower().strip()] += 1

    top = missed.most_common(12)
    if not top:
        ax.text(0.5, 0.5, "No missed keywords", ha="center", va="center")
        return

    keywords, counts = zip(*top)
    y_pos = range(len(keywords))
    ax.barh(y_pos, counts, color="tomato", edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(keywords, fontsize=9)
    ax.set_xlabel("Times Missed", fontsize=11)
    ax.set_title("Most Frequently Missed Keywords", fontsize=13, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)


def plot_matched_keywords(results, ax):
    """Bar chart of most frequently matched keywords with avg similarity."""
    matched = defaultdict(list)
    for r in results:
        for km in r.get("keyword_matches", []):
            if km["matched"]:
                matched[km["human_keyword"].lower().strip()].append(km["similarity"])

    top = sorted(matched.items(), key=lambda x: len(x[1]), reverse=True)[:12]
    if not top:
        ax.text(0.5, 0.5, "No matched keywords", ha="center", va="center")
        return

    keywords = [kw for kw, _ in top]
    counts = [len(sims) for _, sims in top]
    avg_sims = [mean(sims) for _, sims in top]

    y_pos = range(len(keywords))
    bars = ax.barh(y_pos, counts, color="mediumseagreen", edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(keywords, fontsize=9)
    ax.set_xlabel("Times Matched", fontsize=11)
    ax.set_title("Most Commonly Matched Keywords", fontsize=13, fontweight="bold")
    ax.invert_yaxis()

    for bar, sim in zip(bars, avg_sims):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"sim={sim:.2f}", va="center", fontsize=8, color="gray")
    ax.grid(axis="x", alpha=0.3)


def plot_similarity_heatmap(results, ax, colors):
    """Heatmap of avg similarity scores per model (matched keywords only)."""
    by_model = defaultdict(list)
    for r in results:
        for km in r.get("keyword_matches", []):
            by_model[r["model"]].append(km["similarity"])

    models = sorted(by_model.keys(), key=lambda m: mean(by_model[m]), reverse=True)

    # Create bins for similarity distribution
    bins = np.arange(0, 1.05, 0.1)
    bin_labels = [f"{b:.1f}" for b in bins[:-1]]

    data = []
    for m in models:
        sims = by_model[m]
        hist, _ = np.histogram(sims, bins=bins)
        data.append(hist / len(sims) * 100)

    data = np.array(data)
    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(bin_labels)))
    ax.set_xticklabels(bin_labels, fontsize=9)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([short_name(m) for m in models], fontsize=10)
    ax.set_xlabel("Similarity Score", fontsize=11)
    ax.set_title("Similarity Distribution by Model (%)", fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="% of keyword pairs")


def plot_per_paper_coverage(results, ax):
    """Scatter plot of per-paper avg coverage (across models)."""
    by_paper = defaultdict(list)
    for r in results:
        by_paper[r["paper_id"]].append(r["coverage_score"] * 100)

    paper_means = sorted([mean(scores) for scores in by_paper.values()])

    ax.bar(range(len(paper_means)), paper_means, color="mediumpurple",
           edgecolor="none", alpha=0.7, width=1.0)
    ax.axhline(y=mean(paper_means), color="red", linestyle="--", linewidth=2,
               label=f"Mean: {mean(paper_means):.1f}%")
    ax.set_xlabel("Papers (sorted by coverage)", fontsize=11)
    ax.set_ylabel("Avg Coverage (%)", fontsize=11)
    ax.set_title("Per-Paper Coverage (avg across models)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)


# ── Dashboard ──────────────────────────────────────────────────────────

def create_dashboard(results, output_dir):
    """Create a 6-panel evaluation dashboard."""
    colors = plt.cm.Set2(np.linspace(0, 1, 5))

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("LLM Review — Semantic Coverage Evaluation",
                 fontsize=18, fontweight="bold", y=0.98)

    ax1 = fig.add_subplot(2, 3, 1)
    plot_coverage_by_model(results, ax1, colors)

    ax2 = fig.add_subplot(2, 3, 2)
    plot_model_ranking(results, ax2, colors)

    ax3 = fig.add_subplot(2, 3, 3)
    plot_coverage_distribution(results, ax3)

    ax4 = fig.add_subplot(2, 3, 4)
    plot_missed_keywords(results, ax4)

    ax5 = fig.add_subplot(2, 3, 5)
    plot_matched_keywords(results, ax5)

    ax6 = fig.add_subplot(2, 3, 6)
    plot_per_paper_coverage(results, ax6)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = output_dir / "evaluation_dashboard.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")


def create_similarity_detail(results, output_dir):
    """Create detailed similarity analysis charts."""
    colors = plt.cm.Set2(np.linspace(0, 1, 5))

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Keyword Similarity Analysis", fontsize=15, fontweight="bold")

    plot_similarity_heatmap(results, axes[0], colors)

    # Matched vs missed per model
    by_model = defaultdict(lambda: {"matched": 0, "missed": 0})
    for r in results:
        by_model[r["model"]]["matched"] += r["matched_count"]
        by_model[r["model"]]["missed"] += len(r["missed_keywords"])

    models = sorted(by_model.keys(), key=lambda m: short_name(m))
    matched = [by_model[m]["matched"] for m in models]
    missed_vals = [by_model[m]["missed"] for m in models]
    names = [short_name(m) for m in models]

    x = np.arange(len(models))
    width = 0.35
    axes[1].bar(x - width / 2, matched, width, label="Matched", color="mediumseagreen",
                edgecolor="black", linewidth=0.5)
    axes[1].bar(x + width / 2, missed_vals, width, label="Missed", color="tomato",
                edgecolor="black", linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30, ha="right", fontsize=10)
    axes[1].set_ylabel("Total Keywords", fontsize=11)
    axes[1].set_title("Matched vs Missed Keywords per Model", fontsize=13, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = output_dir / "similarity_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")


def create_summary_infographic(results, human_data, output_dir):
    """Create a one-page summary infographic."""
    by_model = defaultdict(list)
    for r in results:
        by_model[r["model"]].append(r["coverage_score"])

    all_cov = [r["coverage_score"] for r in results]
    models = sorted(by_model.keys(), key=lambda m: mean(by_model[m]), reverse=True)

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    # Title
    ax.text(7, 9.5, "Semantic Coverage: Can LLMs Review Like Humans?",
            fontsize=20, fontweight="bold", ha="center", va="top")
    ax.text(7, 8.95, "Top-10 keyword overlap between human and LLM-generated peer reviews",
            fontsize=12, ha="center", va="top", style="italic", color="gray")

    # Key metrics boxes
    box = dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9)
    metrics = [
        (2, "Papers\nEvaluated", f"{len(set(r['paper_id'] for r in results))}"),
        (5, "Models\nCompared", f"{len(models)}"),
        (8, "Total\nComparisons", f"{len(results)}"),
        (11, "Mean\nCoverage", f"{mean(all_cov):.0%}"),
    ]
    for x, label, val in metrics:
        ax.text(x, 7.7, f"{label}\n{val}", fontsize=14, ha="center", va="center",
                bbox=box, fontweight="bold")

    # Model rankings
    ax.text(1.5, 6.0, "Model Rankings", fontsize=14, fontweight="bold")
    medal_colors = ["#FFD700", "#C0C0C0", "#CD7F32", "#A0A0A0", "#808080"]
    for i, m in enumerate(models):
        y = 5.4 - i * 0.55
        ax.scatter([1.3], [y], s=150, c=medal_colors[i], edgecolors="black", zorder=5)
        ax.text(1.8, y, f"{i + 1}. {short_name(m)}", fontsize=11, va="center")
        score = mean(by_model[m])
        ax.text(6, y, f"{score:.1%}", fontsize=12, va="center", fontweight="bold")

    # Key findings
    finding_box = dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.9)
    ax.text(10.5, 5.3,
            f"Key Finding\n\nLLMs semantically cover\nonly {mean(all_cov):.0%} of the keywords\n"
            f"human reviewers emphasize.\n\n~{(1 - mean(all_cov)):.0%} of human concerns\nare missed.",
            fontsize=12, ha="center", va="center", bbox=finding_box)

    # Most missed
    missed = Counter()
    for r in results:
        for kw in r.get("missed_keywords", []):
            missed[kw.lower().strip()] += 1
    top_missed = missed.most_common(5)

    ax.text(1.5, 2.5, "Top Missed Keywords", fontsize=12, fontweight="bold")
    for i, (kw, count) in enumerate(top_missed):
        ax.text(2, 2.0 - i * 0.35, f"  {kw} ({count}x)", fontsize=10, va="center",
                color="tomato")

    # Avg reviews per paper
    avg_reviews = mean(h.get("review_count", 1) for h in human_data)
    ax.text(8, 2.5, "Dataset Stats", fontsize=12, fontweight="bold")
    ax.text(8.5, 2.0, f"  Avg reviews/paper: {avg_reviews:.1f}", fontsize=10)
    ax.text(8.5, 1.65, f"  Keywords per review: 10", fontsize=10)
    ax.text(8.5, 1.3, f"  Similarity threshold: 0.55", fontsize=10)
    ax.text(8.5, 0.95, f"  Embedding: all-MiniLM-L6-v2", fontsize=10)

    # Footer
    best = models[0]
    worst = models[-1]
    gap = mean(by_model[best]) - mean(by_model[worst])
    ax.text(7, 0.2,
            f"Best: {short_name(best)} ({mean(by_model[best]):.1%}) | "
            f"Worst: {short_name(worst)} ({mean(by_model[worst]):.1%}) | "
            f"Gap: {gap:.1%}",
            fontsize=10, ha="center", color="gray")

    path = output_dir / "summary_infographic.png"
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")


# ── Main ───────────────────────────────────────────────────────────────

def main():
    output_dir = Path("evaluation_v3/outputs")
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(exist_ok=True)

    print("Loading data...")
    results = load_jsonl(output_dir / "semantic_coverage_results.jsonl")
    human_data = load_jsonl(output_dir / "consolidated_reviews.jsonl")
    print(f"Loaded {len(results)} coverage results, {len(human_data)} papers")

    print("\nGenerating visualizations...")
    create_dashboard(results, charts_dir)
    create_similarity_detail(results, charts_dir)
    create_summary_infographic(results, human_data, charts_dir)

    print(f"\nAll charts saved to: {charts_dir}")


if __name__ == "__main__":
    main()
