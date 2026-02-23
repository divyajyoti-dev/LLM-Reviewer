#!/usr/bin/env python3
"""
Step 2: Generate LLM reviews from abstracts only (blind).

Each model generates a structured review from the paper's title and abstract,
without any access to the human reviews. This tests whether LLMs can
independently identify the same salient technical points.

Usage:
    export TOGETHER_API_KEY="your-key"
    python evaluation_v3/step2_generate_llm_reviews.py \
        --input evaluation_v3/outputs/consolidated_reviews.jsonl \
        --output evaluation_v3/outputs/llm_reviews.jsonl
"""

import argparse
import asyncio
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx

TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

DEFAULT_MODELS = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "deepseek-ai/DeepSeek-V3",
]

SYSTEM_PROMPT = """You are an expert peer reviewer for a top ML conference (ICLR/NeurIPS/ICML).

Write a thorough review based ONLY on the paper's title and abstract.

Output your review as a JSON object:
{
  "summary": "1-2 sentence summary",
  "strengths": ["strength 1", "strength 2", ...],
  "weaknesses": ["weakness 1", "weakness 2", ...],
  "questions": ["question 1", "question 2", ...],
  "score": <1-10>
}

Guidelines:
- Be specific and technical
- Identify 3-5 strengths and 3-5 weaknesses
- Ask 2-4 clarifying questions
- Score: 1=reject, 5=borderline, 10=strong accept"""

USER_PROMPT_TEMPLATE = """Review this paper:

Title: {title}

Abstract: {abstract}

Provide your review as JSON."""


@dataclass
class LLMReviewResult:
    paper_id: str
    title: str
    model: str
    summary: str
    strengths: list[str]
    weaknesses: list[str]
    questions: list[str]
    score: Optional[int]
    raw_review: str
    latency_ms: float
    error: Optional[str] = None


def parse_review_response(text: str) -> dict:
    """Robustly parse review JSON from LLM response."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"```(?:json)?\n?", "", text)
        text = text.strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return {"summary": "", "strengths": [], "weaknesses": [], "questions": [], "score": None}


async def generate_review_single(
    client: httpx.AsyncClient,
    paper: dict,
    model: str,
    api_key: str,
    temperature: float = 0.3,
    max_retries: int = 3,
) -> LLMReviewResult:
    """Generate a review for a single paper."""
    paper_id = paper.get("paper_id", "unknown")
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")

    if not abstract:
        return LLMReviewResult(
            paper_id=paper_id, title=title, model=model,
            summary="", strengths=[], weaknesses=[], questions=[],
            score=None, raw_review="", latency_ms=0, error="No abstract",
        )

    start = asyncio.get_event_loop().time()

    for attempt in range(max_retries):
        try:
            resp = await client.post(
                TOGETHER_API_URL,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                            title=title, abstract=abstract[:3000],
                        )},
                    ],
                    "temperature": temperature,
                    "max_tokens": 1200,
                },
                timeout=90.0,
            )
            if resp.status_code == 429 or resp.status_code >= 500:
                await asyncio.sleep(2 ** attempt)
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            parsed = parse_review_response(content)
            elapsed = (asyncio.get_event_loop().time() - start) * 1000

            return LLMReviewResult(
                paper_id=paper_id, title=title, model=model,
                summary=parsed.get("summary", ""),
                strengths=parsed.get("strengths", []),
                weaknesses=parsed.get("weaknesses", []),
                questions=parsed.get("questions", []),
                score=parsed.get("score"),
                raw_review=content,
                latency_ms=elapsed,
            )
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            elapsed = (asyncio.get_event_loop().time() - start) * 1000
            return LLMReviewResult(
                paper_id=paper_id, title=title, model=model,
                summary="", strengths=[], weaknesses=[], questions=[],
                score=None, raw_review="", latency_ms=elapsed, error=str(e),
            )

    elapsed = (asyncio.get_event_loop().time() - start) * 1000
    return LLMReviewResult(
        paper_id=paper_id, title=title, model=model,
        summary="", strengths=[], weaknesses=[], questions=[],
        score=None, raw_review="", latency_ms=elapsed, error="Max retries exceeded",
    )


async def generate_for_model(
    papers: list[dict], model: str, api_key: str,
    max_concurrency: int = 5, temperature: float = 0.3,
) -> list[LLMReviewResult]:
    """Generate reviews for all papers using one model."""
    sem = asyncio.Semaphore(max_concurrency)

    async def bounded(client, paper):
        async with sem:
            return await generate_review_single(client, paper, model, api_key, temperature)

    async with httpx.AsyncClient() as client:
        tasks = [bounded(client, p) for p in papers]
        results = []
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            results.append(await coro)
            if (i + 1) % 20 == 0 or (i + 1) == len(papers):
                ok = sum(1 for r in results if not r.error)
                print(f"  {i + 1}/{len(papers)} ({ok} success)")
    return results


def main():
    parser = argparse.ArgumentParser(description="Step 2: Generate LLM reviews from abstracts")
    parser.add_argument("--input", "-i", required=True, help="Input JSONL (consolidated reviews)")
    parser.add_argument("--output", "-o", default="evaluation_v3/outputs/llm_reviews.jsonl")
    parser.add_argument("--models", "-m", help="Comma-separated model IDs (default: 5 predefined)")
    parser.add_argument("--max-concurrency", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        print("Error: TOGETHER_API_KEY required")
        sys.exit(1)

    models = [m.strip() for m in args.models.split(",")] if args.models else DEFAULT_MODELS

    papers = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                papers.append(json.loads(line))
    if args.limit:
        papers = papers[: args.limit]

    print("=" * 60)
    print("STEP 2: Generate LLM Reviews (Blind)")
    print("=" * 60)
    print(f"Papers: {len(papers)}")
    print(f"Models: {len(models)}")
    for m in models:
        print(f"  - {m}")
    print()

    all_results: list[LLMReviewResult] = []
    start = datetime.now(timezone.utc)

    for model in models:
        print(f"--- {model} ---")
        results = asyncio.run(generate_for_model(
            papers, model, api_key, args.max_concurrency, args.temperature,
        ))
        ok = sum(1 for r in results if not r.error)
        print(f"Done: {ok}/{len(results)} success")
        all_results.extend(results)

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in all_results:
            f.write(json.dumps(asdict(r)) + "\n")

    ok = sum(1 for r in all_results if not r.error)
    print()
    print("=" * 60)
    print("STEP 2 COMPLETE")
    print("=" * 60)
    print(f"Total reviews: {len(all_results)} ({ok} success)")
    print(f"Time: {elapsed:.1f}s")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
