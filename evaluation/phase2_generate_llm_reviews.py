#!/usr/bin/env python3
"""
Phase 2: Generate LLM reviews from abstracts only (blind).

Each model generates a structured review without access to the human review.
This tests if LLMs can identify the same salient points as human reviewers.
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

# Together AI API endpoint
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

# Default models to evaluate (Together AI IDs)
DEFAULT_MODELS = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "Qwen/Qwen2.5-7B-Instruct-Turbo",
    "deepseek-ai/DeepSeek-V3",
]

SYSTEM_PROMPT = """You are an expert peer reviewer for a top machine learning conference (ICLR/NeurIPS/ICML).

Your task is to write a thorough review of a paper based ONLY on its title and abstract. Identify the key technical contributions, potential strengths, weaknesses, and questions.

Output your review in this exact JSON format:
{
  "summary": "1-2 sentence summary of the paper's contribution",
  "strengths": ["strength 1", "strength 2", ...],
  "weaknesses": ["weakness 1", "weakness 2", ...],
  "questions": ["question 1", "question 2", ...],
  "technical_keywords": ["keyword1", "keyword2", ...],
  "score": <1-10>,
  "confidence": <1-5>
}

Guidelines:
- Be specific and technical in your critique
- Identify 3-5 strengths and 3-5 weaknesses
- Ask 2-4 clarifying questions
- Extract 10-15 technical keywords that capture the core concepts
- Score: 1=reject, 5=borderline, 10=accept
- Confidence: 1=guess, 3=moderate, 5=certain"""

USER_PROMPT_TEMPLATE = """Please review this paper based on its title and abstract:

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
    technical_keywords: list[str]
    score: Optional[int]
    confidence: Optional[int]
    raw_review: str
    latency_ms: float
    error: Optional[str] = None


def parse_review_response(text: str) -> dict:
    """Robustly parse review JSON from LLM response."""
    text = text.strip()
    
    # Remove markdown code blocks
    if text.startswith("```"):
        text = re.sub(r"```(?:json)?\n?", "", text)
        text = text.strip()
    if text.endswith("```"):
        text = text[:-3].strip()
    
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON object
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    # Return empty structure
    return {
        "summary": "",
        "strengths": [],
        "weaknesses": [],
        "questions": [],
        "technical_keywords": [],
        "score": None,
        "confidence": None,
    }


async def generate_review_single(
    client: httpx.AsyncClient,
    paper: dict,
    model: str,
    api_key: str,
    temperature: float = 0.3,
) -> LLMReviewResult:
    """Generate a review for a single paper."""
    paper_id = paper.get("paper_id", "unknown")
    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    
    if not abstract:
        return LLMReviewResult(
            paper_id=paper_id,
            title=title,
            model=model,
            summary="",
            strengths=[],
            weaknesses=[],
            questions=[],
            technical_keywords=[],
            score=None,
            confidence=None,
            raw_review="",
            latency_ms=0,
            error="No abstract"
        )
    
    start_time = asyncio.get_event_loop().time()
    
    try:
        response = await client.post(
            TOGETHER_API_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
                        title=title, abstract=abstract
                    )},
                ],
                "temperature": temperature,
                "max_tokens": 1200,
            },
            timeout=90.0,
        )
        response.raise_for_status()
        data = response.json()
        
        content = data["choices"][0]["message"]["content"]
        parsed = parse_review_response(content)
        
        elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        return LLMReviewResult(
            paper_id=paper_id,
            title=title,
            model=model,
            summary=parsed.get("summary", ""),
            strengths=parsed.get("strengths", []),
            weaknesses=parsed.get("weaknesses", []),
            questions=parsed.get("questions", []),
            technical_keywords=[k.lower() for k in parsed.get("technical_keywords", [])],
            score=parsed.get("score"),
            confidence=parsed.get("confidence"),
            raw_review=content,
            latency_ms=elapsed_ms,
            error=None
        )
        
    except Exception as e:
        elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        return LLMReviewResult(
            paper_id=paper_id,
            title=title,
            model=model,
            summary="",
            strengths=[],
            weaknesses=[],
            questions=[],
            technical_keywords=[],
            score=None,
            confidence=None,
            raw_review="",
            latency_ms=elapsed_ms,
            error=str(e)
        )


async def generate_reviews_for_model(
    papers: list[dict],
    model: str,
    api_key: str,
    max_concurrency: int = 5,
    temperature: float = 0.3,
) -> list[LLMReviewResult]:
    """Generate reviews for all papers using a single model."""
    semaphore = asyncio.Semaphore(max_concurrency)
    results = []
    
    async def bounded_generate(client: httpx.AsyncClient, paper: dict) -> LLMReviewResult:
        async with semaphore:
            return await generate_review_single(client, paper, model, api_key, temperature)
    
    async with httpx.AsyncClient() as client:
        tasks = [bounded_generate(client, paper) for paper in papers]
        
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro
            results.append(result)
            
            if (i + 1) % 20 == 0 or (i + 1) == len(papers):
                success = sum(1 for r in results if not r.error)
                print(f"  Progress: {i + 1}/{len(papers)} ({success} success)")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2: Generate LLM reviews from abstracts"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSONL file with papers (paper_id, title, abstract)"
    )
    parser.add_argument(
        "--output-dir", "-o",
        required=True,
        help="Output directory for LLM review results"
    )
    parser.add_argument(
        "--models", "-m",
        help="Comma-separated list of Together AI model IDs"
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=5,
        help="Maximum concurrent API requests per model (default: 5)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.3,
        help="Sampling temperature (default: 0.3 for some creativity)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of papers to process"
    )
    
    args = parser.parse_args()
    
    # Check API key
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        print("Error: TOGETHER_API_KEY environment variable required")
        sys.exit(1)
    
    # Parse models
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    else:
        models = DEFAULT_MODELS
    
    # Load papers
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    papers = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                papers.append(json.loads(line))
    
    if args.limit:
        papers = papers[:args.limit]
    
    print(f"Phase 2: Generate LLM reviews from ABSTRACTS (blind)")
    print(f"=" * 60)
    print(f"Loaded {len(papers)} papers from {input_path}")
    print(f"Models to evaluate: {len(models)}")
    for m in models:
        print(f"  - {m}")
    print()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate reviews for each model
    all_results = []
    start_time = datetime.now(timezone.utc)
    
    for model in models:
        print(f"{'=' * 60}")
        print(f"Generating reviews with: {model}")
        print(f"{'=' * 60}")
        
        model_start = datetime.now(timezone.utc)
        results = asyncio.run(generate_reviews_for_model(
            papers=papers,
            model=model,
            api_key=api_key,
            max_concurrency=args.max_concurrency,
            temperature=args.temperature,
        ))
        model_elapsed = (datetime.now(timezone.utc) - model_start).total_seconds()
        
        success = sum(1 for r in results if not r.error)
        print(f"Completed: {success}/{len(results)} success in {model_elapsed:.1f}s")
        
        all_results.extend(results)
    
    total_elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
    
    # Write all results
    results_path = output_dir / "llm_reviews.jsonl"
    with open(results_path, "w") as f:
        for result in all_results:
            f.write(json.dumps(asdict(result)) + "\n")
    
    # Write metadata
    meta_path = output_dir / "phase2_meta.json"
    success_count = sum(1 for r in all_results if not r.error)
    
    with open(meta_path, "w") as f:
        json.dump({
            "phase": 2,
            "description": "Generate LLM reviews from abstracts (blind)",
            "models": models,
            "papers_processed": len(papers),
            "total_reviews": len(all_results),
            "success_count": success_count,
            "temperature": args.temperature,
            "elapsed_seconds": total_elapsed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)
    
    print()
    print("=" * 60)
    print("PHASE 2 COMPLETE")
    print("=" * 60)
    print(f"Total reviews generated: {len(all_results)}")
    print(f"Success: {success_count}, Failures: {len(all_results) - success_count}")
    print(f"Time elapsed: {total_elapsed:.1f}s")
    print(f"Output: {results_path}")


if __name__ == "__main__":
    main()
