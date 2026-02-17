#!/usr/bin/env python3
"""
Phase 1: Extract canonical keywords from HUMAN reviews.

This creates ground truth keywords representing what human reviewers
found important/salient about each paper.
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

SYSTEM_PROMPT = """You are an expert at extracting key technical points from academic peer reviews.

Your task is to identify the canonical keywords and key claims that represent what the reviewer found IMPORTANT about this paper - both positive and negative points.

Guidelines:
- Extract 10-20 keywords/phrases that capture the reviewer's main concerns
- Include: methods critiqued, metrics mentioned, experimental concerns, theoretical issues, missing comparisons, praised innovations
- Use canonical/standard ML/AI terminology
- Include both strengths and weaknesses the reviewer emphasized
- Be specific: "ablation study missing" not just "experiments"
- Output ONLY a JSON object with this format:
  {"keywords": ["keyword1", "keyword2", ...], "strengths": ["str1", ...], "weaknesses": ["weak1", ...]}"""

USER_PROMPT_TEMPLATE = """Extract the key technical points from this peer review:

Paper Title: {title}

Human Review:
{review}

Output only JSON: {{"keywords": [...], "strengths": [...], "weaknesses": [...]}}"""


@dataclass
class HumanKeywordResult:
    paper_id: str
    title: str
    keywords: list[str]
    strengths: list[str]
    weaknesses: list[str]
    extraction_time_ms: float
    model: str
    error: Optional[str] = None


def parse_json_response(text: str) -> dict:
    """Robustly parse JSON from LLM response."""
    text = text.strip()
    
    # Remove markdown code blocks if present
    if text.startswith("```"):
        text = re.sub(r"```(?:json)?\n?", "", text)
        text = text.strip()
    
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON object in text
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    
    return {"keywords": [], "strengths": [], "weaknesses": []}


async def extract_keywords_single(
    client: httpx.AsyncClient,
    paper: dict,
    model: str,
    api_key: str,
    temperature: float = 0,
) -> HumanKeywordResult:
    """Extract keywords from a single human review."""
    paper_id = paper.get("paper_id", "unknown")
    title = paper.get("title", "")
    review = paper.get("review", {})
    
    # Get human review text
    if isinstance(review, dict):
        review_text = review.get("main_review", "")
    else:
        review_text = str(review)
    
    if not review_text:
        return HumanKeywordResult(
            paper_id=paper_id,
            title=title,
            keywords=[],
            strengths=[],
            weaknesses=[],
            extraction_time_ms=0,
            model=model,
            error="No review text"
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
                        title=title, review=review_text
                    )},
                ],
                "temperature": temperature,
                "max_tokens": 800,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()
        
        content = data["choices"][0]["message"]["content"]
        parsed = parse_json_response(content)
        
        keywords = [k.lower().strip() for k in parsed.get("keywords", []) if k]
        strengths = [s.strip() for s in parsed.get("strengths", []) if s]
        weaknesses = [w.strip() for w in parsed.get("weaknesses", []) if w]
        
        elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        
        return HumanKeywordResult(
            paper_id=paper_id,
            title=title,
            keywords=keywords,
            strengths=strengths,
            weaknesses=weaknesses,
            extraction_time_ms=elapsed_ms,
            model=model,
            error=None if keywords else "Failed to parse keywords"
        )
        
    except Exception as e:
        elapsed_ms = (asyncio.get_event_loop().time() - start_time) * 1000
        return HumanKeywordResult(
            paper_id=paper_id,
            title=title,
            keywords=[],
            strengths=[],
            weaknesses=[],
            extraction_time_ms=elapsed_ms,
            model=model,
            error=str(e)
        )


async def extract_all_keywords(
    papers: list[dict],
    model: str,
    api_key: str,
    max_concurrency: int = 5,
    temperature: float = 0,
) -> list[HumanKeywordResult]:
    """Extract keywords for all papers with concurrency control."""
    semaphore = asyncio.Semaphore(max_concurrency)
    results = []
    
    async def bounded_extract(client: httpx.AsyncClient, paper: dict) -> HumanKeywordResult:
        async with semaphore:
            return await extract_keywords_single(client, paper, model, api_key, temperature)
    
    async with httpx.AsyncClient() as client:
        tasks = [bounded_extract(client, paper) for paper in papers]
        
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            result = await coro
            results.append(result)
            
            if (i + 1) % 10 == 0 or (i + 1) == len(papers):
                success = sum(1 for r in results if not r.error)
                print(f"Progress: {i + 1}/{len(papers)} ({success} success)")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Phase 1: Extract canonical keywords from human reviews"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input JSONL file with papers and human reviews"
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output JSONL file for human keyword results"
    )
    parser.add_argument(
        "--model",
        default="deepseek-ai/DeepSeek-V3",
        help="Together AI model ID for extraction (default: deepseek-ai/DeepSeek-V3)"
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=5,
        help="Maximum concurrent API requests (default: 5)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Sampling temperature (default: 0)"
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
    
    print(f"Phase 1: Extract keywords from HUMAN reviews")
    print(f"=" * 60)
    print(f"Loaded {len(papers)} papers from {input_path}")
    print(f"Model: {args.model}")
    print(f"Max concurrency: {args.max_concurrency}")
    print()
    
    # Extract keywords
    start_time = datetime.now(timezone.utc)
    results = asyncio.run(extract_all_keywords(
        papers=papers,
        model=args.model,
        api_key=api_key,
        max_concurrency=args.max_concurrency,
        temperature=args.temperature,
    ))
    elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
    
    # Write results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for result in results:
            f.write(json.dumps(asdict(result)) + "\n")
    
    # Summary
    success_count = sum(1 for r in results if not r.error)
    total_keywords = sum(len(r.keywords) for r in results)
    total_strengths = sum(len(r.strengths) for r in results)
    total_weaknesses = sum(len(r.weaknesses) for r in results)
    
    print()
    print("=" * 60)
    print("PHASE 1 COMPLETE")
    print("=" * 60)
    print(f"Papers processed: {len(results)}")
    print(f"Success: {success_count}, Failures: {len(results) - success_count}")
    print(f"Total keywords: {total_keywords} (avg: {total_keywords/len(results):.1f})")
    print(f"Total strengths: {total_strengths}")
    print(f"Total weaknesses: {total_weaknesses}")
    print(f"Time elapsed: {elapsed:.1f}s")
    print(f"Output: {output_path}")
    
    # Write metadata
    meta_path = output_path.with_suffix(".meta.json")
    with open(meta_path, "w") as f:
        json.dump({
            "phase": 1,
            "description": "Extract canonical keywords from human reviews",
            "model": args.model,
            "input_file": str(input_path),
            "papers_processed": len(results),
            "success_count": success_count,
            "total_keywords": total_keywords,
            "total_strengths": total_strengths,
            "total_weaknesses": total_weaknesses,
            "elapsed_seconds": elapsed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, f, indent=2)


if __name__ == "__main__":
    main()
