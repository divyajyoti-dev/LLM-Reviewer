"""
Enrich processed JSONL with LLM-classified primary_area labels.

This script creates a DERIVED dataset - it does NOT modify the original JSONL.
Each record is enriched with:
  - primary_area_llm: one of the allowed labels
  - primary_area_llm_confidence: float 0.0-1.0
  - meta.llm_labeling: provider, model, prompt_version, temperature, timestamp

Environment variables:
  TOGETHER_API_KEY: Your Together AI API key (required)

Example usage:
  export TOGETHER_API_KEY="your-key-here"
  
  # Enrich all records
  python -m reviewer_sim.ingest.enrich_primary_area \\
    --in-path outputs/review_subset.jsonl \\
    --out-path outputs/review_subset_enriched.jsonl \\
    --model "meta-llama/Llama-3-8b-chat-hf"

  # Enrich first 10 records only
  python -m reviewer_sim.ingest.enrich_primary_area \\
    --in-path outputs/review_subset.jsonl \\
    --out-path outputs/review_subset_enriched.jsonl \\
    --limit 10

  # Resume interrupted run
  python -m reviewer_sim.ingest.enrich_primary_area \\
    --in-path outputs/review_subset.jsonl \\
    --out-path outputs/review_subset_enriched.jsonl \\
    --resume
"""

import argparse
import asyncio
import json
import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx


# Allowed labels for primary area classification
ALLOWED_LABELS = frozenset([
    "computer_vision",
    "nlp",
    "ml",
    "robotics",
    "graphics",
    "systems",
    "theory",
    "hci",
    "bio_medical",
    "multi_modal",
    "other",
])

# Together AI API endpoint
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

# Prompt template
SYSTEM_PROMPT = """You are a research paper classifier. Given a paper's title and abstract, classify it into exactly ONE primary research area.

You MUST respond with ONLY a JSON object in this exact format:
{"primary_area_llm": "<label>", "primary_area_llm_confidence": <float>}

Allowed labels (choose exactly one):
- "computer_vision" - image/video processing, object detection, segmentation, 3D vision
- "nlp" - natural language processing, text understanding, language models for text
- "ml" - general machine learning, optimization, learning theory, neural architectures
- "robotics" - robot control, planning, manipulation, autonomous systems
- "graphics" - rendering, animation, 3D modeling, visual synthesis
- "systems" - distributed systems, databases, networks, hardware/software systems
- "theory" - theoretical CS, algorithms, complexity, formal methods
- "hci" - human-computer interaction, user interfaces, accessibility
- "bio_medical" - bioinformatics, medical imaging, drug discovery, health AI
- "multi_modal" - combining vision+language, audio+text, cross-modal learning
- "other" - if none of the above fit or content is ambiguous

Confidence guidelines:
- 0.9-1.0: Very clear fit, obvious category
- 0.7-0.89: Good fit, minor ambiguity
- 0.5-0.69: Reasonable guess, multiple categories could apply
- 0.3-0.49: Low confidence, ambiguous content
- <0.3: Very uncertain, use "other"

If abstract is empty or too short to classify, return {"primary_area_llm": "other", "primary_area_llm_confidence": 0.3}"""

USER_PROMPT_TEMPLATE = """Classify this paper:

Title: {title}

Abstract: {abstract}

Respond with ONLY the JSON object, no other text."""


async def call_together_api(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    title: str,
    abstract: str,
    max_retries: int = 3,
) -> dict:
    """Call Together AI API to classify a paper."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    # Truncate abstract if too long
    abstract_truncated = abstract[:3000] if abstract else ""

    user_prompt = USER_PROMPT_TEMPLATE.format(
        title=title or "Untitled",
        abstract=abstract_truncated or "[No abstract provided]",
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": 100,
    }

    for attempt in range(max_retries):
        try:
            response = await client.post(
                TOGETHER_API_URL,
                headers=headers,
                json=payload,
                timeout=30.0,
            )

            if response.status_code == 429:
                # Rate limited - exponential backoff
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
                continue

            if response.status_code >= 500:
                # Server error - retry
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
                continue

            response.raise_for_status()
            data = response.json()

            content = data["choices"][0]["message"]["content"].strip()
            return parse_llm_response(content)

        except (httpx.HTTPError, KeyError, json.JSONDecodeError) as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
                continue
            # Final failure - return fallback
            return {"primary_area_llm": "other", "primary_area_llm_confidence": 0.3}

    return {"primary_area_llm": "other", "primary_area_llm_confidence": 0.3}


def parse_llm_response(content: str) -> dict:
    """Parse and validate LLM response JSON."""
    # Try to extract JSON from response
    try:
        # First try direct parse
        result = json.loads(content)
    except json.JSONDecodeError:
        # Try to find JSON in the response
        match = re.search(r'\{[^{}]*"primary_area_llm"[^{}]*\}', content)
        if match:
            try:
                result = json.loads(match.group())
            except json.JSONDecodeError:
                return {"primary_area_llm": "other", "primary_area_llm_confidence": 0.3}
        else:
            return {"primary_area_llm": "other", "primary_area_llm_confidence": 0.3}

    # Validate label
    label = result.get("primary_area_llm", "other")
    if label not in ALLOWED_LABELS:
        label = "other"

    # Validate confidence
    try:
        confidence = float(result.get("primary_area_llm_confidence", 0.3))
        confidence = max(0.0, min(1.0, confidence))
    except (ValueError, TypeError):
        confidence = 0.3

    return {"primary_area_llm": label, "primary_area_llm_confidence": confidence}


async def enrich_record(
    client: httpx.AsyncClient,
    api_key: str,
    model: str,
    record: dict,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Enrich a single record with LLM classification."""
    async with semaphore:
        title = record.get("title", "")
        abstract = record.get("abstract", "")

        classification = await call_together_api(
            client=client,
            api_key=api_key,
            model=model,
            title=title,
            abstract=abstract,
        )

        # Build enriched record (preserve original, add new fields)
        enriched = dict(record)
        enriched["primary_area_llm"] = classification["primary_area_llm"]
        enriched["primary_area_llm_confidence"] = classification["primary_area_llm_confidence"]

        # Add/update meta.llm_labeling
        meta = dict(enriched.get("meta", {}))
        meta["llm_labeling"] = {
            "provider": "together",
            "model": model,
            "prompt_version": "area-v1",
            "temperature": 0,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        enriched["meta"] = meta

        return enriched


async def process_batch(
    records: list[dict],
    api_key: str,
    model: str,
    max_concurrency: int,
) -> list[dict]:
    """Process a batch of records with bounded concurrency."""
    semaphore = asyncio.Semaphore(max_concurrency)

    async with httpx.AsyncClient() as client:
        tasks = [
            enrich_record(client, api_key, model, record, semaphore)
            for record in records
        ]
        return await asyncio.gather(*tasks)


def load_existing_paper_ids(out_path: Path) -> set[str]:
    """Load paper_ids already in output file (for resume)."""
    if not out_path.exists():
        return set()

    paper_ids = set()
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    record = json.loads(line)
                    pid = record.get("paper_id")
                    if pid:
                        paper_ids.add(pid)
                except json.JSONDecodeError:
                    continue
    return paper_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enrich JSONL with LLM-classified primary_area labels."
    )
    parser.add_argument(
        "--in-path",
        type=Path,
        required=True,
        help="Input JSONL file path",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        required=True,
        help="Output JSONL file path (enriched)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3-8b-chat-hf",
        help="Together AI model ID",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process first N records only (0 = all)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        help="Maximum concurrent API requests",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip paper_ids already in output file",
    )

    args = parser.parse_args()

    # Check API key
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        raise ValueError("TOGETHER_API_KEY environment variable is required")

    if not args.in_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.in_path}")

    # Load input records
    records = []
    with open(args.in_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    print(f"Loaded {len(records)} records from {args.in_path}")

    # Handle resume
    existing_ids = set()
    if args.resume:
        existing_ids = load_existing_paper_ids(args.out_path)
        if existing_ids:
            print(f"Resume mode: {len(existing_ids)} records already processed")
            records = [r for r in records if r.get("paper_id") not in existing_ids]
            print(f"Remaining to process: {len(records)}")

    # Apply limit
    if args.limit > 0:
        records = records[:args.limit]
        print(f"Limited to first {args.limit} records")

    if not records:
        print("No records to process.")
        return

    print(f"Processing {len(records)} records with model: {args.model}")
    print(f"Max concurrency: {args.max_concurrency}")

    # Process in batches for progress reporting
    batch_size = 50
    all_enriched = []
    label_counts: Counter = Counter()
    failures = 0

    start_time = time.time()

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        enriched_batch = asyncio.run(
            process_batch(batch, api_key, args.model, args.max_concurrency)
        )
        all_enriched.extend(enriched_batch)

        # Count labels
        for rec in enriched_batch:
            label = rec.get("primary_area_llm", "other")
            label_counts[label] += 1
            if label == "other" and rec.get("primary_area_llm_confidence", 0) <= 0.3:
                failures += 1

        processed = min(i + batch_size, len(records))
        elapsed = time.time() - start_time
        rate = processed / elapsed if elapsed > 0 else 0
        print(f"Progress: {processed}/{len(records)} ({rate:.1f} rec/s)")

    # Write output (append if resume, else write fresh)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.resume and existing_ids else "w"

    with open(args.out_path, mode, encoding="utf-8") as f:
        for rec in all_enriched:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 50)
    print("ENRICHMENT SUMMARY")
    print("=" * 50)
    print(f"Total processed: {len(all_enriched)}")
    print(f"Time elapsed: {elapsed:.1f}s")
    print(f"Fallback failures (other with low confidence): {failures}")
    print("\nLabel distribution:")
    for label, count in label_counts.most_common():
        pct = count / len(all_enriched) * 100
        print(f"  {label}: {count} ({pct:.1f}%)")
    print(f"\nOutput written to: {args.out_path}")


if __name__ == "__main__":
    main()
