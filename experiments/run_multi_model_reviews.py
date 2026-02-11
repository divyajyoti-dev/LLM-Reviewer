"""
Run review generation across multiple models for comparison experiments.

Generates reviews for the same set of papers using different LLM models,
saving structured outputs for downstream evaluation.

Environment variables:
  TOGETHER_API_KEY: API key for Together AI models (required for cloud models)

Example usage:
  # Run with mock + Together AI models
  export TOGETHER_API_KEY="your-key"
  
  python experiments/run_multi_model_reviews.py \
    --input data/processed/review_subset.jsonl \
    --output-dir experiments/outputs/multi_model_run_001 \
    --models "mock,mistralai/Mixtral-8x7B-Instruct-v0.1" \
    --limit 10 \
    --seed 42 \
    --temperature 0

  # Run with local GGUF model
  python experiments/run_multi_model_reviews.py \
    --input data/processed/review_subset.jsonl \
    --output-dir experiments/outputs/run_001 \
    --models "llamacpp:/path/to/model.gguf" \
    --limit 5

Output schema (per line in reviews.jsonl):
  {
    "paper_id": "...",
    "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "review_text": "...",
    "score": 7,
    "meta": {
      "provider": "together",
      "prompt_version": "review-v1",
      "temperature": 0,
      "latency_ms": 1234,
      "token_usage": {"prompt_tokens": 500, "completion_tokens": 200}
    }
  }
"""

import argparse
import asyncio
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import httpx


# System prompt for review generation
SYSTEM_PROMPT = """You are a scientific peer reviewer. Given a paper's title, abstract, and reviewer profile, write a concise, constructive peer review.

Your review should include:
1. A brief summary of the paper's contribution
2. Key strengths (2-3 points)
3. Key weaknesses or areas for improvement (2-3 points)
4. Questions for the authors
5. An overall score from 1-10

Respond with ONLY a valid JSON object with exactly two keys:
- "text": your complete review as a string
- "score": integer from 1 to 10

No markdown, no extra text outside the JSON."""

USER_PROMPT_TEMPLATE = """Review this paper:

Title: {title}

Abstract: {abstract}

Reviewer profile:
- Expertise: {expertise}
- Seniority: {seniority}
- Tone: {tone}

Provide your review as JSON: {{"text": "your review", "score": <1-10>}}"""


@dataclass
class ModelSpec:
    """Parsed model specification."""
    provider: str  # "mock", "together", "llamacpp"
    model_id: str  # model name or path
    display_name: str  # for output/logging


def parse_model_spec(model_str: str) -> ModelSpec:
    """Parse model string into ModelSpec.
    
    Formats:
      - "mock" -> MockGenerator
      - "llamacpp:/path/to/model.gguf" -> local GGUF
      - "provider/model-name" -> Together AI (default for slash-separated)
    """
    model_str = model_str.strip()
    
    if model_str.lower() == "mock":
        return ModelSpec(provider="mock", model_id="mock", display_name="mock")
    
    if model_str.lower().startswith("llamacpp:"):
        path = model_str[len("llamacpp:"):]
        name = Path(path).stem
        return ModelSpec(provider="llamacpp", model_id=path, display_name=f"llamacpp:{name}")
    
    # Default: Together AI model
    return ModelSpec(provider="together", model_id=model_str, display_name=model_str)


def parse_json_response(content: str) -> dict:
    """Robustly parse JSON from LLM response."""
    content = content.strip()
    
    # Try direct parse
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "text" in data:
            return data
    except json.JSONDecodeError:
        pass
    
    # Try to find JSON object in response
    json_match = re.search(r'\{[^{}]*"text"[^{}]*\}', content, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            if isinstance(data, dict) and "text" in data:
                return data
        except json.JSONDecodeError:
            pass
    
    # Fallback: find first { and last }
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(content[start:end + 1])
            if isinstance(data, dict) and "text" in data:
                return data
        except json.JSONDecodeError:
            pass
    
    # Final fallback
    return {"text": content, "score": None}


class MockGenerator:
    """Deterministic mock generator for baseline comparison."""
    
    def generate(self, paper: dict) -> dict:
        title = paper.get("title", "Untitled")
        abstract = paper.get("abstract", "") or ""
        profile = paper.get("reviewer_profile", {}) or {}
        expertise = profile.get("expertise", "general")
        tone = (profile.get("tone", "neutral") or "neutral").lower()
        
        tone_adjust = {"critical": -2, "neutral": 0, "positive": 2}
        score = 6 + tone_adjust.get(tone, 0)
        
        review_text = f"""Summary:
This paper "{title}" addresses: {abstract[:300]}...

Strengths:
- The problem motivation is clear and well-articulated
- The approach shows novelty in the {expertise} domain
- Experimental setup appears reasonable

Weaknesses:
- Some technical details are underspecified
- Comparison with recent baselines could be stronger
- Ablation studies would strengthen the claims

Questions:
1. How does the method scale to larger datasets?
2. What are the main failure modes?

Overall Score: {score}/10"""
        
        return {
            "text": review_text,
            "score": score,
            "meta": {
                "provider": "mock",
                "prompt_version": "review-v1",
                "temperature": 0,
                "latency_ms": 0,
                "token_usage": None,
            }
        }


class TogetherGenerator:
    """Generator using Together AI API."""
    
    def __init__(self, model_id: str, api_key: str, temperature: float = 0):
        self.model_id = model_id
        self.api_key = api_key
        self.temperature = temperature
        self.api_url = "https://api.together.xyz/v1/chat/completions"
    
    async def generate_async(
        self,
        client: httpx.AsyncClient,
        paper: dict,
        max_retries: int = 3,
    ) -> dict:
        """Generate review asynchronously."""
        title = paper.get("title", "Untitled")
        abstract = paper.get("abstract", "") or ""
        profile = paper.get("reviewer_profile", {}) or {}
        expertise = profile.get("expertise", "general")
        seniority = profile.get("seniority", "unknown")
        tone = profile.get("tone", "neutral")
        
        user_prompt = USER_PROMPT_TEMPLATE.format(
            title=title,
            abstract=abstract[:3000],
            expertise=expertise,
            seniority=seniority,
            tone=tone,
        )
        
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "max_tokens": 800,
        }
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        
        start_time = time.time()
        
        for attempt in range(max_retries):
            try:
                response = await client.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=60.0,
                )
                
                if response.status_code == 429:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                
                if response.status_code >= 500:
                    wait_time = 2 ** attempt
                    await asyncio.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                latency_ms = int((time.time() - start_time) * 1000)
                content = data["choices"][0]["message"]["content"].strip()
                parsed = parse_json_response(content)
                
                # Extract token usage if available
                usage = data.get("usage", {})
                token_usage = None
                if usage:
                    token_usage = {
                        "prompt_tokens": usage.get("prompt_tokens"),
                        "completion_tokens": usage.get("completion_tokens"),
                    }
                
                return {
                    "text": parsed.get("text", content),
                    "score": parsed.get("score"),
                    "meta": {
                        "provider": "together",
                        "prompt_version": "review-v1",
                        "temperature": self.temperature,
                        "latency_ms": latency_ms,
                        "token_usage": token_usage,
                    }
                }
            
            except (httpx.HTTPError, KeyError, json.JSONDecodeError) as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                
                latency_ms = int((time.time() - start_time) * 1000)
                return {
                    "text": f"[ERROR: {type(e).__name__}: {e}]",
                    "score": None,
                    "meta": {
                        "provider": "together",
                        "prompt_version": "review-v1",
                        "temperature": self.temperature,
                        "latency_ms": latency_ms,
                        "token_usage": None,
                        "error": str(e),
                    }
                }
        
        latency_ms = int((time.time() - start_time) * 1000)
        return {
            "text": "[ERROR: max retries exceeded]",
            "score": None,
            "meta": {
                "provider": "together",
                "prompt_version": "review-v1",
                "temperature": self.temperature,
                "latency_ms": latency_ms,
                "token_usage": None,
                "error": "max_retries_exceeded",
            }
        }


class LlamaCppGenerator:
    """Generator using local GGUF model via llama-cpp-python."""
    
    def __init__(self, model_path: str, temperature: float = 0):
        self.model_path = model_path
        self.temperature = temperature
        self.llm = None
    
    def _ensure_loaded(self):
        if self.llm is None:
            try:
                from llama_cpp import Llama
            except ImportError as exc:
                raise RuntimeError(
                    "llama-cpp-python not installed. Install with: pip install llama-cpp-python"
                ) from exc
            
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"Model not found: {self.model_path}")
            
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=4096,
                n_gpu_layers=-1,
                verbose=False,
            )
    
    def generate(self, paper: dict) -> dict:
        """Generate review synchronously."""
        self._ensure_loaded()
        
        title = paper.get("title", "Untitled")
        abstract = paper.get("abstract", "") or ""
        profile = paper.get("reviewer_profile", {}) or {}
        expertise = profile.get("expertise", "general")
        seniority = profile.get("seniority", "unknown")
        tone = profile.get("tone", "neutral")
        
        prompt = f"""{SYSTEM_PROMPT}

Title: {title}

Abstract: {abstract[:3000]}

Reviewer profile:
- Expertise: {expertise}
- Seniority: {seniority}
- Tone: {tone}

Respond with ONLY a JSON object: {{"text": "your review", "score": <1-10>}}
"""
        
        start_time = time.time()
        
        try:
            response = self.llm(
                prompt,
                temperature=self.temperature,
                top_p=0.95,
                max_tokens=800,
                stop=None,
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            content = response["choices"][0]["text"]
            parsed = parse_json_response(content)
            
            usage = response.get("usage", {})
            token_usage = None
            if usage:
                token_usage = {
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                }
            
            return {
                "text": parsed.get("text", content),
                "score": parsed.get("score"),
                "meta": {
                    "provider": "llamacpp",
                    "prompt_version": "review-v1",
                    "temperature": self.temperature,
                    "latency_ms": latency_ms,
                    "token_usage": token_usage,
                }
            }
        
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            return {
                "text": f"[ERROR: {type(e).__name__}: {e}]",
                "score": None,
                "meta": {
                    "provider": "llamacpp",
                    "prompt_version": "review-v1",
                    "temperature": self.temperature,
                    "latency_ms": latency_ms,
                    "token_usage": None,
                    "error": str(e),
                }
            }


async def run_together_model(
    papers: list[dict],
    model_spec: ModelSpec,
    api_key: str,
    temperature: float,
    max_concurrency: int = 4,
) -> list[dict]:
    """Run Together AI model on all papers with bounded concurrency."""
    generator = TogetherGenerator(
        model_id=model_spec.model_id,
        api_key=api_key,
        temperature=temperature,
    )
    
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def process_one(client: httpx.AsyncClient, paper: dict) -> dict:
        async with semaphore:
            result = await generator.generate_async(client, paper)
            return {
                "paper_id": paper.get("paper_id"),
                "model": model_spec.display_name,
                "review_text": result["text"],
                "score": result.get("score"),
                "meta": result["meta"],
            }
    
    async with httpx.AsyncClient() as client:
        tasks = [process_one(client, paper) for paper in papers]
        return await asyncio.gather(*tasks)


def run_sync_model(
    papers: list[dict],
    model_spec: ModelSpec,
    temperature: float,
) -> list[dict]:
    """Run synchronous model (mock or llamacpp) on all papers."""
    if model_spec.provider == "mock":
        generator = MockGenerator()
    elif model_spec.provider == "llamacpp":
        generator = LlamaCppGenerator(model_spec.model_id, temperature)
    else:
        raise ValueError(f"Unknown sync provider: {model_spec.provider}")
    
    results = []
    for paper in papers:
        try:
            result = generator.generate(paper)
            results.append({
                "paper_id": paper.get("paper_id"),
                "model": model_spec.display_name,
                "review_text": result["text"],
                "score": result.get("score"),
                "meta": result["meta"],
            })
        except Exception as e:
            print(f"  Error on paper_id={paper.get('paper_id')}: {type(e).__name__}: {e}")
            results.append({
                "paper_id": paper.get("paper_id"),
                "model": model_spec.display_name,
                "review_text": f"[ERROR: {type(e).__name__}: {e}]",
                "score": None,
                "meta": {"provider": model_spec.provider, "error": str(e)},
            })
    
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run multi-model review generation experiment."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL file with papers",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for results",
    )
    parser.add_argument(
        "--models",
        type=str,
        required=True,
        help="Comma-separated list of models (e.g., 'mock,mistralai/Mixtral-8x7B-Instruct-v0.1')",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Process first N papers only (0 = all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic sampling",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0,
        help="Sampling temperature (0 = deterministic)",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=4,
        help="Max concurrent API requests for cloud models",
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    # Parse models
    model_strs = [m.strip() for m in args.models.split(",") if m.strip()]
    if not model_strs:
        raise ValueError("No models specified")
    
    model_specs = [parse_model_spec(m) for m in model_strs]
    
    # Check API key for Together models
    api_key = os.environ.get("TOGETHER_API_KEY", "")
    together_models = [m for m in model_specs if m.provider == "together"]
    if together_models and not api_key:
        raise ValueError(
            f"TOGETHER_API_KEY required for models: {[m.display_name for m in together_models]}"
        )
    
    # Load papers
    papers = []
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(json.loads(line))
    
    print(f"Loaded {len(papers)} papers from {args.input}")
    
    # Apply limit
    if args.limit > 0:
        papers = papers[:args.limit]
        print(f"Limited to first {args.limit} papers")
    
    # Setup output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / "reviews.jsonl"
    
    # Run each model
    all_results = []
    
    for model_spec in model_specs:
        print(f"\n{'='*60}")
        print(f"Running model: {model_spec.display_name}")
        print(f"Provider: {model_spec.provider}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        if model_spec.provider == "together":
            results = asyncio.run(
                run_together_model(
                    papers,
                    model_spec,
                    api_key,
                    args.temperature,
                    args.max_concurrency,
                )
            )
        else:
            results = run_sync_model(papers, model_spec, args.temperature)
        
        elapsed = time.time() - start_time
        
        # Count successes/failures
        successes = sum(1 for r in results if not r.get("meta", {}).get("error"))
        failures = len(results) - successes
        
        print(f"Completed: {successes} success, {failures} failures in {elapsed:.1f}s")
        
        all_results.extend(results)
    
    # Write all results
    with open(output_file, "w", encoding="utf-8") as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    # Write run metadata
    run_meta = {
        "input_file": str(args.input),
        "models": [m.display_name for m in model_specs],
        "num_papers": len(papers),
        "temperature": args.temperature,
        "seed": args.seed,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "total_results": len(all_results),
    }
    
    meta_file = args.output_dir / "run_meta.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)
    
    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}")
    print(f"Total results: {len(all_results)}")
    print(f"Output: {output_file}")
    print(f"Metadata: {meta_file}")


if __name__ == "__main__":
    main()
