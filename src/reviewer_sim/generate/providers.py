import json
import re
from typing import Dict, Protocol

from reviewer_sim.utils.config import ModelConfig


class BaseGenerator(Protocol):
    def generate(self, example: Dict) -> Dict:
        ...


class MockGenerator:
    """Deterministic mock reviewer that uses reviewer_profile if present."""

    def generate(self, example: Dict) -> Dict:
        title = example.get("title", "Untitled")
        abstract = example.get("abstract", "") or ""

        profile = example.get("reviewer_profile", {}) or {}
        expertise = profile.get("expertise", "general")
        seniority = profile.get("seniority", "unknown")
        tone = (profile.get("tone", "neutral") or "neutral").lower()

        tone_adjust = {"critical": -2, "neutral": 0, "positive": 2}
        score = 6 + tone_adjust.get(tone, 0)
        score = max(1, min(10, score))

        tone_summary = {
            "critical": "The work has potential but significant issues limit confidence.",
            "neutral": "The work is clear and contributes in a modest, understandable way.",
            "positive": "The work is strong and well-motivated with promising results.",
        }.get(tone, "The work is clear and contributes in a modest, understandable way.")

        tone_strengths = {
            "critical": "There is a plausible idea, but the evidence is thin.",
            "neutral": "The motivation is sound and the framing is coherent.",
            "positive": "The motivation is compelling and the framing is strong.",
        }.get(tone, "The motivation is sound and the framing is coherent.")

        tone_weaknesses = {
            "critical": "Key {expertise} details are missing, which hurts credibility.",
            "neutral": "Some {expertise} details are missing from the abstract.",
            "positive": "A few {expertise} details could be clarified in the full paper.",
        }.get(tone, "Some {expertise} details are missing from the abstract.")

        review_text = f"""Summary:
This submission "{title}" is about: {abstract[:400]}
{tone_summary}

Reviewer context:
- Expertise: {expertise}
- Seniority: {seniority}
- Tone: {tone}

Strengths:
- {tone_strengths}
- The approach appears relevant to {expertise} based on the abstract.

Weaknesses:
- {tone_weaknesses.format(expertise=expertise)}
- Evaluation details are unclear or incomplete from the provided abstract.

Questions:
1) What are the main failure cases?
2) How does the method compare to the closest prior work?
3) What ablations support the key claims?

Recommendation:
Overall score: {score}
"""
        return {"text": review_text, "score": int(score)}


class LlamaCppGenerator:
    """Generator using llama-cpp-python with a local GGUF model.
    
    Model is loaded ONCE in __init__ and reused for all generate() calls.
    """

    def __init__(self, config: ModelConfig):
        self.config = config

        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise RuntimeError(
                "llama-cpp-python is not installed. "
                "Install with: pip install llama-cpp-python"
            ) from exc

        if not config.model_path:
            raise ValueError("MODEL_PATH must be set for llamacpp provider.")

        self.llm = Llama(
            model_path=config.model_path,
            n_ctx=config.n_ctx,
            n_gpu_layers=config.n_gpu_layers,
            verbose=False,
        )

    def _build_prompt(self, example: Dict) -> str:
        title = example.get("title", "Untitled")
        abstract = example.get("abstract", "") or ""
        profile = example.get("reviewer_profile", {}) or {}
        expertise = profile.get("expertise", "general")
        seniority = profile.get("seniority", "unknown")
        tone = profile.get("tone", "neutral")

        prompt = f"""{self.config.system_prompt}

Title: {title}

Abstract: {abstract}

Reviewer profile:
- Expertise: {expertise}
- Seniority: {seniority}
- Tone: {tone}

Respond with ONLY a JSON object: {{"text": "your review here", "score": <1-10>}}
"""
        return prompt

    def _parse_json_response(self, content: str) -> Dict:
        content = content.strip()

        # Try direct parse first
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "text" in data:
                return data
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in the response
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
                data = json.loads(content[start : end + 1])
                if isinstance(data, dict) and "text" in data:
                    return data
            except json.JSONDecodeError:
                pass

        # Final fallback: return raw content with score=None
        return {"text": content, "score": None}

    def generate(self, example: Dict) -> Dict:
        prompt = self._build_prompt(example)

        response = self.llm(
            prompt,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens,
            stop=None,
        )
        content = response["choices"][0]["text"]
        return self._parse_json_response(content)


def get_generator(config: ModelConfig) -> BaseGenerator:
    """Factory function to create a generator based on config.provider."""
    provider = (config.provider or "mock").lower()
    if provider == "mock":
        return MockGenerator()
    if provider in ("llamacpp", "llama_cpp"):
        return LlamaCppGenerator(config=config)
    raise ValueError(f"Unknown provider: {config.provider}")
