"""Grounding verification — Google Gemini (cloud) or Ollama (local).

Checks whether LLM-generated answers are actually supported by retrieved
evidence.  Uses the LLM itself as an NLI judge.

Provider is selected by ``settings.provider``.
Call ``get_grounding_checker()`` to obtain the correct instance.
"""

from __future__ import annotations

import json
import re

from loguru import logger

from documind.config import settings
from documind.generation.prompts import GROUNDING_PROMPT

# ── Shared JSON parsing helpers ──────────────────────────────────────────────

def _parse_json_response(response: str) -> dict:
    """Extract and parse JSON from an LLM response (handles markdown fences)."""
    # Try to extract JSON from markdown code fences first
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)```", response, re.DOTALL)
    if fence_match:
        response = fence_match.group(1)
    return json.loads(response.strip())


def _heuristic_grounding(response: str, threshold: float) -> dict:
    """Fallback heuristic when JSON parsing fails."""
    lower = response.lower()
    # Count "not_supported" and "contradicted" FIRST, then subtract from
    # total "supported" matches to avoid double-counting substrings.
    not_supported = len(re.findall(r"not[_ ]supported", lower))
    contradicted = lower.count("contradicted")
    # Count only standalone "supported" (not preceded by "not_" or "un")
    supported = len(re.findall(r"(?<!not[_ ])(?<!un)supported", lower))
    total = supported + not_supported + contradicted
    confidence = (supported / total) if total > 0 else 0.5
    return {
        "overall_grounded": confidence >= threshold,
        "confidence": confidence,
        "claims": [],
        "summary": "Heuristic grounding (JSON parse failed)",
    }


def _evaluate_result(raw_response: str, threshold: float) -> dict:
    """Parse the LLM grounding response and compute is_grounded."""
    try:
        result = _parse_json_response(raw_response)
    except Exception as e:
        logger.warning(f"Failed to parse grounding JSON: {e}")
        result = _heuristic_grounding(raw_response, threshold)

    is_grounded = result.get("confidence", 0) >= threshold
    logger.info(
        f"Grounding check: grounded={is_grounded}, "
        f"confidence={result.get('confidence', 0):.2f}"
    )
    return {
        "is_grounded": is_grounded,
        "confidence": result.get("confidence", 0),
        "claims": result.get("claims", []),
        "summary": result.get("summary", ""),
        "raw_response": raw_response,
    }


# ── Gemini Grounding Checker ─────────────────────────────────────────────────

class GeminiGroundingChecker:
    """Grounding verifier using Google Gemini.

    Uses the same gemini-1.5-flash model as the generator — no extra cost.
    JSON-structured output is reliable with Gemini's instruction following.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        confidence_threshold: float = 0.7,
    ):
        import google.generativeai as genai  # lazy import

        _key = api_key or settings.gemini_api_key
        if not _key:
            raise ValueError(
                "GEMINI_API_KEY is required when provider='gemini'. "
                "Get a free key at https://aistudio.google.com/apikey"
            )
        genai.configure(api_key=_key)
        self._model = genai.GenerativeModel(model or settings.gemini_llm_model)
        self.confidence_threshold = confidence_threshold
        logger.info("GeminiGroundingChecker ready")

    def check_grounding(self, answer: str, evidence: str) -> dict:
        """Check if an answer is grounded in the evidence."""
        prompt = GROUNDING_PROMPT.format(evidence=evidence, answer=answer)
        response = self._model.generate_content(
            prompt,
            generation_config={"temperature": 0.0},
        )
        return _evaluate_result(response.text.strip(), self.confidence_threshold)


# ── Ollama Grounding Checker ─────────────────────────────────────────────────

class OllamaGroundingChecker:
    """Grounding verifier using a local Ollama model."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        confidence_threshold: float = 0.7,
    ):
        from ollama import Client  # lazy import

        self.model = model or settings.ollama_llm_model
        self.base_url = base_url or settings.ollama_base_url
        self.client = Client(host=self.base_url)
        self.confidence_threshold = confidence_threshold
        logger.info(f"OllamaGroundingChecker ready (model={self.model})")

    def check_grounding(self, answer: str, evidence: str) -> dict:
        """Check if an answer is grounded in the evidence."""
        prompt = GROUNDING_PROMPT.format(evidence=evidence, answer=answer)
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        return _evaluate_result(response.message.content.strip(), self.confidence_threshold)


# ── Backward-compat alias & factory ─────────────────────────────────────────

GroundingChecker = GeminiGroundingChecker | OllamaGroundingChecker


def get_grounding_checker() -> GroundingChecker:
    """Factory: return the correct GroundingChecker for the configured provider."""
    if settings.using_gemini:
        return GeminiGroundingChecker()
    return OllamaGroundingChecker()

