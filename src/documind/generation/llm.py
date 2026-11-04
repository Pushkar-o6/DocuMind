"""LLM answer generation — Google Gemini (cloud) or Ollama (local).

Provider is selected by ``settings.provider``:
  - "gemini"  → GeminiLLMGenerator  (gemini-1.5-flash, free tier, ~10× faster)
  - "ollama"  → OllamaLLMGenerator  (requires local Ollama)

Call ``get_llm_generator()`` to obtain the correct instance automatically.
"""

from __future__ import annotations

from loguru import logger

from documind.config import settings
from documind.generation.prompts import (
    CLASSIFY_PROMPT,
    RAG_SYSTEM_PROMPT,
    RAG_USER_PROMPT,
    REFORMULATE_PROMPT,
    TRANSFORM_QUERY_PROMPT,
)
from documind.retrieval.hybrid import HybridResult

# ── Shared context-builder (used by both providers) ─────────────────────────

def _build_context_and_citations(results: list[HybridResult]) -> tuple[str, list[dict]]:
    """Build the context string and citations list from retrieval results.

    Applies:
    1. Small-to-big retrieval: Uses rich parent_text section if available,
       falling back to chunk text.
    2. Lost-in-the-middle context layout: Places the most relevant chunk
       at the beginning, second-most at the end, and remaining in the middle.
    3. Untrusted document boundaries: Encloses external text inside explicit
       <document_context> tags to prevent prompt injection.
    """
    if not results:
        return "", []

    # Lost-in-the-middle reordering: [Best, 3rd, ..., 4th, 2nd]
    if len(results) > 2:
        reordered: list[HybridResult] = []
        left = True
        temp_candidates = list(results)
        # Alternate adding from front and back
        while temp_candidates:
            if left:
                reordered.append(temp_candidates.pop(0))
            else:
                reordered.append(temp_candidates.pop(-1))
            left = not left
    else:
        reordered = list(results)

    context_parts = []
    citations = []

    for result in reordered:
        prefix = f"[Page {result.page_number}]"
        if result.block_type == "table":
            prefix += " [Table]"
        if result.section_heading:
            prefix += f" [Section: {result.section_heading}]"

        # Small-to-big: Provide surrounding parent section context if present
        body_text = result.parent_text if getattr(result, "parent_text", "") else result.text

        context_parts.append(
            f"<document_chunk page='{result.page_number}' source='{result.source_file}'>\n"
            f"{prefix}\n{body_text}\n"
            f"</document_chunk>"
        )

        citations.append({
            "page": result.page_number,
            "block_type": result.block_type,
            "source_file": result.source_file,
            "section_heading": getattr(result, "section_heading", ""),
            "snippet": result.text[:200],
            "bbox": list(result.bbox) if result.bbox else None,
        })

    wrapped_context = (
        "<document_context>\n"
        + "\n\n".join(context_parts)
        + "\n</document_context>"
    )
    return wrapped_context, citations


# ── Gemini LLM Generator ─────────────────────────────────────────────────────

class GeminiLLMGenerator:
    """Answer generator using Google Gemini 1.5 Flash.

    Free tier: 1 500 requests/day, 1 M tokens/minute.
    Typical latency: 1–3 s for RAG-length prompts — far faster than local models.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ):
        import google.generativeai as genai  # lazy import

        _key = api_key or settings.gemini_api_key
        if not _key:
            raise ValueError(
                "GEMINI_API_KEY is required when provider='gemini'. "
                "Get a free key at https://aistudio.google.com/apikey"
            )
        genai.configure(api_key=_key)
        self._model_name = model or settings.gemini_llm_model
        self._model = genai.GenerativeModel(self._model_name)
        logger.info(f"GeminiLLMGenerator ready (model={self._model_name})")

    def _chat(self, prompt: str, temperature: float = 0.1) -> str:
        """Send a single-turn prompt and return the text response."""
        response = self._model.generate_content(
            prompt,
            generation_config={"temperature": temperature},
        )
        return response.text

    def generate_answer(
        self,
        question: str,
        results: list[HybridResult],
        temperature: float = 0.1,
    ) -> dict:
        """Generate a grounded answer from retrieved context."""
        context, citations = _build_context_and_citations(results)
        user_prompt = RAG_USER_PROMPT.format(context=context, question=question)
        full_prompt = f"{RAG_SYSTEM_PROMPT}\n\n{user_prompt}"

        answer = self._chat(full_prompt, temperature=temperature)
        logger.info(f"Gemini generated answer ({len(answer)} chars) from {len(results)} chunks")
        return {"answer": answer, "citations": citations, "context_used": context}

    def classify_query(self, question: str) -> str:
        """Classify a query as text, visual, or mixed."""
        raw = self._chat(CLASSIFY_PROMPT.format(question=question), temperature=0.0)
        classification = raw.strip().lower()
        if "visual" in classification:
            return "visual"
        if "mixed" in classification:
            return "mixed"
        return "text"

    def reformulate_query(self, original_question: str) -> str:
        """Reformulate a failed query for re-retrieval."""
        reformulated = self._chat(
            REFORMULATE_PROMPT.format(original_question=original_question),
            temperature=0.3,
        ).strip()
        logger.info(f"Gemini reformulated: '{original_question}' → '{reformulated}'")
        return reformulated

    def transform_query(self, question: str) -> str:
        """Rewrite and expand raw user query prior to retrieval."""
        try:
            transformed = self._chat(
                TRANSFORM_QUERY_PROMPT.format(question=question),
                temperature=0.1,
            ).strip()
            logger.info(f"Gemini query transformed: '{question}' → '{transformed}'")
            return transformed
        except Exception as e:
            logger.warning(f"Query transformation failed ({e}), using original query")
            return question


# ── Ollama LLM Generator ─────────────────────────────────────────────────────

class OllamaLLMGenerator:
    """Answer generator using a locally-running Ollama model.

    Requires: ollama server running + LLM model pulled (e.g. llama3.2:3b).
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
    ):
        from ollama import Client  # lazy import

        self.model = model or settings.ollama_llm_model
        self.base_url = base_url or settings.ollama_base_url
        self.client = Client(host=self.base_url)
        logger.info(f"OllamaLLMGenerator ready (model={self.model})")

    def generate_answer(
        self,
        question: str,
        results: list[HybridResult],
        temperature: float = 0.1,
    ) -> dict:
        context, citations = _build_context_and_citations(results)
        user_prompt = RAG_USER_PROMPT.format(context=context, question=question)

        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            options={"temperature": temperature},
        )
        answer = response.message.content
        logger.info(f"Ollama generated answer ({len(answer)} chars) from {len(results)} chunks")
        return {"answer": answer, "citations": citations, "context_used": context}

    def classify_query(self, question: str) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": CLASSIFY_PROMPT.format(question=question)}],
            options={"temperature": 0.0},
        )
        classification = response.message.content.strip().lower()
        if "visual" in classification:
            return "visual"
        if "mixed" in classification:
            return "mixed"
        return "text"

    def reformulate_query(self, original_question: str) -> str:
        response = self.client.chat(
            model=self.model,
            messages=[{
                "role": "user",
                "content": REFORMULATE_PROMPT.format(original_question=original_question),
            }],
            options={"temperature": 0.3},
        )
        reformulated = response.message.content.strip()
        logger.info(f"Ollama reformulated: '{original_question}' → '{reformulated}'")
        return reformulated

    def transform_query(self, question: str) -> str:
        """Rewrite and expand raw user query with local LLM prior to retrieval."""
        try:
            response = self.client.chat(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": TRANSFORM_QUERY_PROMPT.format(question=question),
                }],
                options={"temperature": 0.1},
            )
            transformed = response.message.content.strip()

            # Handle 3B model conversational artifacts or refusals
            refusal_triggers = [
                "i can't", "i cannot", "i am sorry", "i'm sorry", "as an ai",
                "here is the rewritten", "rewritten search query", "i made the following"
            ]
            lower = transformed.lower()
            if any(trig in lower for trig in refusal_triggers) or len(transformed.splitlines()) > 3:
                # If the LLM wrapped the query in quotation marks within commentary, extract it
                import re
                quotes = re.findall(r'"([^"]+)"', transformed)
                if quotes and not any(trig in quotes[0].lower() for trig in refusal_triggers):
                    clean_query = quotes[0].strip()
                    logger.info(f"Ollama extracted search query from explanation: '{clean_query}'")
                    return clean_query
                logger.info(f"Ollama rewrite triggered refusal/verbose commentary; falling back to: '{question}'")
                return question

            logger.info(f"Ollama query transformed: '{question}' → '{transformed}'")
            return transformed
        except Exception as e:
            logger.warning(f"Ollama query transformation failed ({e}), using original query")
            return question


# ── Backward-compat alias & factory ─────────────────────────────────────────

LLMGenerator = GeminiLLMGenerator | OllamaLLMGenerator


def get_llm_generator() -> LLMGenerator:
    """Factory: return the correct LLMGenerator for the configured provider."""
    if settings.using_gemini:
        return GeminiLLMGenerator()
    return OllamaLLMGenerator()

