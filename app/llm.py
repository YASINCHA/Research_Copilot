"""
Single place that talks to the model. Swap providers here without
touching any agent code. Start with OpenAI (or an OpenAI-compatible
endpoint); switch to Ollama later for the "fully local" story if
you want that on your CV too.
"""
"""
Single place that talks to the model. Swap providers here without
touching any agent code.

Deliberately NOT cached as a module-level singleton: a shared,
mutable client reused across agents with different structured-output
schemas (ResearchScope, ResearchFindings, CriticVerdict) caused tool
-binding state to leak between calls once agents ran through
LangGraph's async execution. A fresh, cheap client per call avoids
that class of bug entirely.
"""
import os
import asyncio
import logging
from langchain_groq import ChatGroq
from groq import RateLimitError, APIConnectionError

logger = logging.getLogger("research_copilot")


def get_llm(temperature: float = 0.2):
    # Swap to Ollama later for a fully local story:
    # from langchain_community.chat_models import ChatOllama
    # return ChatOllama(model="llama3.2", temperature=temperature)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add "
            "a free key from https://console.groq.com/keys"
        )
    return ChatGroq(
        model="openai/gpt-oss-20b",
        api_key=api_key,
        temperature=temperature,
    )


async def safe_ainvoke(llm, messages, max_retries: int = 3, base_delay: float = 10.0):
    """Shared retry/backoff wrapper used by every agent. Handles the
    two transient failure modes seen in practice: Groq's free-tier
    rate limit (429) and occasional connection hiccups. Without this,
    a single blip anywhere in the pipeline kills the whole run."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return await llm.ainvoke(messages)
        except (RateLimitError, APIConnectionError) as e:
            last_error = e
            logger.warning(
                "LLM call failed (attempt %d/%d): %s - retrying in %.0fs",
                attempt + 1, max_retries, e, base_delay * (attempt + 1),
            )
            await asyncio.sleep(base_delay * (attempt + 1))
        except Exception as e:
            # Malformed JSON / schema mismatch from the model - worth
            # one immediate retry, but don't backoff for this class.
            last_error = e
            logger.warning("LLM call failed (attempt %d/%d): %s", attempt + 1, max_retries, e)
    raise last_error
