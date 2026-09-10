"""
DeepEval expects an OpenAI-compatible judge model by default.
This wraps a dedicated Groq model (independent from the app's shared
agent model) so evaluation runs on the same free-tier setup as the
rest of the project - no separate paid key.

Uses its own instance rather than app.llm.get_llm()'s cached
singleton, because DeepEval's metrics ask for long, verbose JSON
(e.g. extracting every atomic claim from a brief) that needs more
output tokens than the agents' default settings allow.
"""
import os
import time
from langchain_groq import ChatGroq
from groq import RateLimitError
from deepeval.models import DeepEvalBaseLLM


class GroqJudge(DeepEvalBaseLLM):
    def __init__(self):
        self.model = ChatGroq(
            model="openai/gpt-oss-20b",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
            max_tokens=4096,  # DeepEval's claim-extraction JSON can be long
        )

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        for attempt in range(3):
            try:
                content = self.load_model().invoke(prompt).content
                if content.strip():
                    return content
            except RateLimitError:
                pass
            time.sleep(10 * (attempt + 1))
        return "{}"  # last resort so JSON parsing fails gracefully, not silently

    async def a_generate(self, prompt: str) -> str:
        for attempt in range(3):
            try:
                result = await self.load_model().ainvoke(prompt)
                if result.content.strip():
                    return result.content
            except RateLimitError:
                pass
            time.sleep(10 * (attempt + 1))
        return "{}"

    def get_model_name(self) -> str:
        return "Groq openai/gpt-oss-20b (judge)"
