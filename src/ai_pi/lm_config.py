"""
Configuration for language models used in different tasks.
Enables per-task LM specification and fallback to default models.

Notes:
image_caption_extraction must be a multimodal model such as 4o, llama3.2-vision
"""
from dataclasses import dataclass
from typing import Dict, Optional
import dspy
import os

@dataclass
class LMConfig:
    model_name: str
    temperature: float = 0.9
    api_base: str = "https://openrouter.ai/api/v1"
    api_key: str = os.getenv("OPENROUTER_API_KEY")
    max_tokens: Optional[int] = None

    def create_lm(self) -> dspy.LM:
        params = {
            "model": self.model_name,
            "api_base": self.api_base,
            "api_key": self.api_key,
            "temperature": self.temperature,
            **({"max_tokens": self.max_tokens} if self.max_tokens is not None else {})
        }
        return dspy.LM(**params)


class Models:
    class OpenRouter:
        O1_PREVIEW = LMConfig(model_name="openrouter/openai/o1-preview", temperature=1.0, max_tokens=12000)
        GPT4_O = LMConfig("openrouter/openai/gpt-4o")
        GPT4_O_MINI = LMConfig("openrouter/openai/gpt-4o-mini")
        CLAUDE_SONNET = LMConfig("openrouter/anthropic/claude-3.5-sonnet:beta")
        CLAUDE_OPUS = LMConfig("openrouter/anthropic/claude-3-opus:beta")
        DEEPSEEK_V3 = LMConfig("openrouter/deepseek/deepseek-chat")
        GEMINI_PRO = LMConfig("openrouter/google/learnlm-1.5-pro-experimental:free")

    class Ollama:
        QWQ = LMConfig(model_name="ollama_chat/qwq:latest", api_base="http://localhost:11434/")
        LLAMA31_7_8B = LMConfig(model_name="ollama_chat/llama3.1:latest", api_base="http://localhost:11434/")
        LLAMA32_VISION = LMConfig(model_name="ollama_chat/llama3.2-vision:latest", api_base="http://localhost:11434/")
        LLAMA33_70B = LMConfig(model_name="ollama_chat/llama3.3:latest", api_base="http://localhost:11434/")


DEFAULT_CONFIGS: Dict[str, LMConfig] = {
    "summarization": Models.OpenRouter.DEEPSEEK_V3,
    "document_review": Models.OpenRouter.DEEPSEEK_V3,
    "section_review": Models.OpenRouter.DEEPSEEK_V3,
    "image_caption_extraction": Models.Ollama.LLAMA32_VISION,
    "caption_analysis": Models.OpenRouter.DEEPSEEK_V3,
    "caption_combination": Models.Ollama.LLAMA32_VISION,
    "markdown_segmentation": Models.OpenRouter.DEEPSEEK_V3,
    "storm_writer": Models.OpenRouter.DEEPSEEK_V3,
    "storm_questions": Models.OpenRouter.DEEPSEEK_V3,
    "default": Models.Ollama.LLAMA32_VISION,
}


def get_lm_for_task(task: str, custom_config: Optional[LMConfig] = None) -> dspy.LM:
    """Get a language model for a specific task"""
    config = custom_config if custom_config else DEFAULT_CONFIGS.get(task, DEFAULT_CONFIGS["default"])
    return config.create_lm()