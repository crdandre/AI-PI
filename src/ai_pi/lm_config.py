"""
Configuration for language models used in different tasks.
Enables per-task LM specification and fallback to default models.
"""
from dataclasses import dataclass
import dspy
import os

@dataclass
class LMConfig:
    model_name: str
    temperature: float = 0.9
    api_base: str = "https://openrouter.ai/api/v1"
    api_key: str = os.getenv("OPENROUTER_API_KEY")

    def create_lm(self) -> dspy.LM:
        return dspy.LM(
            self.model_name,
            api_base=self.api_base,
            api_key=self.api_key,
            temperature=self.temperature
        )

# Default configurations for different tasks
DEFAULT_CONFIGS = {
    "summarization": LMConfig("openrouter/openai/gpt-4o"),
    "review": LMConfig("openrouter/anthropic/claude-3-opus"),
    #image_caption_extraction must be at least dual-modal (image+text) to take image as input and return text
    "image_caption_extraction": LMConfig("openrouter/openai/gpt-4o"),
    "caption_analysis": LMConfig("openrouter/openai/gpt-4o"),
    "caption_combination": LMConfig("openrouter/openai/gpt-4o"),
    "markdown_segmentation": LMConfig("openrouter/openai/gpt-4o"),
    "default": LMConfig("openrouter/openai/gpt-4o"),
}

def get_lm_for_task(task: str, custom_config: LMConfig = None) -> dspy.LM:
    """Get a language model for a specific task"""
    config = custom_config if custom_config else DEFAULT_CONFIGS.get(task, DEFAULT_CONFIGS["default"])
    return config.create_lm()