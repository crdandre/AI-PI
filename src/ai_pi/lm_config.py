"""
Configuration for language models used in different tasks.
Enables per-task LM specification and fallback to default models.

Notes:
image_caption_extraction must be a multimodal model such as 4o, llama3.2-vision
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

#ADD AS NEEDED        
OPENROUTER_4O = LMConfig("openrouter/openai/gpt-4o")
OPENROUTER_4O_MINI = LMConfig("openrouter/openai/gpt-4o-mini")
OPENROUTER_SONNET = LMConfig("openrouter/anthropic/claude-3.5-sonnet:beta")
OPENROUTER_OPUS = LMConfig("openrouter/anthropic/claude-3-opus:beta")
OPENROUTER_DEEPSEEK_V3 = LMConfig("openrouter/deepseek/deepseek-chat")
OPENROUTER_GEMINI_PRO = LMConfig("openrouter/google/learnlm-1.5-pro-experimental:free")
OLLAMA_QWQ = LMConfig(model_name="ollama_chat/qwq:latest", api_base="http://localhost:11434/")
OLLAMA_LLAMA31_7_8B = LMConfig(model_name="ollama_chat/llama3.1:latest", api_base="http://localhost:11434/")
OLLAMA_LLAMA32_VISION = LMConfig(model_name="ollama_chat/llama3.2-vision:latest", api_base="http://localhost:11434/")
OLLAMA_LLAMA33_70B = LMConfig(model_name="ollama_chat/llama3.3:latest", api_base="http://localhost:11434/")

DEFAULT_CONFIGS = {
    "summarization":OPENROUTER_DEEPSEEK_V3,
    "review": OPENROUTER_4O,
    "image_caption_extraction": OLLAMA_LLAMA32_VISION,
    "caption_analysis": OPENROUTER_DEEPSEEK_V3,
    "caption_combination": OPENROUTER_DEEPSEEK_V3,
    "markdown_segmentation": OPENROUTER_DEEPSEEK_V3,
    "default": OPENROUTER_DEEPSEEK_V3,
}

def get_lm_for_task(task: str, custom_config: LMConfig = None) -> dspy.LM:
    """Get a language model for a specific task"""
    config = custom_config if custom_config else DEFAULT_CONFIGS.get(task, DEFAULT_CONFIGS["default"])
    return config.create_lm()