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
from enum import Enum

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

    
class PredictorType(Enum):
    """Maps known DSPy modules for LM requests"""
    CHAIN_OF_THOUGHT = "ChainOfThought"
    REACT_AGENT = "ReAct"
    PREDICT = "Predict"


@dataclass
class TaskConfig:
    """Configuration for a specific task, including both LM and predictor settings"""
    lm_config: LMConfig
    predictor_type: PredictorType = PredictorType.CHAIN_OF_THOUGHT

    def create_lm(self) -> dspy.LM:
        return self.lm_config.create_lm()


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


class LMForTask(Enum):
    SUMMARIZATION = "summarization"
    DOCUMENT_REVIEW = "document_review"
    SECTION_IDENTIFICATION = "section_identification"
    SECTION_REVIEW = "section_review"
    IMAGE_CAPTION_EXTRACTION = "image_caption_extraction"
    CAPTION_ANALYSIS = "caption_analysis"
    CAPTION_COMBINATION = "caption_combination"
    MARKDOWN_SEGMENTATION = "markdown_segmentation"
    STORM_WRITER = "storm_writer"
    STORM_QUESTIONS = "storm_questions"

    def get_lm(self, custom_config: Optional[TaskConfig] = None) -> dspy.LM:
        """Get the language model for this task"""
        config = custom_config if custom_config else DEFAULT_CONFIGS.get(self, DEFAULT_CONFIGS["default"])
        return config.create_lm()

    def get_predictor_type(self, custom_config: Optional[TaskConfig] = None) -> PredictorType:
        """Get the predictor type for this task"""
        config = custom_config if custom_config else DEFAULT_CONFIGS.get(self, DEFAULT_CONFIGS["default"])
        return config.predictor_type

DEFAULT_CONFIGS: Dict[LMForTask, TaskConfig] = {
    LMForTask.SUMMARIZATION: TaskConfig(
        lm_config=Models.Ollama.LLAMA32_VISION,
        predictor_type=PredictorType.CHAIN_OF_THOUGHT
    ),
    LMForTask.DOCUMENT_REVIEW: TaskConfig(
        lm_config=Models.OpenRouter.DEEPSEEK_V3,
        predictor_type=PredictorType.CHAIN_OF_THOUGHT
    ),
    LMForTask.SECTION_IDENTIFICATION: TaskConfig(
        lm_config=Models.OpenRouter.DEEPSEEK_V3,
        predictor_type=PredictorType.CHAIN_OF_THOUGHT
    ),
    LMForTask.SECTION_REVIEW: TaskConfig(
        lm_config=Models.OpenRouter.DEEPSEEK_V3,
        predictor_type=PredictorType.CHAIN_OF_THOUGHT
    ),
    LMForTask.IMAGE_CAPTION_EXTRACTION: TaskConfig(
        lm_config=Models.Ollama.LLAMA32_VISION,
        predictor_type=PredictorType.PREDICT
    ),
    LMForTask.CAPTION_ANALYSIS: TaskConfig(
        lm_config=Models.OpenRouter.DEEPSEEK_V3,
        predictor_type=PredictorType.CHAIN_OF_THOUGHT
    ),
    LMForTask.CAPTION_COMBINATION: TaskConfig(
        lm_config=Models.Ollama.LLAMA32_VISION,
        predictor_type=PredictorType.PREDICT
    ),
    LMForTask.MARKDOWN_SEGMENTATION: TaskConfig(
        lm_config=Models.OpenRouter.DEEPSEEK_V3,
        predictor_type=PredictorType.PREDICT
    ),
    LMForTask.STORM_WRITER: TaskConfig(
        lm_config=Models.OpenRouter.DEEPSEEK_V3,
        predictor_type=PredictorType.CHAIN_OF_THOUGHT
    ),
    LMForTask.STORM_QUESTIONS: TaskConfig(
        lm_config=Models.OpenRouter.DEEPSEEK_V3,
        predictor_type=PredictorType.CHAIN_OF_THOUGHT
    ),
    "default": TaskConfig(
        lm_config=Models.Ollama.LLAMA32_VISION,
        predictor_type=PredictorType.PREDICT
    ),
}