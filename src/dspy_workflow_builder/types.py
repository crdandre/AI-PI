"""
Core types and dataclasses for the workflow builder.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Type, Optional, Union

import dspy
from dspy_workflow_builder.parse_lm_config import LMForTask, TaskConfig

@dataclass
class BaseStep:
    """Base configuration for any processing step"""
    step_type: Union[str, Enum]
    processor_class: str  # Store as string instead of Type reference
    output_key: str
    depends_on: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LMStep(BaseStep):
    """Configuration specific to LLM-based processing steps"""
    lm_name: LMForTask
    task_config: Optional[TaskConfig] = None
    signatures: Optional[List[Type[dspy.Signature]]] = None 