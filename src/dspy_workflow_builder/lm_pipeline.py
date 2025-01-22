"""
Core building blocks for structuring LLM-based workflows.
This module provides the base classes and utilities for creating modular,
configurable pipelines for any LLM-based processing task.
"""

from dataclasses import dataclass, field
from enum import Enum
from inspect import isclass
import logging
from typing import List, Type, Dict, Any, Optional, Union

import dspy
from dspy_workflow_builder.parse_lm_config import LMForTask, TaskConfig
from dspy_workflow_builder.utils.logging import log_step

@dataclass
class BaseStep:
    """Base configuration for any processing step"""
    step_type: Union[str, Enum]
    processor_class: Type["BaseProcessor"]
    output_key: str
    depends_on: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)

@dataclass
class LLMStep(BaseStep):
    """Configuration specific to LLM-based processing steps"""
    lm_name: LMForTask
    task_config: Optional[TaskConfig] = None
    signatures: Optional[List[Type[dspy.Signature]]] = None
    
    def __post_init__(self):
        """Use provided signatures or auto-discover from processor class"""
        if self.signatures is None:
            self.signatures = [
                member for _, member in vars(self.processor_class).items()
                if isclass(member) and issubclass(member, dspy.Signature) 
                and member != dspy.Signature
            ]
        if not self.signatures:
            raise ValueError(f"No Signature classes found for {self.processor_class.__name__}")

@dataclass
class ComputeStep(BaseStep):
    """Configuration for non-LLM processing steps"""
    pass

@dataclass
class PipelineConfig:
    """Configuration for the entire processing pipeline"""
    steps: List[Union[LLMStep, ComputeStep]]
    verbose: bool = False
    validation: bool = True

class BaseProcessor:
    """Base class for all processors"""
    def __init__(self, step: BaseStep):
        self.step = step
        self.logger = logging.getLogger(f"processor.{step.step_type}")

    @log_step()
    def process(self, data: dict) -> dict:
        self._validate_dependencies(data)
        return self._process(data)

    def _process(self, data: dict) -> dict:
        raise NotImplementedError()
        
    def _validate_dependencies(self, data: dict) -> None:
        """Validate that declared dependencies exist in data"""
        if not self.step.depends_on:
            return
            
        missing = [dep for dep in self.step.depends_on if dep not in data]
        if missing:
            raise ValueError(f"{self.__class__.__name__} missing required dependencies: {missing}")

class LLMProcessor(dspy.Module, BaseProcessor):
    """Base class for LM-based processors"""
    def __init__(self, step: LLMStep):
        dspy.Module.__init__(self)
        BaseProcessor.__init__(self, step)
        
        self.lm = step.lm_name.get_lm(step.task_config)
        predictor_type = step.lm_name.get_predictor_type(step.task_config)
        
        try:
            predictor_class = getattr(dspy, predictor_type.value)
        except AttributeError:
            raise ValueError(f"Invalid predictor type: {predictor_type.value}")
        
        self.predictors = {
            sig.__name__: predictor_class(sig)
            for sig in step.signatures
        }

class ComputeProcessor(BaseProcessor):
    """Base class for non-LM processors"""
    pass

class ProcessingPipeline(dspy.Module):
    """Pipeline for executing a series of LLM processing steps.
    Manages the flow of data through multiple processing steps, handling
    dependencies and storing results.
    """
    def __init__(self, config: PipelineConfig):
        super().__init__()
        self.config = config
        self.processors: Dict[Union[str, Enum], Type[BaseProcessor]] = {}
        self.logger = logging.getLogger("pipeline")
        
        for step in config.steps:
            step_type = step.step_type.value if isinstance(step.step_type, Enum) else step.step_type
            self.register_processor(step_type, step.processor_class)

    def register_processor(self, 
                         step_type: Union[str, Enum], 
                         processor_class: Type[BaseProcessor]):
        """Register a processor class for a specific step type.
        Args:
            step_type: Identifier for the processing type
            processor_class: Class that implements the processing
        """
        self.processors[step_type] = processor_class

    def execute(self, data: dict) -> dict:
        """Execute the complete processing pipeline.
        Args:
            data: Initial input data
        Returns:
            dict: Processed data with all step results
        Raises:
            ValueError: If no processor is registered for a step type
            ValidationError: If step output validation fails
        """
        for step in self.config.steps:
            if self.config.verbose:
                self.logger.info(f"Executing step: {step.step_type}")

            if step.step_type not in self.processors:
                raise ValueError(f"No processor registered for step type: {step.step_type}")

            processor = self.processors[step.step_type](step)

            try:
                with dspy.context(lm=processor.lm):
                    result = processor.process(data)

                if self.config.validation and not processor.validate_output(result):
                    raise ValidationError(f"Output validation failed for step: {step.step_type}")

                if step.output_key:
                    data[step.output_key] = result
                else:
                    data.update(result)
                    
                if self.config.verbose:
                    self.logger.info(f"Completed step: {step.step_type}")

            except Exception as e:
                self.logger.error(f"Error in step {step.step_type}: {str(e)}")
                raise

        return data


class ValidationError(Exception):
    """Raised when step output validation fails."""
    pass
