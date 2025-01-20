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
from dspy_workflow_builder.lm_config import LMForTask, PredictorType, TaskConfig
from dspy_workflow_builder.utils.logging import log_step

@dataclass
class ProcessingStep:
    """Configuration for any processing step in an LLM pipeline.
    
    Attributes:
        step_type: Identifier for the type of processing (usually from an Enum)
        lm_name: Task-specific language model configuration
        processor_class: Class that implements the processing
        output_key: Key under which to store this step's output
        task_config: Optional override for the task's default configuration
        depends_on: List of step names whose outputs this step requires
        signatures: List of DSPy signatures defining the LLM interaction
        config: Additional configuration parameters specific to this step
    """
    step_type: Union[str, Enum]
    lm_name: LMForTask
    processor_class: Type["BaseProcessor"]
    output_key: str
    task_config: Optional[TaskConfig] = None
    depends_on: List[str] = field(default_factory=list)
    signatures: Optional[List[Type[dspy.Signature]]] = None
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Use provided signatures or auto-discover from processor class"""
        if self.signatures is None:
            self.signatures = [
                member for _, member in vars(self.processor_class).items()
                if isclass(member) and issubclass(member, dspy.Signature) and member != dspy.Signature
            ]
        if not self.signatures:
            raise ValueError(f"No Signature classes found for {self.processor_class.__name__}")

@dataclass
class PipelineConfig:
    """Configuration for the entire processing pipeline.
    
    Attributes:
        steps: Ordered list of processing steps
        verbose: Enable detailed logging
        validation: Enable output validation
    """
    steps: List[ProcessingStep]
    verbose: bool = False
    validation: bool = True

class BaseProcessor(dspy.Module):
    """Base class for implementing LLM processing steps.
    
    Each specific processor should inherit from this class and implement
    the process() method with its specific logic.
    """
    def __init__(self, step: ProcessingStep):
        super().__init__()
        self.step = step
        self.logger = logging.getLogger(f"processor.{step.step_type}")
        
        # Only set up LM if specified
        if step.lm_name:
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

    @log_step()
    def process(self, data: dict) -> dict:
        """Process the input data and return results.
        Args:
            data: Input data containing results from previous steps
        Returns:
            dict: Processing results
        Raises:
            NotImplementedError: Must be implemented by subclasses
            ValueError: If required dependencies are missing
        """
        self._validate_dependencies(data)
        return self._process(data)

    def validate_output(self, output: dict) -> bool:
        """
        Override this for specific validation
        """
        return True
    
    def _process(self, data: dict) -> dict:
        """
        Override with process step logic
        """
        raise NotImplementedError()

    def _validate_dependencies(self, data: dict) -> None:
        """Validate that declared dependencies exist in data"""
        if not self.step.depends_on:
            return
            
        missing = [dep for dep in self.step.depends_on if dep not in data]
        if missing:
            raise ValueError(f"{self.__class__.__name__} missing required dependencies: {missing}")


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
        
        # Auto-register processors from steps
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
                self.logger.info(f"Executing step: {step.lm_name.value}")

            if step.step_type not in self.processors:
                raise ValueError(f"No processor registered for step type: {step.step_type}")

            processor = self.processors[step.step_type](step)

            try:
                with dspy.context(lm=processor.lm):
                    result = processor.process(data)

                if self.config.validation and not processor.validate_output(result):
                    raise ValidationError(f"Output validation failed for step: {step.lm_name.value}")

                if step.output_key:
                    data[step.output_key] = result
                else:
                    data.update(result)
                    
                if self.config.verbose:
                    self.logger.info(f"Completed step: {step.lm_name.value}")

            except Exception as e:
                self.logger.error(f"Error in step {step.lm_name.value}: {str(e)}")
                raise

        return data


class ValidationError(Exception):
    """Raised when step output validation fails."""
    pass
