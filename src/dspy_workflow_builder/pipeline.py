"""
Pipeline implementation for executing LLM-based workflows.
"""

from dataclasses import dataclass
from enum import Enum
import logging
from typing import Dict, List, Type, Union

import dspy

from dspy_workflow_builder.steps import BaseStep, LMStep
from dspy_workflow_builder.processors import BaseProcessor


class ValidationError(Exception):
    """Raised when step output validation fails."""
    pass


@dataclass
class PipelineConfig:
    """Configuration for the entire processing pipeline"""
    steps: List[Union[LMStep, BaseStep]]
    verbose: bool = False
    validation: bool = True 


class Pipeline(dspy.Module):
    """Pipeline for executing a series of processing steps.
    Manages the flow of data through multiple processing steps, handling
    dependencies and storing results.
    """
    def __init__(self, config: PipelineConfig):
        super().__init__()
        self.config = config
        self.processors: Dict[Union[str, Enum], Type[BaseProcessor]] = {}
        self.logger = logging.getLogger("pipeline")
        
        for step in config.steps:
            self.register_processor(step.step_type, step.processor_class)

    def register_processor(self, 
                         step_type: Union[str, Enum], 
                         processor_class: Type[BaseProcessor]):
        """Register a processor class for a specific step type."""
        self.processors[step_type] = processor_class

    def execute(self, data: dict) -> dict:
        """Execute the complete processing pipeline."""
        for step in self.config.steps:
            if self.config.verbose:
                self.logger.info(f"Executing step: {step.step_type}")

            if step.step_type not in self.processors:
                raise ValueError(f"No processor registered for step type: {step.step_type}")

            processor = self.processors[step.step_type](step)

            try:
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