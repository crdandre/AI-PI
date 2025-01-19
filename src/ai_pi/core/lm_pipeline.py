"""
Core building blocks for structuring LLM-based workflows.
This module provides the base classes and utilities for creating modular,
configurable pipelines for any LLM-based processing task.
"""

from typing import List, Type, Dict, Any, Optional, Union, Literal
from dataclasses import dataclass, field
from enum import Enum
import dspy
import logging
from ai_pi.lm_config import get_lm_for_task
from ai_pi.core.utils.logging import log_step

@dataclass
class ProcessingStep:
    """Configuration for any processing step in an LLM pipeline.
    
    Attributes:
        step_type: Identifier for the type of processing (usually from an Enum)
        name: Unique name for this step instance
        lm_name: Name of the language model configuration to use
        signatures: List of DSPy signatures defining the LLM interaction
        predictor_type: Type of DSPy predictor to use ("predict" or "chain_of_thought")
        depends_on: List of step names whose outputs this step requires
        output_key: Key under which to store this step's output
        config: Additional configuration parameters specific to this step
    """
    step_type: Union[str, Enum]
    name: str
    lm_name: str
    signatures: List[Type[dspy.Signature]]
    output_key: str
    predictor_type: Literal["predict", "chain_of_thought"] = "chain_of_thought"
    depends_on: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)

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
        super().__init__()  # Initialize dspy.Module
        self.step = step
        self.lm = get_lm_for_task(step.lm_name)
        self.logger = logging.getLogger(f"processor.{step.name}")
        
        # Choose predictor type based on configuration
        predictor_class = dspy.ChainOfThought if step.predictor_type == "chain_of_thought" else dspy.Predict
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
        """
        raise NotImplementedError()

    def get_dependencies(self, data: dict) -> Dict[str, Any]:
        """Retrieve required data from previous steps.
        Args:
            data: Input data containing all pipeline results 
        Returns:
            dict: Data required by this step from previous steps
        """
        return {
            dep: data.get(dep, {})
            for dep in self.step.depends_on
        }

    def validate_output(self, output: dict) -> bool:
        """Validate the step's output.
        Args:
            output: Step processing results
        Returns:
            bool: True if output is valid, False otherwise
        """
        return True  # Override for specific validation

class ProcessingPipeline(dspy.Module):
    """Pipeline for executing a series of LLM processing steps.
    Manages the flow of data through multiple processing steps, handling
    dependencies and storing results.
    """
    def __init__(self, config: PipelineConfig):
        super().__init__()  # Initialize dspy.Module
        self.config = config
        self.processors: Dict[Union[str, Enum], Type[BaseProcessor]] = {}
        self.logger = logging.getLogger("pipeline")

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
                self.logger.info(f"Executing step: {step.name}")

            if step.step_type not in self.processors:
                raise ValueError(f"No processor registered for step type: {step.step_type}")

            processor = self.processors[step.step_type](step)

            try:
                with dspy.context(lm=processor.lm):
                    result = processor.process(data)

                if self.config.validation and not processor.validate_output(result):
                    raise ValidationError(f"Output validation failed for step: {step.name}")

                if step.output_key:
                    data[step.output_key] = result
                else:
                    data.update(result)
                    
                if self.config.verbose:
                    self.logger.info(f"Completed step: {step.name}")

            except Exception as e:
                self.logger.error(f"Error in step {step.name}: {str(e)}")
                raise

        return data


class ValidationError(Exception):
    """Raised when step output validation fails."""
    pass

# Example base class for specific domains
class DomainProcessor(BaseProcessor):
    """Base class for domain-specific processors.
    
    Provides common functionality for a specific processing domain.
    Inherit from this to create processors for your domain.
    """
    def __init__(self, step: ProcessingStep):
        super().__init__(step)  # This will call both BaseProcessor and dspy.Module initialization
        self.domain_config = step.config.get("domain_specific", {})

    def preprocess(self, data: dict) -> dict:
        """Prepare data for processing."""
        return data

    def postprocess(self, result: dict) -> dict:
        """Clean up and format processing results."""
        return result

    def process(self, data: dict) -> dict:
        """Template method for domain processing."""
        preprocessed = self.preprocess(data)
        result = self.process_domain_specific(preprocessed)
        return self.postprocess(result)

    def process_domain_specific(self, data: dict) -> dict:
        """Implement domain-specific processing logic."""
        raise NotImplementedError() 