"""
Processor implementations for LLM and compute-based workflow steps.
"""

import logging
import dspy
from dspy_workflow_builder.steps import BaseStep, LMStep
from dspy_workflow_builder.utils.logging import log_step


class BaseProcessor:
    """Base class for all processors"""
    def __init__(self, step: BaseStep):
        self.step = step
        self.logger = logging.getLogger(f"processor.{step.step_type}")

    @log_step()
    def process(self, data: dict) -> dict:
        self._validate_dependencies(data)
        return self._process(data)
    
    def validate_output(self, result: dict) -> bool:
        """Validate processor output. Override in subclasses for specific validation."""
        return True

    def _process(self, data: dict) -> dict:
        raise NotImplementedError()
        
    def _validate_dependencies(self, data: dict) -> None:
        """Validate that declared dependencies exist in data"""
        if not self.step.depends_on:
            return
            
        missing = [dep for dep in self.step.depends_on if dep not in data]
        if missing:
            raise ValueError(f"{self.__class__.__name__} missing required dependencies: {missing}")


class LMProcessor(dspy.Module, BaseProcessor):
    """Base class for LM-based processors"""
    def __init__(self, step: LMStep):
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