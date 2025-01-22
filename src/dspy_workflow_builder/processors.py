"""
Processor implementations for LLM and compute-based workflow steps.
"""

import logging
import dspy
from dspy_workflow_builder.steps import BaseStep, LMStep
from dspy_workflow_builder.utils.logging import log_step
from dspy_workflow_builder.utils.text_utils import normalize_text_fields, serialize_paths


class BaseProcessor:
    """Base class for all processors"""
    def __init__(self, step: BaseStep):
        self.step = step
        self.logger = logging.getLogger(f"processor.{step.step_type}")

    @log_step()
    def process(self, data: dict) -> dict:
        pre_processed = self._pre_process(data)
        result = self._process(pre_processed)
        return self._post_process(result)
    
    def _process(self, data: dict) -> dict:
        raise NotImplementedError()
    
    def _pre_process(self, data: dict) -> dict:
        self._validate_dependencies(data)
        return normalize_text_fields(data)
    
    def _post_process(self, data: dict) -> dict:
        result = serialize_paths(data)
        if not self._validate_output(result):
            raise ValueError(f"Output validation failed for step: {self.step.step_type}")
        return result
        
    def _validate_dependencies(self, data: dict) -> None:
        """Validate that declared dependencies exist in data"""
        if not self.step.depends_on:
            return
            
        missing = [dep for dep in self.step.depends_on if dep not in data]
        if missing:
            raise ValueError(f"{self.__class__.__name__} missing required dependencies: {missing}")
            
    def _validate_output(self, result: dict) -> bool:
        """Validate processor output. Override in subclasses for specific validation."""
        return isinstance(result, dict) and bool(result)


class LMProcessor(dspy.Module, BaseProcessor):
    """Base class for LM-based processors"""
    def __init__(self, step: LMStep):
        dspy.Module.__init__(self)
        BaseProcessor.__init__(self, step)
        
        self.lm = step.lm_name.get_lm()
        predictor_type = step.lm_name.get_predictor_type()
        
        try:
            predictor_class = getattr(dspy, predictor_type.value)
        except AttributeError:
            raise ValueError(f"Invalid predictor type: {predictor_type.value}")
        
        self.predictors = {
            sig.__name__: predictor_class(sig)
            for sig in step.signatures
        }
