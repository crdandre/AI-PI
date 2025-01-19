from typing import List, Type, Dict, Any, Optional
from dataclasses import dataclass, field
import dspy
from enum import Enum

from ai_pi.core.lm_pipeline import ProcessingStep, ProcessingPipeline, PipelineConfig, BaseProcessor

class ReviewStepType(Enum):
    """Types of review steps available"""
    DOCUMENT_KNOWLEDGE = "document_knowledge"
    FULL_DOCUMENT_REVIEW = "full_document_review"
    SECTION_REVIEW = "section_review"
    REVIEW_ITEMS = "review_items"

class BaseReviewer(BaseProcessor):
    """Base reviewer that modifies the document JSON structure"""
    
    def validate_output(self, output: dict) -> bool:
        """Validate review items structure is present and contains required fields"""
        if 'review_items' in output:
            required_fields = {'match_string', 'comment', 'revision', 'section_type'}
            return all(
                isinstance(item, dict) and 
                all(field in item for field in required_fields)
                for item in output['review_items']
            )
        return True

# Example reviewer implementations
class PaperKnowledgeBuilder(BaseReviewer):   
    def process(self, data: dict) -> dict:
        # Implementation for building paper knowledge
        raise NotImplementedError()
    
class FullDocumentReviewer(BaseReviewer):
    class Signature(dspy.Signature):
        """Signature for performing a full document review"""
        document_text: str = dspy.InputField(desc="The full text of the document to review")
        context: dict = dspy.InputField(desc="Additional context including research problem, sections, and topic context")
        criteria: dict = dspy.InputField(desc="Review criteria to apply")
        
        overall_assessment: str = dspy.OutputField(desc="Overall assessment of the document")
        key_strengths: List[str] = dspy.OutputField(desc="List of key strengths identified")
        key_weaknesses: List[str] = dspy.OutputField(desc="List of key weaknesses identified")
        global_suggestions: List[str] = dspy.OutputField(desc="List of suggestions for improvement")

    def process(self, data: dict) -> dict:
        if not self.step.signatures:
            raise ValueError("No signatures configured for FullDocumentReviewer")        
        
        dependencies = self.get_dependencies(data)       
        predictor = self.predictors[self.step.signatures[0].__name__]
        document_text = data.get('full_text', '')
        context = {
            'research_problem': data.get('research_problem', ''),
            'sections': data.get('sections', []),
            'topic_context': data.get('topic_context', ''),
            **dependencies,
        }
        criteria = data.get('criteria', {})
        
        result = predictor(
            document_text=document_text,
            context=context,
            criteria=criteria
        )
        
        return {
            'overall_assessment': result.overall_assessment,
            'key_strengths': result.key_strengths,
            'key_weaknesses': result.key_weaknesses,
            'global_suggestions': result.global_suggestions
        }


def create_pipeline(custom_steps: Optional[List[ProcessingStep]] = None) -> ProcessingPipeline:
    """Create pipeline with default or custom steps"""
    default_steps = [
        ProcessingStep(
            step_type=ReviewStepType.FULL_DOCUMENT_REVIEW,
            name="full_document_review",
            lm_name="document_review",
            signatures=[FullDocumentReviewer.Signature],
            predictor_type="chain_of_thought",
            depends_on=["hierarchical_summary"],
            output_key=["full_document_review"],
        ),
    ]
    
    steps = custom_steps if custom_steps is not None else default_steps
    config = PipelineConfig(steps=steps)
    pipeline = ProcessingPipeline(config)
    
    # Register reviewers
    pipeline.register_processor(ReviewStepType.FULL_DOCUMENT_REVIEW, FullDocumentReviewer)
    
    return pipeline