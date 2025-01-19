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

    
class FullDocumentReviewProcessor(BaseProcessor):
    class Signature(dspy.Signature):
        """(You're a freaking genius scientist who is driven by creating insightful publications)
        Review the entire document and assess it's pros and cons. Think beyond the paper, about
        the way it addresses it's topic and whether that is best, whether the paper is well-positioned
        in it's field given current progress, whether the narrative flow is coherent, etc."""
        
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

    def validate_output(self, output: dict) -> bool:
        """No specific validation needed for full document review"""
        return True


class ReviewItemsProcessor(BaseProcessor):
    class Signature(dspy.Signature):
        """(You're a freaking genius scientist who is driven by creating insightful publications)
        Generate a list of relevant review items to address for the writer. These
        items should exist downstream of the broader review of the paper as concrete steps
        to realize the improvements suggested. Ensure the broader review's context is
        reflected in the review items you create."""
        section_text = dspy.InputField(desc="The text content being reviewed")
        section_type = dspy.InputField(desc="The type of section")
        context = dspy.InputField(desc="Additional context about the paper")
        reason = dspy.OutputField(desc="The thought pattern and origin of the suggestion.")
        review_items = dspy.OutputField(
            desc="""List of review items. Each item contains these fields:
            - match_string: The exact text from the paper that needs revision
            - comment: The review comment explaining what should be changed
            - revision: The suggested revised text
            - section_type: the section in which the item was found
            
            The comment and revision fields are each optional but at least
            one of them must be present (no need for both everytime).
            
            If there is neither a comment or revision to be made, do not add the item.
            """,
            format=List[Dict]
        )

    def process(self, data: dict) -> dict:
        raise NotImplementedError()
    
    def validate_output(self, output: dict) -> bool:
        """Validate review items structure is present and contains required fields"""
        if 'review_items' in output:
            required_fields = {'match_string', 'comment', 'revision', 'section_type'}
            return all(
                isinstance(item, dict) and 
                all(field in item for field in required_fields)
                for item in output['review_items']
            )
        else:
            raise ValueError("Required field 'review_items' not found in output")


def create_pipeline(custom_steps: Optional[List[ProcessingStep]] = None) -> ProcessingPipeline:
    """Create pipeline with default or custom steps"""
    default_steps = [
        ProcessingStep(
            step_type=ReviewStepType.FULL_DOCUMENT_REVIEW,
            lm_name="document_review",
            processor_class=FullDocumentReviewProcessor,
            predictor_type="chain_of_thought",
            depends_on=["hierarchical_summary"],
            output_key="full_document_review",
        ),
        ProcessingStep(
            step_type=ReviewStepType.REVIEW_ITEMS,
            lm_name="review_items",
            processor_class=ReviewItemsProcessor
        ),
    ]
    
    steps = custom_steps if custom_steps is not None else default_steps
    return ProcessingPipeline(PipelineConfig(steps=steps, verbose=True))


if __name__ == "__main__":
    import logging
    
    # Setup basic logging with a handler
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(console_handler)

    pipeline = create_pipeline()
    
    # Example input data with some content
    data = {
        "full_text": "This is a sample document text.",
        "sections": ["Introduction", "Methods", "Results"],
        "hierarchical_summary": {"key": "value"},
        "research_problem": "Sample research problem",
        "topic_context": "Sample topic context",
        "criteria": {"quality": "high"}
    }
    
    results = pipeline.execute(data)
    
    # Print results in a readable JSON format
    import json
    print("\nPipeline Results:")
    print(json.dumps(results, indent=2))