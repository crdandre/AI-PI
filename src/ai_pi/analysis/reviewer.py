from enum import Enum
import json
from typing import List, Dict
import dspy
from big_dict_energy.pipeline import Pipeline, PipelineConfig
from big_dict_energy.processors import LMProcessor
from big_dict_energy.steps import LMStep
from big_dict_energy.lm_setup import LMForTask

class ReviewStepType(Enum):
    """Types of review steps available"""
    DOCUMENT_KNOWLEDGE = "document_knowledge"
    FULL_DOCUMENT_REVIEW = "full_document_review"
    SECTION_REVIEW = "section_review"
    REVIEW_ITEMS = "review_items"


class FullDocumentReviewProcessor(LMProcessor):
    class Signature(dspy.Signature):
        """(You're a freaking genius scientist whose ego rests on ability to create insightful publications)
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

    def _process(self, data: dict) -> dict:
        if not self.step.signatures:
            raise ValueError("No signatures configured for FullDocumentReviewer")        
        
        predictor = self.predictors[self.step.signatures[0].__name__]
        result = predictor(
            document_text=data.get('full_text', ''),
            context=data,
            criteria=data.get('criteria', {})
        )
        
        return {
            'overall_assessment': result.overall_assessment,
            'key_strengths': result.key_strengths,
            'key_weaknesses': result.key_weaknesses,
            'global_suggestions': result.global_suggestions
        }


class ReviewItemsProcessor(LMProcessor):
    class Signature(dspy.Signature):
        """(You're a freaking genius scientist whose ego rests on ability to create insightful publications)
        Generate a list of relevant review items to address for the writer. These
        items should exist downstream of the broader review of the paper as concrete steps
        to realize the improvements suggested. Ensure the broader review's context is
        reflected in the review items you create."""
        section_text = dspy.InputField(desc="The text content being reviewed")
        section_type = dspy.InputField(desc="The type of section")
        context = dspy.InputField(desc="Additional context about the paper")
        review_items = dspy.OutputField(
            desc="""List of review items. Each item contains these fields:
            - match_string: The exact text from the paper that needs revision
            - comment: The review comment explaining what should be changed
            - revision: The suggested revised text
            - section_type: the section in which the item was found
            - reason: The thought pattern or rationale behind this specific suggestion
            
            The comment and revision fields are each optional but at least
            one of them must be present (no need for both everytime).
            
            If there is neither a comment or revision to be made, do not add the item.
            """,
            format=List[Dict]
        )

    def _process(self, data: dict) -> dict:
        """Process the input data to generate review items"""
        if not self.step.signatures:
            raise ValueError("No signatures configured for ReviewItemsProcessor")
        
        all_review_items = []
        section_text = data.get('section_text', '')
        section_type = data.get('section_type', '')
        
        for signature in self.step.signatures:
            predictor = self.predictors[signature.__name__]
            result = predictor(
                section_text=section_text,
                section_type=section_type,
                context=data
            )
            
            review_items = result.review_items
            if isinstance(review_items, str):
                if '```json' in review_items:
                    json_str = review_items.split('```json\n')[1].split('\n```')[0]
                    review_items = json.loads(json_str)
            
            # Ensure section_type is set for each review item
            if isinstance(review_items, list):
                for item in review_items:
                    if isinstance(item, dict):
                        item['section_type'] = section_type
                all_review_items.extend(review_items)
        
        return {'review_items': all_review_items}
    
    def _validate_output(self, output: dict) -> bool:
        """Validate review items structure is present and contains required fields"""
        if 'review_items' in output:
            required_fields = {'match_string', 'comment', 'revision', 'section_type', 'reason'}
            return all(
                isinstance(item, dict) and 
                all(field in item for field in required_fields)
                for item in output['review_items']
            )
        return False


def create_reviewer_pipeline(verbose: bool = False) -> Pipeline:
    """Create pipeline with review steps"""
    steps = [
        LMStep(
            step_type=ReviewStepType.FULL_DOCUMENT_REVIEW,
            lm_name=LMForTask.DOCUMENT_REVIEW,
            processor_class=FullDocumentReviewProcessor,
            depends_on=["hierarchical_summary"],
            output_key="full_document_review",
        ),
        LMStep(
            step_type=ReviewStepType.REVIEW_ITEMS,
            lm_name=LMForTask.DOCUMENT_REVIEW,
            processor_class=ReviewItemsProcessor,
            depends_on=["full_document_review"],
            output_key="review_items",
        ),
    ]
    return Pipeline(PipelineConfig(steps=steps, verbose=verbose))


class Reviewer:
    """Creates a comprehensive review of an input document"""
    def __init__(self, verbose: bool = False):
        self.pipeline = create_reviewer_pipeline(verbose)

    def review_document(self, document_json: dict, topic_context: dict, hierarchical_summary: dict) -> dict:
        """
        Review the document using the pipeline, incorporating topic context
        and hierarchical summary information.
        """
        input_data = {
            'full_text': document_json.get('full_text', ''),
            'sections': document_json.get('sections', []),
            'hierarchical_summary': hierarchical_summary,
            'research_problem': hierarchical_summary.get('topic', ''),
            'topic_context': topic_context,
            'criteria': {'quality': 'high'}  # Can be expanded based on needs
        }
        
        results = self.pipeline.execute(input_data)
        
        return {
            'reviews': {
                'full_document_review': results['full_document_review'],
                'review_items': results['review_items']
            }
        }


if __name__ == "__main__":
    import logging
    
    # Setup basic logging with a handler
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Add console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(console_handler)

    pipeline = create_reviewer_pipeline()
    
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
    print("\nPipeline Results:")
    print(json.dumps(results['review_items'], indent=2))