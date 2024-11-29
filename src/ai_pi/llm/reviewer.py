"""
This file contains the code for an agent which creates suggestions for comments 
and revisions, simulating the style of feedback a PI would give when reviewing a paper.
"""
import dspy
import json
from typing import Union
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class ReviewItem(dspy.Signature):
    """Individual review item structure"""
    match_text = dspy.OutputField(desc="Exact text to match")
    comment = dspy.OutputField(desc="Review comment")
    revision = dspy.OutputField(desc="Complete revised text")

class ReviewerSignature(dspy.Signature):
    """Generate review feedback for a section of text."""
    section_text = dspy.InputField()
    section_type = dspy.InputField()
    context = dspy.InputField()
    feedback = dspy.OutputField()

class SectionReviewer(dspy.Module):
    """Reviews individual sections with awareness of full paper context"""
    
    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel], verbose=False):
        super().__init__()
        self.reviewer = dspy.Predict(ReviewerSignature)
        self.engine = engine
        self.verbose = verbose

    def review_section(self, section_text: str, section_type: str, paper_context: dict) -> dict:
        """
        Main entry point for reviewing sections. Handles error cases and logging.
        """
        try:
            if self.verbose:
                print(f"Reviewing {section_type} section...")
            
            result = self.forward(section_text, section_type, paper_context)
            return result.review
            
        except Exception as e:
            if self.verbose:
                print(f"Error reviewing section: {str(e)}")
            return {
                'match_strings': [],
                'comments': [],
                'revisions': []
            }

    def forward(
        self,
        section_text: str,
        section_type: str,
        paper_context: dict
    ) -> dspy.Prediction:
        """
        Review a section using direct prediction.
        """
        if not section_text or not section_type:
            logger.warning("Empty input received")
            return self._create_empty_review()

        try:
            with dspy.settings.context(lm=self.engine):
                logger.debug(f"Attempting review with engine: {type(self.engine)}")
                
                # Prepare context as string
                context_str = paper_context.get('paper_summary', '')
                
                # Get the raw prediction
                result = self.reviewer(
                    section_text=str(section_text),
                    section_type=str(section_type),
                    context=str(context_str)
                )
                
                # Parse the feedback into our structure
                review_items = {
                    'match_strings': [],
                    'comments': [],
                    'revisions': []
                }
                
                if hasattr(result, 'feedback'):
                    feedback = str(result.feedback)
                    items = feedback.split('---')
                    for item in items:
                        if not item.strip():
                            continue
                        
                        match_text = ""
                        comment = ""
                        revision = ""
                        
                        for line in item.strip().split('\n'):
                            if line.startswith('MATCH:'):
                                match_text = line[6:].strip()
                            elif line.startswith('COMMENT:'):
                                comment = line[8:].strip()
                            elif line.startswith('REVISION:'):
                                revision = line[9:].strip()
                        
                        if match_text and match_text in section_text:
                            review_items['match_strings'].append(match_text)
                            review_items['comments'].append(comment)
                            review_items['revisions'].append(revision)
                
                logger.debug(f"Final review_items structure: {review_items}")
                return dspy.Prediction(review=review_items)
                
        except Exception as e:
            logger.error(f"Error in forward method: {str(e)}", exc_info=True)
            return self._create_empty_review()

    def _create_empty_review(self) -> dspy.Prediction:
        """Helper method to create an empty review structure"""
        return dspy.Prediction(review={
            'match_strings': [],
            'comments': [],
            'revisions': []
        })

    def _process_review_items(self, review_json: str, section_text: str) -> dict:
        """Helper method to process review items and handle serialization"""
        review_items = {
            'match_strings': [],
            'comments': [],
            'revisions': []
        }
        
        try:
            # Handle potential string escaping
            if isinstance(review_json, str):
                review_json = review_json.replace("'", '"')
            
            parsed_response = json.loads(review_json)
            
            for item in parsed_response.get('review_items', []):
                match_text = str(item.get('match_text', '')).strip()
                
                if match_text and match_text in section_text:
                    review_items['match_strings'].append(match_text)
                    review_items['comments'].append(str(item.get('comment', '')))
                    review_items['revisions'].append(str(item.get('revision', '')))
                    
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"JSON parsing error: {str(e)}")
                
        return review_items


if __name__ == "__main__":
    # Test the reviewer with DSPy's LM directly
    lm = dspy.LM('openai/gpt-4')
    reviewer = SectionReviewer(lm)
    
    test_context = {
        'paper_summary': 'Test paper about scoliosis modeling.'
    }
    
    test_section = {
        'text': 'This is a test section about scoliosis modeling.',
        'type': 'Methods'
    }
    
    review = reviewer.review_section(
        test_section['text'],
        test_section['type'],
        test_context
    )
    print("\nFinal Review:")
    print(json.dumps(review, indent=2))