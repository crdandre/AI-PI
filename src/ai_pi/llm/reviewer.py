"""
This file contains the code for an agent which creates suggestions for comments 
and revisions, simulating the style of feedback a PI would give when reviewing a paper.
"""
import dspy
import json
from typing import Union
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReviewItem(dspy.Signature):
    """Individual review item structure"""
    match_text = dspy.OutputField(desc="Exact text to match")
    comment = dspy.OutputField(desc="Review comment")
    revision = dspy.OutputField(desc="Complete revised text")

class ReviewerSignature(dspy.Signature):
    """Generate review feedback for a section of text.
    
    You are an expert academic reviewer. For the given section of text, identify areas that need improvement
    and suggest specific revisions. Focus on clarity, completeness, and scientific rigor.
    
    For example:
    Input:
        section_text: "The model predicted curve progression with an average error of 5 degrees."
        section_type: "Results"
        context: "Paper about scoliosis modeling"
    Output:
        reasoning: "This result statement needs more specificity about the error metric and comparison to existing methods."
        review_items: [
            {
                "match_text": "The model predicted curve progression with an average error of 5 degrees.",
                "comment": "The error metric should be specified (RMSE, MAE?) and compared to previous work.",
                "revision": "The model predicted curve progression with a root mean square error of 5 degrees, improving upon previous methods which reported errors of 8-10 degrees."
            }
        ]
    """
    section_text = dspy.InputField(desc="The text content of the section to review")
    section_type = dspy.InputField(desc="The type of section (e.g., Methods, Results)")
    context = dspy.InputField(desc="Additional context about the paper")
    reasoning = dspy.OutputField(desc="Explanation of review decisions")
    review_items = dspy.OutputField(desc="List of review items in JSON format", format=list[ReviewItem])

class SectionReviewer(dspy.Module):
    """Reviews individual sections with awareness of full paper context"""
    
    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel], verbose=False):
        super().__init__()
        
        # Instead of using examples in ChainOfThought initialization,
        # we should use a prompt template or modify the signature
        self.reviewer = dspy.ChainOfThought(ReviewerSignature)
        
        # Add example as a class attribute if needed for reference
        self.example_input = {
            "section_text": "The model predicted curve progression with an average error of 5 degrees.",
            "section_type": "Results",
            "context": "Paper about scoliosis modeling"
        }
        self.example_output = {
            "reasoning": "This result statement needs more specificity about the error metric and comparison to existing methods.",
            "review_items": [
                {
                    "match_text": "The model predicted curve progression with an average error of 5 degrees.",
                    "comment": "The error metric should be specified (RMSE, MAE?) and compared to previous work.",
                    "revision": "The model predicted curve progression with a root mean square error of 5 degrees, improving upon previous methods which reported errors of 8-10 degrees."
                }
            ]
        }
        
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
            
            # The result is a dspy.Prediction object with a 'review' field
            # We should return the dictionary directly since workflow2.py expects it
            if hasattr(result, 'review'):
                return result.review  # Return just the review dict, not wrapped in another dict
            
            logger.warning("No review field found in result")
            return {
                'review': {  # Keep the nested structure
                    'match_strings': [],
                    'comments': [],
                    'revisions': []
                }
            }
                
        except Exception as e:
            if self.verbose:
                print(f"Error reviewing section: {str(e)}")
            return {
                'review': {  # Keep the nested structure
                    'match_strings': [],
                    'comments': [],
                    'revisions': []
                }
            }

    def forward(
        self,
        section_text: str,
        section_type: str,
        paper_context: dict
    ) -> dspy.Prediction:
        """Review a section using chain-of-thought reasoning."""
        if not section_text or not section_type:
            logger.warning("Empty input received")
            return self._create_empty_review()

        try:
            with dspy.settings.context(lm=self.engine):
                result = self.reviewer(
                    section_text=section_text,
                    section_type=section_type,
                    context=paper_context.get('paper_summary', '')
                )
                
                # Create the expected review structure
                review_data = {
                    'match_strings': [],
                    'comments': [],
                    'revisions': []
                }
                
                # Parse the review_items if they came as a string
                if isinstance(result.review_items, str):
                    import json
                    try:
                        items = json.loads(result.review_items)
                    except json.JSONDecodeError:
                        logger.error("Failed to parse review items JSON")
                        return self._create_empty_review()
                else:
                    items = result.review_items
                
                # Process review items
                for item in items:
                    if isinstance(item, dict):  # Direct dictionary access
                        if 'match_text' in item and item['match_text'] in section_text:
                            review_data['match_strings'].append(item['match_text'])
                            review_data['comments'].append(item['comment'])
                            review_data['revisions'].append(item['revision'])
                    else:  # DSPy object access
                        if hasattr(item, 'match_text') and item.match_text in section_text:
                            review_data['match_strings'].append(item.match_text)
                            review_data['comments'].append(item.comment)
                            review_data['revisions'].append(item.revision)
                
                logger.debug(f"Processed review data: {review_data}")
                return dspy.Prediction(review=review_data)
                
        except Exception as e:
            logger.error(f"Error in forward method: {str(e)}", exc_info=True)
            return self._create_empty_review()

    def _create_empty_review(self) -> dspy.Prediction:
        """Create an empty review structure"""
        return dspy.Prediction(
            review={
                'match_strings': [],
                'comments': [],
                'revisions': []
            }
        )

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
    lm = dspy.LM('openai/gpt-4o-mini')
    reviewer = SectionReviewer(lm)
    
    test_context = {
        'paper_summary': 'Test paper about scoliosis modeling.'
    }
    
    test_section = {
        'text': 'This is a test section about scoliosis modeling. I like potatosss REPLACE ME THIS IAN ERROR.!!!',
        'type': 'Methods'
    }
    
    review = reviewer.review_section(
        test_section['text'],
        test_section['type'],
        test_context
    )
    print("\nFinal Review:")
    print(json.dumps(review, indent=2))