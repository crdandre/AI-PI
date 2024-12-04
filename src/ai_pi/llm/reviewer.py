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
    """Generate high-level scientific review feedback for a section of text.
    
    You are a senior academic reviewer with expertise in providing strategic, high-level feedback.
    Focus on scientific merit, methodology soundness, and theoretical contributions rather than
    minor writing issues. After generating feedback, reflect on whether your comments address
    substantive scientific concerns.
    
    For example:
    Input:
        section_text: "The model predicted curve progression with an average error of 5 degrees."
        section_type: "Results"
        context: "Paper about scoliosis modeling"
    Output:
        initial_analysis: "The results lack scientific rigor in three areas: (1) no statistical validation, 
            (2) missing comparison with state-of-the-art methods, (3) no discussion of clinical significance."
        reflection: "My feedback appropriately focuses on core scientific issues rather than superficial edits. 
            The comments address methodology, validation, and clinical relevance."
        review_items: [
            {
                "match_text": "The model predicted curve progression with an average error of 5 degrees.",
                "comment": "The results require statistical validation (e.g., confidence intervals) and 
                    discussion of clinical significance. How does this error rate impact treatment decisions?",
                "revision": "The model predicted curve progression with an average error of 5° (95% CI: 3.2-6.8°). 
                    This accuracy level is clinically significant as it falls within the threshold needed for 
                    reliable treatment planning (< 7°), based on established clinical guidelines."
            }
        ]
    """
    section_text = dspy.InputField(desc="The text content of the section to review")
    section_type = dspy.InputField(desc="The type of section (e.g., Methods, Results)")
    context = dspy.InputField(desc="Additional context about the paper")
    initial_analysis = dspy.OutputField(desc="Initial high-level analysis of scientific concerns")
    reflection = dspy.OutputField(desc="Self-reflection on whether feedback addresses core scientific issues")
    review_items = dspy.OutputField(desc="List of review items in JSON format", format=list[ReviewItem])

class FinalReviewSignature(dspy.Signature):
    """Generate final synthesized review across all sections."""
    section_analyses = dspy.InputField(desc="List of section-level analyses")
    paper_context = dspy.InputField(desc="Overall paper context")
    overall_assessment = dspy.OutputField(desc="Synthesized scientific assessment of the entire paper")
    key_strengths = dspy.OutputField(desc="Major scientific strengths identified")
    key_weaknesses = dspy.OutputField(desc="Major scientific weaknesses identified")
    recommendations = dspy.OutputField(desc="High-priority recommendations for improvement")

class SectionReviewer(dspy.Module):
    """Reviews individual sections with awareness of full paper context"""
    
    def __init__(self, 
                 engine: dspy.dsp.LM, 
                 use_cot: bool = False,  # Add flag for ChainOfThought vs Predict
                 verbose: bool = False):
        super().__init__()
        
        # Choose between ChainOfThought and Predict based on flag
        ReviewerClass = dspy.ChainOfThought if use_cot else dspy.Predict
        self.reviewer = ReviewerClass(ReviewerSignature)
        self.final_reviewer = ReviewerClass(FinalReviewSignature)
        
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
        """Review a section using chain-of-thought reasoning with scientific depth validation."""
        if not section_text or not section_type:
            logger.warning("Empty input received")
            return self._create_empty_review()

        try:
            with dspy.settings.context(lm=self.engine):
                # Get initial review result
                result = self.reviewer(
                    section_text=section_text,
                    section_type=section_type,
                    context=paper_context.get('paper_summary', '')
                )
                
                # Validate scientific depth of the review
                if not self._validate_scientific_depth(result):
                    logger.warning("Review lacks scientific depth - requesting revision")
                    # Retry with explicit instruction to focus on scientific aspects
                    result = self.reviewer(
                        section_text=section_text,
                        section_type=section_type,
                        context=f"{paper_context.get('paper_summary', '')} INSTRUCTION: Focus specifically on "
                               f"methodology, theoretical foundations, and scientific implications. Avoid "
                               f"superficial writing suggestions."
                    )
                
                # Create the expected review structure with additional fields
                review_data = {
                    'match_strings': [],
                    'comments': [],
                    'revisions': [],
                    'initial_analysis': result.initial_analysis,
                    'reflection': result.reflection
                }
                
                # Process review items (rest of the processing remains the same)
                items = self._parse_review_items(result.review_items)
                for item in items:
                    if self._is_valid_review_item(item) and self._contains_match_text(item, section_text):
                        review_data['match_strings'].append(item.get('match_text') if isinstance(item, dict) else item.match_text)
                        review_data['comments'].append(item.get('comment') if isinstance(item, dict) else item.comment)
                        review_data['revisions'].append(item.get('revision') if isinstance(item, dict) else item.revision)
                
                logger.debug(f"Processed review data: {review_data}")
                return dspy.Prediction(review=review_data)
                
        except Exception as e:
            logger.error(f"Error in forward method: {str(e)}", exc_info=True)
            return self._create_empty_review()

    def _validate_scientific_depth(self, result) -> bool:
        """
        Validate that the review focuses on substantive scientific issues.
        Returns True if the review meets scientific depth criteria.
        """
        # Define keywords/phrases that indicate scientific depth
        scientific_indicators = [
            'methodology', 'theoretical', 'statistical', 'validation',
            'evidence', 'hypothesis', 'implications', 'limitations',
            'mechanism', 'causality', 'framework', 'analysis'
        ]
        
        # Define patterns that suggest superficial feedback
        superficial_indicators = [
            'spelling', 'grammar', 'punctuation', 'word choice',
            'formatting', 'typo', 'rephrase'
        ]
        
        # Check initial analysis
        initial_analysis = result.initial_analysis.lower()
        scientific_count = sum(1 for indicator in scientific_indicators if indicator in initial_analysis)
        superficial_count = sum(1 for indicator in superficial_indicators if indicator in initial_analysis)
        
        # Check reflection
        reflection = result.reflection.lower()
        reflection_scientific_focus = any(indicator in reflection for indicator in scientific_indicators)
        
        # Check review items
        review_items = self._parse_review_items(result.review_items)
        substantive_comments = 0
        total_comments = len(review_items)
        
        for item in review_items:
            comment = item.get('comment', '') if isinstance(item, dict) else getattr(item, 'comment', '')
            comment = comment.lower()
            if any(indicator in comment for indicator in scientific_indicators):
                substantive_comments += 1
        
        # Criteria for passing validation
        criteria = [
            scientific_count >= 2,  # At least 2 scientific indicators in initial analysis
            superficial_count <= 1,  # No more than 1 superficial indicator
            reflection_scientific_focus,  # Reflection mentions scientific aspects
            substantive_comments / max(total_comments, 1) >= 0.7  # At least 70% of comments are substantive
        ]
        
        return sum(criteria) >= 3  # Pass if at least 3 of 4 criteria are met

    def _parse_review_items(self, review_items):
        """Helper method to parse review items regardless of input format"""
        if isinstance(review_items, str):
            try:
                return json.loads(review_items)
            except json.JSONDecodeError:
                logger.error("Failed to parse review items JSON")
                return []
        return review_items

    def _is_valid_review_item(self, item):
        """Check if a review item has all required fields"""
        required_fields = ['match_text', 'comment', 'revision']
        if isinstance(item, dict):
            return all(field in item for field in required_fields)
        return all(hasattr(item, field) for field in required_fields)

    def _contains_match_text(self, item, section_text):
        """Check if the match_text exists in the section"""
        match_text = item.get('match_text', '') if isinstance(item, dict) else getattr(item, 'match_text', '')
        return match_text and match_text in section_text

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

    def compile_final_review(self, section_analyses: list, paper_context: dict) -> dict:
        """Compile a final high-level review synthesizing all sections."""
        try:
            with dspy.settings.context(lm=self.engine):
                final_review = self.final_reviewer(
                    section_analyses=section_analyses,
                    paper_context=paper_context
                )
                
                return {
                    'overall_assessment': str(final_review.overall_assessment),
                    'key_strengths': [str(s) for s in (final_review.key_strengths if isinstance(final_review.key_strengths, list) else [final_review.key_strengths])],
                    'key_weaknesses': [str(w) for w in (final_review.key_weaknesses if isinstance(final_review.key_weaknesses, list) else [final_review.key_weaknesses])],
                    'recommendations': [str(r) for r in (final_review.recommendations if isinstance(final_review.recommendations, list) else [final_review.recommendations])],
                    'model_info': {  # Add model information
                        'engine': str(self.engine),
                        'using_cot': isinstance(self.reviewer, dspy.ChainOfThought)
                    }
                }
        except Exception as e:
            logger.error(f"Error compiling final review: {str(e)}")
            return {
                'overall_assessment': "Error generating final review",
                'key_strengths': [],
                'key_weaknesses': [],
                'recommendations': [],
                'model_info': {
                    'engine': str(self.engine),
                    'using_cot': isinstance(self.reviewer, dspy.ChainOfThought)
                }
            }


if __name__ == "__main__":
    import os
    # Test the reviewer with DSPy's LM directly
    # lm = dspy.LM('openai/gpt-4o-mini')
    lm = dspy.LM(
        'openrouter/qwen/qwq-32b-preview',
        api_base="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.7,
    )
    reviewer = SectionReviewer(
        lm
    )
    
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
