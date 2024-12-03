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
    revision = dspy.OutputField(desc="Complete revised text, leave empty if only commenting")
    comment_only = dspy.OutputField(desc="Boolean indicating if this should be just a comment without revision")

class ReviewerSignature(dspy.Signature):
    """Generate high-level scientific review feedback for a section of text.
    
    You are a senior academic reviewer with expertise in providing strategic feedback.
    For each issue you identify, carefully consider the most appropriate type of feedback:

    1. Comments only (no revision) when:
       - Asking for clarification or additional information
       - Suggesting broader methodological changes
       - Questioning scientific assumptions
       - Requesting additional analyses
       - Noting potential implications
    
    2. Revisions only when:
       - Fixing clear grammatical errors
       - Improving sentence structure
       - Correcting technical terminology
       - Adding missing references
    
    3. Both comment and revision when:
       - Suggesting structural improvements while explaining why
       - Addressing clarity issues with specific suggestions
       - Improving scientific precision while explaining the importance
       - Enhancing methodology descriptions with rationale
    
    You must identify at least 3-5 issues per section, considering:
    - Scientific rigor and methodology
    - Clarity and logical flow
    - Technical accuracy
    - Writing style and effectiveness
    - Grammar and structure (when impacting comprehension)
    
    For each issue, reflect on whether it needs a comment, revision, or both.
    """
    section_text = dspy.InputField(desc="The text content of the section to review")
    section_type = dspy.InputField(desc="The type of section (e.g., Methods, Results)")
    context = dspy.InputField(desc="Additional context about the paper")
    initial_analysis = dspy.OutputField(desc="Initial high-level analysis of scientific concerns")
    reflection = dspy.OutputField(
        desc="Reflection on feedback types chosen and their appropriateness"
    )
    review_items = dspy.OutputField(
        desc="List of review items with carefully chosen feedback types", 
        format=list[ReviewItem]
    )

class CommunicationReview(dspy.Signature):
    """Evaluate writing style, narrative clarity, and communication effectiveness.
    
    Focus on high-level writing aspects like narrative flow, clarity of explanations,
    and effectiveness of scientific communication. Avoid minor grammar/spelling issues.
    
    Example Output:
        writing_assessment: "The paper presents complex ideas clearly, but lacks smooth 
            transitions between sections. Technical concepts are well-explained for the 
            target audience."
        narrative_strengths: ["Clear problem statement", "Effective use of examples"]
        narrative_weaknesses: ["Section transitions need work", "Methods section assumes 
            too much background knowledge"]
        style_recommendations: ["Add transition paragraphs between major sections", 
            "Include more context for technical terms"]
    """
    section_analyses = dspy.InputField(desc="List of section-level analyses")
    paper_context = dspy.InputField(desc="Overall paper context including target audience")
    writing_assessment = dspy.OutputField(desc="Overall assessment of writing quality and clarity")
    narrative_strengths = dspy.OutputField(desc="Key strengths in narrative and communication")
    narrative_weaknesses = dspy.OutputField(desc="Areas for improvement in narrative and communication")
    style_recommendations = dspy.OutputField(desc="Specific recommendations for improving writing style")

class FinalReviewSignature(dspy.Signature):
    """Generate final synthesized review across all sections."""
    section_analyses = dspy.InputField(desc="List of section-level analyses")
    paper_context = dspy.InputField(desc="Overall paper context")
    overall_assessment = dspy.OutputField(desc="Synthesized scientific assessment of the entire paper")
    key_strengths = dspy.OutputField(desc="Major scientific strengths identified")
    key_weaknesses = dspy.OutputField(desc="Major scientific weaknesses identified")
    recommendations = dspy.OutputField(desc="High-priority recommendations for improvement")
    communication_review = dspy.OutputField(desc="Assessment of writing style and clarity")
    style_recommendations = dspy.OutputField(desc="Recommendations for improving communication")

class SectionReviewer(dspy.Module):
    """Reviews individual sections with awareness of full paper context"""
    
    def __init__(self, engine: Union[dspy.dsp.LM, dspy.dsp.HFModel], verbose=False):
        super().__init__()
        
        # Instead of using examples in ChainOfThought initialization,
        # we should use a prompt template or modify the signature
        self.reviewer = dspy.ChainOfThought(ReviewerSignature)
        self.final_reviewer = dspy.ChainOfThought(FinalReviewSignature)
        self.communication_reviewer = dspy.ChainOfThought(CommunicationReview)
        
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
                    "revision": "",  # Empty since this is a suggestion for the author to implement
                    "comment_only": False
                }
            ]
        }
        
        self.engine = engine
        self.verbose = verbose

    def review_paper(self, sections: list[dict], paper_context: dict) -> dict:
        """Review paper by getting overall view first, then diving into sections"""
        try:
            with dspy.settings.context(lm=self.engine):
                # First get the overall paper review
                final_review = self.final_reviewer(
                    section_analyses=sections,  # Pass all sections for initial overview
                    paper_context=paper_context
                )
                
                # Update context with the overall analysis
                enhanced_context = {
                    **paper_context,
                    'overall_assessment': final_review.overall_assessment,
                    'key_strengths': final_review.key_strengths,
                    'key_weaknesses': final_review.key_weaknesses
                }

                # Now review individual sections with enhanced context
                section_reviews = []
                for section in sections:
                    review = self.review_section(
                        section['text'],
                        section['type'],
                        enhanced_context  # Use enhanced context for better section reviews
                    )
                    section_reviews.append(review)

                return {
                    'section_reviews': section_reviews,
                    'final_review': final_review
                }

        except Exception as e:
            logger.error(f"Error in paper review: {str(e)}")
            return self._create_empty_review()

    def review_section(self, section_text: str, section_type: str, paper_context: dict) -> dict:
        """
        Main entry point for reviewing sections. Handles error cases and logging.
        """
        try:
            if self.verbose:
                print(f"Reviewing {section_type} section...")
            
            result = self.forward(section_text, section_type, paper_context)
            
            # Return the result directly without unwrapping
            return result
                
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
        review_items = {
            'match_strings': [],
            'comments': [],
            'revisions': []
        }
        
        try:
            parsed_response = json.loads(review_json) if isinstance(review_json, str) else review_json
            
            for item in parsed_response.get('review_items', []):
                match_text = str(item.get('match_text', '')).strip()
                
                if match_text and match_text in section_text:
                    comment = str(item.get('comment', ''))
                    revision = str(item.get('revision', ''))
                    comment_only = item.get('comment_only', False)
                    
                    # Always include the comment
                    review_items['match_strings'].append(match_text)
                    review_items['comments'].append(comment)
                    
                    # Include revision only if it's not a comment-only item and has content
                    review_items['revisions'].append(revision if not comment_only and revision else '')
                    
        except Exception as e:
            logger.error(f"Error processing review items: {str(e)}")
            
        return review_items

    def compile_final_review(self, section_analyses: list, paper_context: dict) -> dict:
        """Compile a final high-level review synthesizing all sections."""
        try:
            with dspy.settings.context(lm=self.engine):
                # Get scientific review
                final_review = self.final_reviewer(
                    section_analyses=section_analyses,
                    paper_context=paper_context
                )
                
                # Get communication review
                comm_review = self.communication_reviewer(
                    section_analyses=section_analyses,
                    paper_context=paper_context
                )
                
                return {
                    'overall_assessment': str(final_review.overall_assessment),
                    'key_strengths': [str(s) for s in (final_review.key_strengths if isinstance(final_review.key_strengths, list) else [final_review.key_strengths])],
                    'key_weaknesses': [str(w) for w in (final_review.key_weaknesses if isinstance(final_review.key_weaknesses, list) else [final_review.key_weaknesses])],
                    'recommendations': [str(r) for r in (final_review.recommendations if isinstance(final_review.recommendations, list) else [final_review.recommendations])],
                    'communication_review': {
                        'writing_assessment': str(comm_review.writing_assessment),
                        'narrative_strengths': [str(s) for s in (comm_review.narrative_strengths if isinstance(comm_review.narrative_strengths, list) else [comm_review.narrative_strengths])],
                        'narrative_weaknesses': [str(w) for w in (comm_review.narrative_weaknesses if isinstance(comm_review.narrative_weaknesses, list) else [comm_review.narrative_weaknesses])],
                        'style_recommendations': [str(r) for r in (comm_review.style_recommendations if isinstance(comm_review.style_recommendations, list) else [comm_review.style_recommendations])]
                    }
                }
        except Exception as e:
            logger.error(f"Error compiling final review: {str(e)}")
            return {
                'overall_assessment': "Error generating final review",
                'key_strengths': [],
                'key_weaknesses': [],
                'recommendations': [],
                'communication_review': {
                    'writing_assessment': "Error generating communication review",
                    'narrative_strengths': [],
                    'narrative_weaknesses': [],
                    'style_recommendations': []
                }
            }

    def _build_enhanced_context(self, paper_context: dict, section_type: str) -> str:
        """Build richer context including paper-level insights"""
        context_parts = []
        
        # Basic paper info
        context_parts.append(paper_context.get('paper_summary', ''))
        
        # Overall assessment
        if 'overall_assessment' in paper_context:
            context_parts.append(f"Overall assessment: {paper_context['overall_assessment']}")
        
        # Key points
        for field in ['key_strengths', 'key_weaknesses', 'recommendations']:
            if field in paper_context:
                items = paper_context[field]
                if items:
                    context_parts.append(f"{field.replace('_', ' ').title()}:")
                    context_parts.extend([f"- {item}" for item in items])
        
        return "\n".join(context_parts)

    def _get_section_specific_guidance(
        self,
        section_type: str,
        weaknesses: list,
        recommendations: list
    ) -> str:
        """Generate section-specific guidance based on paper-level insights"""
        # Filter relevant weaknesses and recommendations for this section
        relevant_items = [
            item for item in weaknesses + recommendations
            if section_type.lower() in item.lower()
        ]
        
        if relevant_items:
            return "Key considerations for this section:\n" + "\n".join(
                f"- {item}" for item in relevant_items
            )
        return ""

    def _check_section_alignment(
        self,
        section_text: str,
        section_type: str,
        paper_context: dict
    ) -> dict:
        """Check how well section aligns with paper-level goals"""
        try:
            # Use a simpler prompt to check alignment
            alignment_prompt = f"""
            Given the paper's main goals:
            {paper_context.get('overall_assessment', '')}
            
            Check how well this {section_type} section supports those goals.
            """
            
            with dspy.settings.context(lm=self.engine):
                alignment = self.reviewer(
                    section_text=section_text,
                    section_type=section_type,
                    context=alignment_prompt
                )
                
            return {
                'supports_goals': alignment.initial_analysis,
                'alignment_issues': alignment.reflection
            }
        except Exception as e:
            logger.warning(f"Alignment check failed: {str(e)}")
            return {}


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
    # Convert Prediction to dict before JSON serialization
    review_dict = {
        'review': review.review if hasattr(review, 'review') else review
    }
    print(json.dumps(review_dict, indent=2))