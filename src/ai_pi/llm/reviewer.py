"""
This file contains the code for an agent which, given relevant context
from prior reviews and any other guidelines for the review process, can
create suggestions for comments and revisions, simulating the style of feedback
a PI would given when reviewing a paper

For now, some manual token management for fitting within context length...
"""
import os
import dspy

class ReviewerAgent:
    def __init__(
        self,
        llm=dspy.LM(
            "nvidia_nim/meta/llama3-70b-instruct",
            api_key=os.environ['NVIDIA_API_KEY'],
            api_base=os.environ['NVIDIA_API_BASE']
        ),
    ):
        self.llm = llm
        dspy.configure(lm=llm)
        self.call_review = dspy.ChainOfThought("question -> answer")
        self.call_revise = dspy.ChainOfThought(
            "question -> revision_list: list[list[str]]"
        )


class SectionReviewer:
    """Reviews individual sections with awareness of full paper context"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def review_section(
        self,
        section_text: str,
        section_type: str,
        paper_context: dict
    ) -> dict:
        """
        Review a section using the paper's overall context
        """
        prompt = """
        Review this {section_type} section in the context of the full paper.
        
        Paper Summary:
        {paper_summary}
        
        Section to Review:
        {section_text}
        
        Provide a detailed review addressing:
        1. Role Fulfillment
           - Does this section serve its intended role?
           - Does it support the paper's main claims?
           - Is it properly positioned in the larger narrative?
        
        2. Content Quality
           - Clarity and organization
           - Evidence quality
           - Methodology appropriateness
           - Results interpretation
        
        3. Integration
           - Connections to other sections
           - Support for overall arguments
           - Consistency with paper's approach
        
        4. Specific Improvements
           - Required revisions
           - Suggested enhancements
           - Missing elements
        
        Focus on how this section contributes to the paper's goals.
        """
        
        review = self.llm.complete(
            prompt.format(
                section_type=section_type,
                paper_summary=paper_context['paper_summary'],
                section_text=section_text
            )
        )
        
        return {
            'section_type': section_type,
            'review': review,
            'section_text': section_text
        }