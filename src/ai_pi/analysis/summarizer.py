class PaperSummarizer:
    """Creates a comprehensive paper summary to guide section-by-section review"""
    
    def __init__(self, llm):
        self.llm = llm
    
    def create_review_context(self, paper_text: str) -> dict:
        """
        Generate a summary focused on elements needed for review guidance
        """
        prompt = """
        Create a comprehensive summary of this academic paper to guide its review.
        Focus on:
        1. Core Contribution
           - Main claims/findings
           - Significance in field
           - Novel elements
        
        2. Research Approach
           - Key methodological choices
           - Data/evidence types
           - Analysis strategies
        
        3. Paper Structure
           - How arguments are built
           - Evidence flow
           - Key dependencies between sections
        
        4. Expected Standards
           - Critical elements to verify
           - Potential weak points to examine
           - Required supporting evidence
        
        Paper: {text}
        
        Provide a structured summary that will guide a detailed review.
        """
        
        summary = self.llm.complete(prompt.format(text=paper_text))
        
        return {
            'paper_summary': summary,
            'original_text': paper_text
        }