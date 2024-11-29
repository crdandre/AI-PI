from llama_index.llms.nvidia import NVIDIA

from ai_pi.llm.context import ContextAgent
from ai_pi.llm.summarizer import PaperSummarizer
from ai_pi.llm.reviewer import SectionReviewer
from ai_pi.documents.document_output import output_commented_document


class PaperReview:
    def __init__(self, llm, verbose=False):
        self.context_agent = ContextAgent(llm, verbose=verbose)
        self.summarizer = PaperSummarizer(llm)
        self.section_reviewer = SectionReviewer(llm)
        self.verbose = verbose
    
    def review_paper(self, paper_text: str, output_path: str = None) -> dict:
        """Two-step review process with proper section handling"""
        
        # 1. Use ContextAgent to identify sections
        document_structure = self.context_agent.analyze_document(paper_text)
        
        # 2. Create high-level summary for review context
        paper_context = self.summarizer.create_review_context(paper_text)
        
        # 3. Review each section with full context
        section_reviews = []
        match_strings = []
        comments = []
        revisions = []
        
        for section in document_structure['section_summaries']:
            review = self.section_reviewer.review_section(
                section_text=section['original_text'],
                section_type=section['section_title'],
                paper_context=paper_context
            )
            section_reviews.append(review)
            
            # Extract review points for document output
            review_points = self._extract_review_points(review['review'])
            match_strings.extend(review_points['match_strings'])
            comments.extend(review_points['comments'])
            revisions.extend(review_points['revisions'])
        
        # Create output structure
        review_output = {
            'match_strings': match_strings,
            'comments': comments,
            'revisions': revisions
        }
        
        # Generate output document if path provided
        if output_path:
            if self.verbose:
                print(f"Generating reviewed document at: {output_path}")
            output_commented_document(
                input_doc_path=paper_text,  # Assuming this is a path
                document_review_items=review_output,
                output_doc_path=output_path
            )
        
        return {
            'paper_context': paper_context,
            'section_reviews': section_reviews,
            'document_structure': document_structure,
            'review_output': review_output
        }
    
    def _extract_review_points(self, review_text: str) -> dict:
        """
        Extract review points from the review text into the format needed
        for document output.
        
        Expected format from review_text:
        - Required revisions
        - Suggested enhancements
        - Missing elements
        """
        # Initialize lists
        match_strings = []
        comments = []
        revisions = []
        
        # Split review into sections (this is a simple example;
        # might need more robust parsing based on actual review format)
        sections = review_text.split('\n\n')

        """
        ^^^
        Generated summary tree with 14 sections
Traceback (most recent call last):
  File "/home/christian/projects/agents/ai_pi/src/ai_pi/workflow2.py", line 119, in <module>
    output = paper_review.review_paper(
  File "/home/christian/projects/agents/ai_pi/src/ai_pi/workflow2.py", line 40, in review_paper
    review_points = self._extract_review_points(review['review'])
  File "/home/christian/projects/agents/ai_pi/src/ai_pi/workflow2.py", line 86, in _extract_review_points
    sections = review_text.split('\n\n')
  File "/home/christian/projects/agents/ai_pi/.venv/lib/python3.10/site-packages/pydantic/main.py", line 856, in __getattr__
    raise AttributeError(f'{type(self).__name__!r} object has no attribute {item!r}')
AttributeError: 'CompletionResponse' object has no attribute 'split'
        
        """
        
        for section in sections:
            if 'Required revisions' in section or 'Suggested enhancements' in section:
                lines = section.split('\n')
                for line in lines[1:]:  # Skip header
                    if line.strip():
                        # Extract the text to be revised
                        # This assumes some consistent formatting in the review text
                        parts = line.split(': ', 1)
                        if len(parts) == 2:
                            original, suggestion = parts
                            match_strings.append(original.strip())
                            comments.append(f"Suggested revision: {suggestion.strip()}")
                            revisions.append(suggestion.strip())
        
        return {
            'match_strings': match_strings,
            'comments': comments,
            'revisions': revisions
        }


if __name__ == "__main__":
    # Example usage
    with open("processed_documents/ScolioticFEPaper_v7_processed.txt", "r") as file:
        paper_text = file.read()
    
    paper_review = PaperReview(
        llm=NVIDIA(model="meta/llama3-70b-instruct"), 
        verbose=True
    )
    
    output = paper_review.review_paper(
        paper_text=paper_text,
        output_path="reviewed_documents/ScolioticFEPaper_v7_reviewed.docx"
    )