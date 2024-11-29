from llama_index.llms.nvidia import NVIDIA

from ai_pi.llm.context import ContextAgent
from ai_pi.llm.summarizer import PaperSummarizer
from ai_pi.llm.reviewer import SectionReviewer
from ai_pi.documents.document_output import output_commented_document
from ai_pi.documents.document_ingestion import extract_document_history


class PaperReview:
    def __init__(self, llm, verbose=False):
        self.context_agent = ContextAgent(llm, verbose=verbose)
        self.summarizer = PaperSummarizer(llm)
        self.section_reviewer = SectionReviewer(llm)
        self.verbose = verbose
    
    def review_paper(self, input_doc_path: str, output_path: str = None) -> dict:
        """Two-step review process with proper section handling"""
        if self.verbose:
            print(f"\nProcessing document: {input_doc_path}")
        
        # Use document_ingestion to properly extract text content
        try:
            paper_text = extract_document_history(input_doc_path)
        except Exception as e:
            raise ValueError(f"Failed to process document: {str(e)}")
        
        # Use input path directly
        doc_path = input_doc_path
        if self.verbose:
            print(f"Using document path: {doc_path}")
        
        if self.verbose:
            print(f"Found original document: {doc_path}")
        
        # 1. Use ContextAgent to identify sections and create structure
        document_structure = self.context_agent.analyze_document(paper_text)
        
        # 2. Review each section using the existing document structure
        section_reviews = []
        match_strings = []
        comments = []
        revisions = []
        
        for i, section in enumerate(document_structure['section_summaries'], 1):
            review = self.section_reviewer.review_section(
                section_text=section['original_text'],
                section_type=section['section_title'],
                paper_context={'paper_summary': document_structure['document_summary']['document_analysis']}
            )
            if self.verbose:
                print(f"First 100 chars of review: {str(review['review'])[:100]}")
            section_reviews.append(review)
            
            # The review is already parsed, so just extend the lists directly
            if self.verbose:
                print("Adding review points...")
            
            match_strings.extend(review['review']['match_strings'])
            comments.extend(review['review']['comments'])
            revisions.extend(review['review']['revisions'])
            
            if self.verbose:
                print(f"Found {len(review['review']['match_strings'])} review points")
        
        # Create output structure
        review_output = {
            'match_strings': match_strings,
            'comments': comments,
            'revisions': revisions
        }
        
        if self.verbose:
            print(f"\nTotal review points generated: {len(match_strings)}")
        
        # Generate output document if path provided
        if output_path:
            if self.verbose:
                print(f"\nGenerating reviewed document at: {output_path}")
            output_commented_document(
                input_doc_path=doc_path,
                document_review_items=review_output,
                output_doc_path=output_path
            )
        
        return {
            'paper_context': document_structure['document_summary'],
            'section_reviews': section_reviews,
            'document_structure': document_structure,
            'review_output': review_output
        }


if __name__ == "__main__":
    from llama_index.llms.openai import OpenAI

    # Example usage - using the same paths as document_output.py test    
    input_path = "examples/example_abstract.docx"
    output_path = "examples/test_output_workflow2.docx"

    paper_review = PaperReview(
        llm=OpenAI(model="gpt-4o-mini"), 
        verbose=True
    )
    
    try:
        output = paper_review.review_paper(
            input_doc_path=input_path,
            output_path=output_path
        )
        print(f"Successfully created reviewed document at {output_path}")
    except Exception as e:
        print(f"Error processing document: {str(e)}")