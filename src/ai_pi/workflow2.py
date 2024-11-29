import logging
import os
from datetime import datetime
from pathlib import Path
import dspy

from llama_index.llms.nvidia import NVIDIA

from ai_pi.llm.context import ContextAgent
from ai_pi.llm.summarizer import PaperSummarizer
from ai_pi.llm.reviewer import SectionReviewer
from ai_pi.documents.document_output import output_commented_document
from ai_pi.documents.document_ingestion import extract_document_history


class PaperReview:
    def __init__(self, llm=None, lm=None, verbose=False, log_dir="logs"):
        """Initialize with both LLM types:
        - llm: LlamaIndex LLM for context and summarizer
        - lm: DSPy LM for reviewer
        """
        if llm is None:
            raise ValueError("Must provide llm parameter for context and summarizer")
        if lm is None:
            raise ValueError("Must provide lm parameter for reviewer")
            
        # LlamaIndex components
        self.context_agent = ContextAgent(llm, verbose=verbose)
        self.summarizer = PaperSummarizer(llm)
        
        # DSPy component
        self.section_reviewer = SectionReviewer(lm, verbose=verbose)
        
        self.verbose = verbose
        
        # Setup logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create a unique log file for this review session
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"paper_review_{timestamp}.log"
        
        # Configure logging
        self.logger = logging.getLogger(f"paper_review_{timestamp}")
        self.logger.setLevel(logging.DEBUG)
        
        # File handler for detailed logging
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Console handler for abbreviated logging
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
    
    def review_paper(self, input_doc_path: str, output_path: str = None) -> dict:
        """Two-step review process with proper section handling"""
        self.logger.info(f"Starting review of document: {input_doc_path}")
        self.logger.debug(f"Full processing path: {input_doc_path}")
        
        # Use document_ingestion to properly extract text content
        try:
            paper_text = extract_document_history(input_doc_path)
            self.logger.debug(f"Successfully extracted text content")
        except Exception as e:
            error_msg = f"Failed to process document: {str(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Use input path directly
        doc_path = input_doc_path
        self.logger.debug(f"Using document path: {doc_path}")
        
        # 1. Use ContextAgent to identify sections and create structure
        self.logger.info("Analyzing document structure...")
        document_structure = self.context_agent.analyze_document(paper_text)
        self.logger.debug(f"Document structure: {document_structure}")
        
        # 2. Review each section using the existing document structure
        section_reviews = []
        match_strings = []
        comments = []
        revisions = []
        
        total_sections = len(document_structure['section_summaries'])
        self.logger.info(f"Reviewing {total_sections} sections...")
        
        for i, section in enumerate(document_structure['section_summaries'], 1):
            self.logger.info(f"Reviewing section {i}/{total_sections}: {section['section_title']}")
            self.logger.debug(f"Full section content: {section['original_text'][:200]}...")
            
            try:
                review = self.section_reviewer.review_section(
                    section_text=section['original_text'],
                    section_type=section['section_title'],
                    paper_context={'paper_summary': document_structure['document_summary']['document_analysis']}
                )
                
                self.logger.debug(f"Full review response: {review}")
                section_reviews.append(review)
                
                # Add review points if they exist
                if review and 'review' in review and review['review']:
                    current_matches = len(review['review'].get('match_strings', []))
                    self.logger.info(f"Found {current_matches} review points in section {i}")
                    self.logger.debug(f"Review points: {review['review']}")
                    
                    match_strings.extend(review['review'].get('match_strings', []))
                    comments.extend(review['review'].get('comments', []))
                    revisions.extend(review['review'].get('revisions', []))
                else:
                    self.logger.warning(f"No review points found for section {i}")
                    
            except Exception as e:
                self.logger.error(f"Error reviewing section {i}: {str(e)}")
                continue
        
        # Create output structure
        review_output = {
            'match_strings': match_strings,
            'comments': comments,
            'revisions': revisions
        }
        
        total_points = len(match_strings)
        self.logger.info(f"Review complete: {total_points} total review points generated")
        self.logger.debug(f"Full review output: {review_output}")
        
        # Generate output document if path provided
        if output_path:
            self.logger.info(f"Generating reviewed document at: {output_path}")
            output_commented_document(
                input_doc_path=doc_path,
                document_review_items=review_output,
                output_doc_path=output_path
            )
            self.logger.debug("Document generation complete")
        
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

    # Initialize both LLM types
    llm = OpenAI(model="gpt-4o-mini")  # For context and summarizer
    lm = dspy.LM('openai/gpt-4o-mini')  # For reviewer
    
    paper_review = PaperReview(
        llm=llm,
        lm=lm,
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