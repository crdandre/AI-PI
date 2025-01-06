import logging
import os
from datetime import datetime
from pathlib import Path
import dspy
import json

from ai_pi.analysis.summarizer import Summarizer
from ai_pi.analysis.reviewer import Reviewer
from ai_pi.document_handling.document_output import output_commented_document
from ai_pi.document_handling.document_ingestion import extract_document_history


class PaperReview:
    def __init__(self, lm, verbose=False, log_dir="logs", reviewer_class="Predict"):
        """Initialize with DSPy LM:
        - lm: DSPy LM for both summarizer and reviewer
        - reviewer_class: Type of reviewer to use ("ReAct", "ChainOfThought", or "Predict")
        """
        self.summarizer = Summarizer(lm=lm, verbose=verbose)
        self.section_reviewer = Reviewer(
            engine=lm,
            reviewer_class=reviewer_class,
            verbose=verbose
        )
        
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
    
    def review_paper(self, input_doc_path: str) -> dict:
        """Two-step review process with proper section handling"""
        self.logger.info(f"Starting review of document: {input_doc_path}")
        
        try:
            # Create default output directory structure
            paper_title = Path(input_doc_path).stem
            base_dir = Path('processed_documents').resolve()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = base_dir / f"{paper_title}_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)

            # Extract document content
            document_history = extract_document_history(
                input_doc_path,
                lm=self.summarizer.lm,
                write_to_file=False
            )
            
            if not document_history:
                raise ValueError("Document extraction failed - no content returned")

            # Validate document_history structure
            if not isinstance(document_history, dict):
                raise ValueError(f"Invalid document_history format: {type(document_history)}")
            
            required_keys = ['sections', 'comments', 'revisions', 'metadata']
            missing_keys = [key for key in required_keys if key not in document_history]
            if missing_keys:
                raise ValueError(f"Missing required keys in document_history: {missing_keys}")
            
            # 1. Use Summarizer to analyze document structure
            self.logger.info("Analyzing document structure...")
            document_structure = self.summarizer.analyze_sectioned_document(document_history)
            
            # 2. Use Reviewer to handle section-by-section review and final compilation
            self.logger.info("Starting document review...")
            reviewed_document = self.section_reviewer.review_document(document_history)
            
            # Now write the complete reviewed document to JSON
            output_json = output_dir / f"{paper_title}_reviewed.json"
            with open(output_json, 'w', encoding='utf-8') as f:
                json.dump(reviewed_document, f, indent=4)
            self.logger.info(f"Complete review written to: {output_json}")
            
            # Generate output document with matching name pattern
            output_path = output_dir / f"{paper_title}_reviewed.docx"
            
            self.logger.info(f"Generating reviewed document at: {output_path}")
            output_commented_document(
                input_doc_path=input_doc_path,
                document_review_items=reviewed_document['reviews'],
                output_doc_path=output_path
            )
            self.logger.debug("Document generation complete")
            
            return {
                'paper_context': document_structure['hierarchical_summary']['document_summary']['document_analysis'],
                'document_structure': document_structure,
                'reviews': reviewed_document['reviews'],
                'output_dir': str(output_dir),
                'output_files': {
                    'pdf': str(output_dir / f"{paper_title}.pdf"),
                    'markdown': str(output_dir / f"{paper_title}.md"),
                    'json': str(output_json),
                    'reviewed_docx': str(output_path)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            raise ValueError(f"Error processing document: {str(e)}")


if __name__ == "__main__":
    # Example usage - using the same paths as document_output.py test    
    input_path = "examples/ScolioticFEPaper_v7.docx"
    # input_path = "examples/example_abstract.docx"

    # openrouter_model = 'openrouter/openai/gpt-4o'
    openrouter_model = 'openrouter/anthropic/claude-3.5-sonnet:beta'
    
    # Initialize and run
    lm = dspy.LM(
        openrouter_model,
        api_base="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.9,
    )
    dspy.settings.configure(lm=lm)
    
    paper_review = PaperReview(
        lm=lm,
        verbose=True,
        reviewer_class="Predict",
    )
    
    try:
        output = paper_review.review_paper(input_doc_path=input_path)
        print(f"Successfully created reviewed document")
    except Exception as e:
        print(f"Error processing document: {str(e)}")