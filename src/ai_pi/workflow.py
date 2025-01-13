import logging
import os
from datetime import datetime
from pathlib import Path
import dspy
import json
import copy
from ai_pi.lm_config import DEFAULT_CONFIGS

from ai_pi.analysis.generate_storm_context import StormContextGenerator
from ai_pi.analysis.summarizer import Summarizer
from ai_pi.analysis.reviewer import Reviewer
from ai_pi.document_handling.document_output import output_commented_document
from ai_pi.document_handling.document_ingestion import extract_document_history
from ai_pi.utils.logging import setup_logging


class PaperReview:
    def __init__(self, verbose=False, log_dir="logs", reviewer_class="Predict"):
        """Initialize components using task-specific LM configurations"""
        # Initialize components with task-specific LMs
        self.summarizer = Summarizer(verbose=verbose)
        self.section_reviewer = Reviewer(verbose=verbose, validate_reviews=False)
        self.verbose = verbose
        
        # Setup logging
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = setup_logging(self.log_dir, timestamp, "paper_review")
        
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

            # Save LM configuration state
            lm_config_state = {
                "timestamp": timestamp,
                "configs": {
                    task: {
                        "model_name": config.model_name,
                        "temperature": config.temperature,
                        "api_base": config.api_base,
                        "max_tokens": config.max_tokens
                    } for task, config in copy.deepcopy(DEFAULT_CONFIGS).items()
                }
            }
            
            config_json = output_dir / "lm_config_state.json"
            with open(config_json, 'w', encoding='utf-8') as f:
                json.dump(lm_config_state, f, indent=4)
            self.logger.info(f"LM configuration state saved to: {config_json}")

            # Extract document content
            document_history = extract_document_history(
                input_doc_path,
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
            topic, document_structure = self.summarizer.analyze_sectioned_document(document_history)
            
            # 2. Use STORM to create an informed and concise topic context
            self.logger.info("Generating topic context...")
            topic_context = StormContextGenerator(
                output_dir=output_dir
            ).generate_context(topic)
            self.logger.info("Topic context generation complete")

            # 3. Use Reviewer to handle section-by-section review and final compilation
            self.logger.info("Starting document review...")
            reviewed_document = self.section_reviewer.review_document(document_history, topic_context)
            self.logger.info("Document review complete")
            
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
                review_struct=reviewed_document,
                output_doc_path=output_path,
                match_threshold=90
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
                    'reviewed_docx': str(output_path),
                    'lm_config': str(config_json)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            raise ValueError(f"Error processing document: {str(e)}")


if __name__ == "__main__":
    # Example usage with different configuration approaches
    input_path = "examples/NormativeFEPaper_v7.docx"
    
    paper_review = PaperReview(verbose=True)
    
    try:
        output = paper_review.review_paper(input_doc_path=input_path)
        print(f"Successfully created reviewed document")
    except Exception as e:
        print(f"Error processing document: {str(e)}")