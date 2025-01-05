import logging
import os
from datetime import datetime
from pathlib import Path
import dspy

from llama_index.llms.nvidia import NVIDIA

from ai_pi.analysis.context import ContextAgent
from ai_pi.analysis.summarizer import PaperSummarizer
from ai_pi.analysis.reviewer import SectionReviewer
from ai_pi.document_handling.document_output import output_commented_document
from ai_pi.document_handling.document_ingestion import extract_document_history


class PaperReview:
    def __init__(self, llm=None, lm=None, verbose=False, log_dir="logs", reviewer_class="Predict"):
        """Initialize with both LLM types:
        - llm: LlamaIndex LLM for context and summarizer
        - lm: DSPy LM for reviewer
        - reviewer_class: Type of reviewer to use ("ReAct", "ChainOfThought", or "Predict")
        """
        if llm is None:
            raise ValueError("Must provide llm parameter for context and summarizer")
        if lm is None:
            raise ValueError("Must provide lm parameter for reviewer")
            
        # LlamaIndex components
        self.context_agent = ContextAgent(llm, verbose=verbose)
        self.summarizer = PaperSummarizer(llm)
        
        # DSPy component with reviewer_class parameter
        self.section_reviewer = SectionReviewer(
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
    
    def review_paper(self, input_doc_path: str, output_path: str = None) -> dict:
        """Two-step review process with proper section handling"""
        self.logger.info(f"Starting review of document: {input_doc_path}")
        self.logger.debug(f"Full processing path: {input_doc_path}")
        
        # Use document_ingestion to properly extract text content and sections
        try:
            # Pass the DSPy LM to document ingestion
            document_history = extract_document_history(
                input_doc_path,
                lm=self.section_reviewer.engine  # Use the same LLM as the reviewer
            )
            self.logger.debug(f"Successfully extracted text content")
            self.logger.debug(f"Document history type: {type(document_history)}")
            self.logger.debug(f"Document history keys: {document_history.keys() if isinstance(document_history, dict) else 'Not a dict'}")
        except Exception as e:
            error_msg = f"Failed to process document: {str(e)}\nType: {type(e)}\nFull error: {repr(e)}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            # Use input path directly
            doc_path = input_doc_path
            self.logger.debug(f"Using document path: {doc_path}")
            
            # 1. Use ContextAgent to identify sections and create structure
            self.logger.info("Analyzing document structure...")
            if not isinstance(document_history, dict) or 'sections' not in document_history:
                raise ValueError(f"Invalid document_history format: {type(document_history)}")
            
            document_structure = self.context_agent.analyze_document(document_history['sections'])
            self.logger.debug(f"Document structure type: {type(document_structure)}")
            self.logger.debug(f"Document structure keys: {document_structure.keys() if isinstance(document_structure, dict) else 'Not a dict'}")
            
            # Now we can use the sections directly from document_history
            sections = document_history['sections']
            self.logger.debug(f"Number of sections found: {len(sections)}")
            self.logger.debug(f"Section types: {[s.get('type', 'unknown') for s in sections]}")
            
            section_reviews = []
            match_strings = []
            comments = []
            revisions = []
            section_analyses = []
            
            total_sections = len(sections)
            self.logger.info(f"Reviewing {total_sections} identified sections...")
            
            for i, section in enumerate(sections, 1):
                self.logger.info(f"Reviewing section {i}/{total_sections}: {section['type']}")
                
                try:
                    paper_context = {
                        'paper_summary': str(document_structure['document_summary']['document_analysis'].text),
                        'section_type': section['type']  # Now we have confident section types
                    }
                    
                    review = self.section_reviewer.review_section(
                        section_text=section['text'],
                        section_type=section['type'],
                        paper_context=paper_context
                    )
                    
                    self.logger.debug(f"Full review response: {review}")
                    section_reviews.append(review)
                    
                    # Extract review data and add validation logging
                    if review:
                        current_matches = review.get('match_strings', [])
                        current_comments = review.get('comments', [])
                        current_revisions = review.get('revisions', [])
                        
                        self.logger.info(f"Found {len(current_matches)} review points in section {i}")
                        # Add logging for section-specific validation
                        self.logger.debug(f"Section type: {section['type']}")
                        self.logger.debug(f"Initial analysis: {review.get('initial_analysis', 'N/A')}")
                        self.logger.debug(f"Reflection: {review.get('reflection', 'N/A')}")
                        
                        match_strings.extend(current_matches)
                        comments.extend(current_comments)
                        revisions.extend(current_revisions)
                        
                        # Store enhanced section analysis for final review
                        section_analyses.append({
                            'section_type': section['type'],
                            'analysis': {
                                'initial_analysis': review.get('initial_analysis', ''),
                                'reflection': review.get('reflection', ''),
                                'review_items': list(zip(current_matches, current_comments, current_revisions))
                            }
                        })
                    else:
                        self.logger.warning(f"No review points found for section {i} - validation may have failed")
                        
                except Exception as e:
                    self.logger.error(f"Error reviewing section {i}: {str(e)}")
                    continue
            
            # Generate final high-level review
            final_review = self.section_reviewer.compile_final_review(
                section_analyses=section_analyses,
                paper_context=document_structure['document_summary']
            )
            
            # Create complete output structure
            review_output = {
                'match_strings': match_strings,
                'comments': comments,
                'revisions': revisions,
                'high_level_review': final_review,  # Add synthesized review
                'section_analyses': section_analyses  # Keep section-level analyses if needed
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
        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            raise ValueError(f"Error processing document: {str(e)}")


if __name__ == "__main__":
    from llama_index.llms.openai import OpenAI
    
    # Example usage - using the same paths as document_output.py test    
    # input_path = "examples/ScolioticFEPaper_v7.docx"
    input_path = "examples/example_abstract.docx"
    output_path = "examples/test_output_workflow2.docx"

    # Initialize both LLM types
    llm = OpenAI(model="gpt-4o-mini")  # For context and summarizer
    lm = dspy.LM('openai/gpt-4o-mini')  # For reviewer
    # lm = dspy.LM(
    #     'openrouter/openai/o1-mini',
    #     api_base="https://openrouter.ai/api/v1",
    #     api_key=os.getenv("OPENROUTER_API_KEY"),
    #     temperature=1.0,
    #     max_tokens=9999
    # )
    
    paper_review = PaperReview(
        llm=llm,
        lm=lm,
        verbose=True,
        reviewer_class="Predict",
    )
    
    try:
        output = paper_review.review_paper(
            input_doc_path=input_path,
            output_path=output_path
        )
        print(f"Successfully created reviewed document at {output_path}")
    except Exception as e:
        print(f"Error processing document: {str(e)}")