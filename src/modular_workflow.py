from enum import Enum
from pathlib import Path
from datetime import datetime
import json
import copy

from dspy_workflow_builder.parse_lm_config import DEFAULT_CONFIGS
from dspy_workflow_builder.processors import BaseProcessor
from dspy_workflow_builder.pipeline import Pipeline, PipelineConfig
from dspy_workflow_builder.steps import BaseStep
from dspy_workflow_builder.utils.logging import setup_logging

from ai_pi.analysis.generate_storm_context import StormContextGenerator
from ai_pi.analysis.summarizer import Summarizer
from ai_pi.analysis.reviewer import Reviewer
from ai_pi.document_handling.document_output import output_commented_document
from ai_pi.document_handling.document_ingestion import extract_document_history

class WorkflowStepType(Enum):
    """Types of workflow steps available"""
    DOCUMENT_EXTRACTION = "document_extraction"
    DOCUMENT_ANALYSIS = "document_analysis"
    TOPIC_CONTEXT = "topic_context"
    DOCUMENT_REVIEW = "document_review"
    OUTPUT_GENERATION = "output_generation"


class DocumentExtractionProcessor(BaseProcessor):
    """Handles document extraction and validation"""
    def _process(self, data: dict) -> dict:
        document_history = extract_document_history(
            data['input_doc_path'], 
            write_to_file=False
        )
        
        if not document_history or not isinstance(document_history, dict):
            raise ValueError("Document extraction failed or invalid format")
            
        required_keys = ['sections', 'comments', 'revisions', 'metadata']
        if not all(key in document_history for key in required_keys):
            raise ValueError(f"Missing required keys in document_history")
            
        return {'document_history': document_history}


class DocumentAnalysisProcessor(BaseProcessor):
    """Orchestrates document analysis using Summarizer"""
    def _process(self, data: dict) -> dict:
        summarizer = Summarizer(verbose=self.step.verbose)
        topic, document_structure = summarizer.analyze_sectioned_document(
            data['document_history']
        )
        return {
            'topic': topic,
            'document_structure': document_structure,
            'hierarchical_summary': document_structure['hierarchical_summary']
        }


class TopicContextProcessor(BaseProcessor):
    """Orchestrates topic context generation using STORM"""
    def _process(self, data: dict) -> dict:
        context_generator = StormContextGenerator(
            output_dir=data['output_dir']
        )
        topic_context = context_generator.generate_context(data['topic'])
        return {'topic_context': topic_context}


class DocumentReviewProcessor(BaseProcessor):
    """Orchestrates document review using Reviewer"""
    def _process(self, data: dict) -> dict:
        reviewer = Reviewer(verbose=self.step.verbose)
        reviewed_document = reviewer.review_document(
            data['document_history'],
            data['topic_context'],
            data['hierarchical_summary']
        )
        return {'reviewed_document': reviewed_document}


class OutputProcessor(BaseProcessor):
    """Handles final document output generation"""
    def _process(self, data: dict) -> dict:
        paper_title = data['paper_title']
        output_dir = data['output_dir']
        
        # Write reviewed document to JSON
        output_json = output_dir / f"{paper_title}_reviewed.json"
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(data['reviewed_document'], f, indent=4)
            
        # Save LM configuration state
        lm_config_state = {
            "timestamp": data['timestamp'],
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
            
        # Generate output document
        output_path = output_dir / f"{paper_title}_reviewed.docx"
        output_commented_document(
            input_doc_path=data['input_doc_path'],
            review_struct=data['reviewed_document'],
            output_doc_path=output_path,
            match_threshold=90
        )
        
        return {
            'output_paths': {
                'json': str(output_json),
                'docx': str(output_path),
                'pdf': str(output_dir / f"{paper_title}.pdf"),
                'markdown': str(output_dir / f"{paper_title}.md"),
                'lm_config': str(config_json)
            }
        }


def create_pipeline(verbose: bool = False) -> Pipeline:
    """Create the document processing pipeline"""
    steps = [
        BaseStep(
            step_type=WorkflowStepType.DOCUMENT_EXTRACTION,
            processor_class=DocumentExtractionProcessor,
            output_key="document_history"
        ),
        BaseStep(
            step_type=WorkflowStepType.DOCUMENT_ANALYSIS,
            processor_class=DocumentAnalysisProcessor,
            output_key="document_structure",
            depends_on=["document_history"]
        ),
        BaseStep(
            step_type=WorkflowStepType.TOPIC_CONTEXT,
            processor_class=TopicContextProcessor,
            output_key="topic_context",
            depends_on=["topic"]
        ),
        BaseStep(
            step_type=WorkflowStepType.DOCUMENT_REVIEW,
            processor_class=DocumentReviewProcessor,
            output_key="reviewed_document",
            depends_on=["document_history", "topic_context", "hierarchical_summary"]
        ),
        BaseStep(
            step_type=WorkflowStepType.OUTPUT_GENERATION,
            processor_class=OutputProcessor,
            output_key="output_paths",
            depends_on=["reviewed_document"]
        )
    ]
    return Pipeline(PipelineConfig(steps=steps, verbose=verbose))


class PaperReview:
    """Orchestrates the paper review workflow using a processing pipeline"""
    
    def __init__(self, verbose: bool = False, log_dir: str = "logs"):
        self.verbose = verbose
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = setup_logging(self.log_dir, timestamp, "paper_review")
        
        self.pipeline = create_pipeline(verbose=verbose)

    def review_paper(self, input_doc_path: str) -> dict:
        """Execute the document review pipeline"""
        self.logger.info(f"Starting review of document: {input_doc_path}")
        
        try:
            paper_title = Path(input_doc_path).stem
            base_dir = Path('processed_documents').resolve()
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_dir = base_dir / f"{paper_title}_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)

            results = self.pipeline.execute({
                'input_doc_path': input_doc_path,
                'paper_title': paper_title,
                'output_dir': output_dir,
                'timestamp': timestamp
            })
            
            return {

                'paper_context': results['document_structure']['hierarchical_summary']['document_summary']['document_analysis'],
                'document_structure': results['document_structure'],
                'reviews': results['reviewed_document']['reviews'],
                'output_dir': str(output_dir),
                'output_files': results['output_paths']
            }
            
        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            raise ValueError(f"Error processing document: {str(e)}")