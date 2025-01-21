import dspy
from dspy_workflow_builder.steps import LMStep
from dspy_workflow_builder.processors import LMProcessor
from dspy_workflow_builder.pipeline import Pipeline, PipelineConfig
from dspy_workflow_builder.lm_config import LMForTask


class SectionProcessor(LMProcessor):
    """LLM-based section processing"""
    
    class Signature(dspy.Signature):
        """Signature for section summarization"""
        section_type = dspy.InputField(desc="Type of section being summarized")
        text = dspy.InputField(desc="Section text to summarize")
        summary = dspy.OutputField(desc="Focused summary containing main points, evidence, findings, and significance")
    
    def _process(self, data: dict) -> dict:
        # Get sections from document_history
        document_history = data.get('document_history', {})
        sections = document_history.get('sections', [])
        summaries = []
        
        for section in sections:
            with dspy.context(lm=self.lm):
                result = self.predictors['Signature'](
                    section_type=section['section_type'],
                    text=section['text']
                )
                summaries.append({
                    'section_type': section['section_type'],
                    'summary': result.summary,
                    'match_strings': section['match_strings']
                })
        return {'section_summaries': summaries}


class RelationshipProcessor(LMProcessor):
    """Analyzes relationships between sections"""
    
    class Signature(dspy.Signature):
        """Signature for analyzing relationships between sections"""
        summaries = dspy.InputField(desc="Section summaries to analyze")
        analysis = dspy.OutputField(desc="Analysis of logical flow, argument development, and key dependencies")
    
    def _process(self, data: dict) -> dict:
        section_summaries = data.get('section_summaries', [])
        formatted_summaries = "\n\n".join(section_summaries)
        
        with dspy.context(lm=self.lm):
            result = self.predictors['Signature'](
                summaries=formatted_summaries
            )
        return {'relationship_analysis': result.analysis}


class DocumentProcessor(LMProcessor):
    """Creates document-level summary"""
    
    class Signature(dspy.Signature):
        """Signature for document-level summary"""
        section_summaries = dspy.InputField(desc="All section summaries")
        relationships = dspy.InputField(desc="Relationship analysis between sections")
        analysis = dspy.OutputField(desc="Comprehensive review-oriented summary")
    
    def _process(self, data: dict) -> dict:
        section_summaries = data.get('section_summaries', [])
        relationship_analysis = data.get('relationship_analysis', '')
        
        formatted_summaries = "\n".join(section_summaries)
        
        with dspy.context(lm=self.lm):
            result = self.predictors['Signature'](
                section_summaries=formatted_summaries,
                relationships=relationship_analysis
            )
        return {'document_analysis': result.analysis}


class TopicProcessor(LMProcessor):
    """Extracts document topic"""
    
    class Signature(dspy.Signature):
        """Signature for extracting concise document topics"""
        analysis = dspy.InputField(desc="Document analysis text to extract topic from")
        topic = dspy.OutputField(desc="Concise topic (10 words or less) capturing main document focus")
    
    def _process(self, data: dict) -> dict:
        document_analysis = data.get('document_analysis', '')
        with dspy.context(lm=self.lm):
            result = self.predictors['Signature'](
                analysis=document_analysis
            )
        return {'topic': result.topic.strip()}


def create_summarizer_pipeline(verbose: bool = False) -> Pipeline:
    steps = [
        LMStep(
            step_type="section",
            lm_name=LMForTask.SUMMARIZATION,
            processor_class=SectionProcessor,
            output_key="section_summaries",
            depends_on=["document_history"]
        ),
        LMStep(
            step_type="relationship",
            lm_name=LMForTask.SUMMARIZATION,
            processor_class=RelationshipProcessor,
            output_key="relationship_analysis",
            depends_on=["section_summaries"]
        ),
        LMStep(
            step_type="document",
            lm_name=LMForTask.SUMMARIZATION,
            processor_class=DocumentProcessor,
            output_key="document_analysis",
            depends_on=["section_summaries", "relationship_analysis"]
        ),
        LMStep(
            step_type="topic",
            lm_name=LMForTask.SUMMARIZATION,
            processor_class=TopicProcessor,
            output_key="topic",
            depends_on=["document_analysis"]
        )
    ]
    return Pipeline(PipelineConfig(steps=steps, verbose=verbose))


class Summarizer:
    """
    Creates a hierarchical summary of an input document. The input is
    a structured json file with full section text separated by key.
    """
    def __init__(self, verbose: bool = False):
        self.pipeline = create_summarizer_pipeline(verbose)

    def analyze_sectioned_document(self, document_json: dict) -> dict:
        """
        Create a true summary tree where each level summarizes its children,
        working with pre-sectioned content from document_json.
        """
        results = self.pipeline.execute(document_json)
        
        hierarchical_summary = {
            'topic': results['topic'],
            'document_summary': {'document_analysis': results['document_analysis']},
            'relationship_summary': {'relationship_analysis': results['relationship_analysis']},
            'section_summaries': results['section_summaries']
        }
        
        document_json['hierarchical_summary'] = hierarchical_summary
        return results['topic'], document_json


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    import json
    import logging
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(console_handler)
    
    with open("/home/christian/projects/agents/ai_pi/processed_documents/ScolioticFEPaper_v7_20250113_222724/ScolioticFEPaper_v7_reviewed.json", "r", encoding="utf-8") as f:
        document_json = json.load(f)
    
    context_agent = Summarizer(verbose=True)
    
    topic, analyzed_document = context_agent.analyze_sectioned_document(document_json)
    
    logger.info("Analysis complete. Printing results...")
    print(json.dumps(analyzed_document, indent=4))
