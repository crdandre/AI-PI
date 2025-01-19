import dspy
from ai_pi.core.lm_pipeline import ProcessingStep, BaseProcessor, ProcessingPipeline, PipelineConfig
from ai_pi.lm_config import get_lm_for_task


class SectionProcessor(BaseProcessor):
    """Processes individual document sections"""
    
    class Signature(dspy.Signature):
        """Signature for section summarization"""
        section_type = dspy.InputField(desc="Type of section being summarized")
        text = dspy.InputField(desc="Section text to summarize")
        summary = dspy.OutputField(desc="Focused summary containing main points, evidence, findings, and significance")
    
    def process(self, data: dict) -> dict:
        sections = data.get('sections', [])
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


class RelationshipProcessor(BaseProcessor):
    """Analyzes relationships between sections"""
    
    class Signature(dspy.Signature):
        """Signature for analyzing relationships between sections"""
        summaries = dspy.InputField(desc="Section summaries to analyze")
        analysis = dspy.OutputField(desc="Analysis of logical flow, argument development, and key dependencies")
    
    def process(self, data: dict) -> dict:
        section_summaries = data.get('section_summaries', [])
        formatted_summaries = "\n\n".join(section_summaries)
        
        with dspy.context(lm=self.lm):
            result = self.predictors['Signature'](
                summaries=formatted_summaries
            )
        return {'relationship_analysis': result.analysis}


class DocumentProcessor(BaseProcessor):
    """Creates document-level summary"""
    
    class Signature(dspy.Signature):
        """Signature for document-level summary"""
        section_summaries = dspy.InputField(desc="All section summaries")
        relationships = dspy.InputField(desc="Relationship analysis between sections")
        analysis = dspy.OutputField(desc="Comprehensive review-oriented summary")
    
    def process(self, data: dict) -> dict:
        section_summaries = data.get('section_summaries', [])
        relationship_analysis = data.get('relationship_analysis', '')
        
        formatted_summaries = "\n".join(section_summaries)
        
        with dspy.context(lm=self.lm):
            result = self.predictors['Signature'](
                section_summaries=formatted_summaries,
                relationships=relationship_analysis
            )
        return {'document_analysis': result.analysis}


class TopicProcessor(BaseProcessor):
    """Extracts document topic"""
    
    class Signature(dspy.Signature):
        """Signature for extracting concise document topics"""
        analysis = dspy.InputField(desc="Document analysis text to extract topic from")
        topic = dspy.OutputField(desc="Concise topic (10 words or less) capturing main document focus")
    
    def process(self, data: dict) -> dict:
        document_analysis = data.get('document_analysis', '')
        with dspy.context(lm=self.lm):
            result = self.predictors['Signature'](
                analysis=document_analysis
            )
        return {'topic': result.topic.strip()}


class Summarizer:
    """
    Creates a hierarchical summary of an input document. The input is
    a structured json file with full section text separated by key.
    """
    def __init__(self, verbose: bool = False):
        steps = [
            ProcessingStep(
                step_type="section",
                lm_name="summarization",
                processor_class=SectionProcessor,
                output_key="section_summaries",
                depends_on=[]
            ),
            ProcessingStep(
                step_type="relationship",
                lm_name="summarization",
                processor_class=RelationshipProcessor,
                output_key="relationship_analysis",
                depends_on=["section_summaries"]
            ),
            ProcessingStep(
                step_type="document",
                lm_name="summarization",
                processor_class=DocumentProcessor,
                output_key="document_analysis",
                depends_on=["section_summaries", "relationship_analysis"]
            ),
            ProcessingStep(
                step_type="topic",
                lm_name="summarization",
                processor_class=TopicProcessor,
                output_key="topic",
                depends_on=["document_analysis"]
            )
        ]
        
        self.pipeline = ProcessingPipeline(
            PipelineConfig(steps=steps, verbose=verbose)
        )

    def analyze_sectioned_document(self, document_json: dict) -> dict:
        """
        Create a true summary tree where each level summarizes its children,
        working with pre-sectioned content from document_json.
        """
        results = self.pipeline.execute({'sections': document_json['sections']})
        
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
