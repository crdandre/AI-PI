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
        summary_texts = [
            f"{s['section_type']}:\n{s['summary']}" 
            for s in section_summaries
        ]
        formatted_summaries = "\n\n".join(summary_texts)
        
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
        
        formatted_summaries = "\n".join(
            f"{s['section_type']}: {s['summary']}" 
            for s in section_summaries
        )
        
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
    Analyzes pre-sectioned academic papers using a hierarchical approach that mirrors
    how PIs review papers. Creates true summary trees where parent nodes
    summarize their children.
    """
    def __init__(self, lm: dspy.LM = None, verbose: bool = False):
        # Configure pipeline steps
        steps = [
            ProcessingStep(
                step_type="section",
                name="section_summary",
                lm_name="summarization",
                signatures=[SectionProcessor.Signature],
                output_key="section_summaries",
                depends_on=[]
            ),
            ProcessingStep(
                step_type="relationship",
                name="relationship_analysis",
                lm_name="summarization",
                signatures=[RelationshipProcessor.Signature],
                output_key="relationship_analysis",
                depends_on=["section_summaries"]
            ),
            ProcessingStep(
                step_type="document",
                name="document_summary",
                lm_name="summarization",
                signatures=[DocumentProcessor.Signature],
                output_key="document_analysis",
                depends_on=["section_summaries", "relationship_analysis"]
            ),
            ProcessingStep(
                step_type="topic",
                name="topic_extraction",
                lm_name="summarization",
                signatures=[TopicProcessor.Signature],
                output_key="topic",
                depends_on=["document_analysis"]
            )
        ]
        
        # Initialize pipeline
        self.pipeline = ProcessingPipeline(
            PipelineConfig(steps=steps, verbose=verbose)
        )
        
        # Register processors
        self.pipeline.register_processor("section", SectionProcessor)
        self.pipeline.register_processor("relationship", RelationshipProcessor)
        self.pipeline.register_processor("document", DocumentProcessor)
        self.pipeline.register_processor("topic", TopicProcessor)
        
        # Override LM if provided
        self.lm = lm if isinstance(lm, dspy.LM) else get_lm_for_task("summarization")
        if self.lm:
            for step in steps:
                step.lm = self.lm

    def analyze_sectioned_document(self, document_json: dict) -> dict:
        """
        Create a true summary tree where each level summarizes its children,
        working with pre-sectioned content from document_json.
        """
        results = self.pipeline.execute({'sections': document_json['sections']})
        
        # Create hierarchical summary structure
        hierarchical_summary = {
            'topic': results['topic'],
            'document_summary': {'document_analysis': results['document_analysis']},
            'relationship_summary': {'relationship_analysis': results['relationship_analysis']},
            'section_summaries': results['section_summaries']
        }
        
        # Add summary to document
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
    
    with open("ref/sample_output.json", "r", encoding="utf-8") as f:
        document_json = json.load(f)
    
    context_agent = Summarizer(verbose=True)
    
    topic, analyzed_document = context_agent.analyze_sectioned_document(document_json)
    
    logger.info("Analysis complete. Printing results...")
    print(json.dumps(analyzed_document, indent=4))
