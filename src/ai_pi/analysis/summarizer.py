from dotenv import load_dotenv
load_dotenv()
from typing import List, Dict
import dspy
from ..lm_config import get_lm_for_task

class SectionSummary(dspy.Signature):
    """Signature for section summarization"""
    section_type = dspy.InputField(desc="Type of section being summarized")
    text = dspy.InputField(desc="Section text to summarize")
    summary = dspy.OutputField(desc="Focused summary containing main points, evidence, findings, and significance")

class RelationshipAnalysis(dspy.Signature):
    """Signature for analyzing relationships between sections"""
    summaries = dspy.InputField(desc="Section summaries to analyze")
    analysis = dspy.OutputField(desc="Analysis of logical flow, argument development, and key dependencies")

class DocumentAnalysis(dspy.Signature):
    """Signature for document-level summary"""
    section_summaries = dspy.InputField(desc="All section summaries")
    relationships = dspy.InputField(desc="Relationship analysis between sections")
    analysis = dspy.OutputField(desc="Comprehensive review-oriented summary")

class TopicExtraction(dspy.Signature):
    """Signature for extracting concise document topics"""
    analysis = dspy.InputField(desc="Document analysis text to extract topic from")
    topic = dspy.OutputField(desc="Concise topic (10 words or less) capturing main document focus")

class Summarizer:
    """
    Analyzes pre-sectioned academic papers using a hierarchical approach that mirrors
    how PIs review papers. Creates true summary trees where parent nodes
    summarize their children.
    """
    def __init__(self, lm: dspy.LM = None, verbose=False):
        """
        Initialize summarizer with optional custom LM
        Args:
            lm: Optional dspy.LM instance (falls back to default if None)
            verbose: Enable detailed logging
        """
        # If LM is provided directly, use it; otherwise get default
        self.lm = lm if isinstance(lm, dspy.LM) else get_lm_for_task("summarization")
        self.verbose = verbose
        
        # Initialize predictors with the configured LM
        with dspy.context(lm=self.lm):
            self.section_predictor = dspy.ChainOfThought(SectionSummary)
            self.relationship_predictor = dspy.ChainOfThought(RelationshipAnalysis)
            self.document_predictor = dspy.ChainOfThought(DocumentAnalysis)
            self.topic_predictor = dspy.ChainOfThought(TopicExtraction)
    
    def analyze_sectioned_document(self, document_json: dict) -> dict:
        """
        Create a true summary tree where each level summarizes its children,
        working with pre-sectioned content from document_json.
        """
        if self.verbose:
            print("Starting sectioned document analysis...")
        
        # 1. Extract sections and create summaries
        section_summaries = self._summarize_sections(document_json['sections'])
        
        # 2. Analyze relationships between sections
        relationship_summary = self._summarize_relationships(section_summaries)
        
        # 3. Create overall document summary
        document_summary = self._create_document_summary(
            section_summaries, 
            relationship_summary
        )
        
        # 4. Extract topic from Abstract
        document_topic = self._get_document_topic(document_summary['document_analysis'])
        
        # 5. Create hierarchical summary structure
        hierarchical_summary = {
            'topic': document_topic,
            'document_summary': document_summary,
            'relationship_summary': relationship_summary,
            'section_summaries': section_summaries
        }
        
        # 6. Add the hierarchical summary to the input document
        document_json['hierarchical_summary'] = hierarchical_summary
        
        if self.verbose:
            print(f"Generated summary tree with {len(section_summaries)} sections")
            
        return document_topic, document_json
    
    def _summarize_sections(self, sections: List[Dict]) -> List[Dict]:
        """Create detailed summaries of pre-defined sections"""
        summaries = []
        
        for i, section in enumerate(sections):
            if self.verbose:
                print(f"Summarizing section {i+1}/{len(sections)}: {section['section_type']}")
            
            with dspy.context(lm=self.lm):
                result = self.section_predictor(
                    section_type=section['section_type'],
                    text=section['text']
                )
                
                summaries.append({
                    'section_type': section['section_type'],
                    'summary': result.summary,
                    'match_strings': section['match_strings']
                })
            
        return summaries
    
    def _summarize_relationships(self, section_summaries: List[Dict]) -> Dict:
        """Analyze relationships between sections with IMRaD structure in mind"""
        if self.verbose:
            print(f"Analyzing relationships between {len(section_summaries)} sections...")
        
        # Format summaries for input
        summary_texts = [
            f"{s['section_type']}:\n{s['summary']}" 
            for s in section_summaries
        ]
        formatted_summaries = "\n\n".join(summary_texts)
        
        with dspy.context(lm=self.lm):
            result = self.relationship_predictor(
                summaries=formatted_summaries
            )
            
            return {
                'relationship_analysis': result.analysis
            }
    
    def _create_document_summary(
        self, 
        section_summaries: List[Dict],
        relationship_summary: Dict
    ) -> Dict:
        """Create high-level document summary with review guidance"""
        if self.verbose:
            print("Generating final document-level summary...")
        
        formatted_summaries = "\n".join(
            f"{s['section_type']}: {s['summary']}" 
            for s in section_summaries
        )
        
        with dspy.context(lm=self.lm):
            result = self.document_predictor(
                section_summaries=formatted_summaries,
                relationships=relationship_summary['relationship_analysis']
            )
            
            return {
                'document_analysis': result.analysis
            }

    def _get_document_topic(self, document_analysis: str) -> str:
        """
        Extract a concise topic description from the document analysis.
        
        Args:
            document_analysis: String containing the comprehensive document analysis
            
        Returns:
            str: A concise topic/title describing the document's main focus
        """
        with dspy.context(lm=self.lm):
            result = self.topic_predictor(analysis=document_analysis)
            return result.topic.strip()

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    import json
    
    # Configure OpenRouter LLM
    lm = dspy.LM(
        "openrouter/openai/gpt-4o",
        api_base="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.01,
    )
    
    # Load sample document
    with open("ref/sample_output.json", "r", encoding="utf-8") as f:
        document_json = json.load(f)
    
    # Initialize context agent
    context_agent = Summarizer(lm=lm, verbose=True)
    
    # Process document
    topic, analyzed_document = context_agent.analyze_sectioned_document(document_json)
    
    # Print results
    print(json.dumps(analyzed_document, indent=4))
