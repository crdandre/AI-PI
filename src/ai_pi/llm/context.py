from dotenv import load_dotenv
load_dotenv()
from llama_index.llms.nvidia import NVIDIA
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import AgentRunner, ReActAgent, ReActAgentWorker
from llama_index.core import Document
from llama_index.core.node_parser import SimpleNodeParser
from typing import List, Dict

from ai_pi.documents.document_embedding import ContextStorageInterface

import os
import dspy

class ContextAgent:
    """
    Analyzes academic papers using a hierarchical approach that mirrors
    how PIs review papers. Creates true summary trees where parent nodes
    summarize their children.
    """
    def __init__(
        self,
        llm=NVIDIA(model="meta/llama3-70b-instruct"),
        verbose=False
    ):
        self.llm = llm
        self.verbose = verbose
        
        # Use simple parser for initial sectioning
        self.node_parser = SimpleNodeParser.from_defaults(
            chunk_size=512,
            chunk_overlap=50
        )
    
    def analyze_document(self, document_text: str) -> dict:
        """
        Create a true summary tree where each level summarizes its children.
        """
        # 1. First split into base sections
        doc = Document(text=document_text)
        base_nodes = self.node_parser.get_nodes_from_documents([doc])
        
        # 2. Build summary tree bottom-up
        section_summaries = self._summarize_base_sections(base_nodes)
        relationship_summary = self._summarize_relationships(section_summaries)
        document_summary = self._create_document_summary(
            section_summaries, 
            relationship_summary
        )
        
        tree = {
            'document_summary': document_summary,
            'relationship_summary': relationship_summary,
            'section_summaries': section_summaries
        }
        
        if self.verbose:
            print(f"Generated summary tree with {len(section_summaries)} sections")
            
        return tree
    
    def _summarize_base_sections(self, nodes) -> List[Dict]:
        """Create detailed summaries of base sections"""
        summaries = []
        
        for node in nodes:
            prompt = """
            Provide a detailed summary of this academic paper section:
            1. Main Arguments/Points
            2. Evidence/Methods Presented
            3. Key Findings/Claims
            4. Writing Style and Clarity
            5. Potential Review Points
            
            Section Text: {text}
            
            Format as a structured summary that captures the essence of this section.
            """
            
            summary = self.llm.complete(prompt.format(text=node.text))
            summaries.append({
                'section_title': self._infer_section_title(node.text),
                'summary': summary,
                'original_text': node.text
            })
            
        return summaries
    
    def _summarize_relationships(self, section_summaries: List[Dict]) -> Dict:
        """Create a summary of relationships between sections"""
        prompt = """
        Analyze the relationships between these section summaries:
        1. Logical Flow and Transitions
        2. Argument Development
        3. Evidence Chain
        4. Narrative Coherence
        5. Cross-Section Dependencies
        
        Section Summaries:
        {summaries}
        
        Create a structured summary of how these sections work together.
        """
        
        # Extract just the summaries for the prompt
        summary_texts = [
            f"{s['section_title']}:\n{s['summary']}" 
            for s in section_summaries
        ]
        
        return {
            'relationship_analysis': self.llm.complete(
                prompt.format(summaries="\n\n".join(summary_texts))
            )
        }
    
    def _create_document_summary(
        self, 
        section_summaries: List[Dict],
        relationship_summary: Dict
    ) -> Dict:
        """Create high-level document summary"""
        prompt = """
        Create a comprehensive document-level summary:
        1. Overall Contribution and Significance
        2. Paper Structure and Organization
        3. Key Arguments and Evidence Chain
        4. Writing Style and Clarity
        5. Critical Review Points
        
        Using these section summaries:
        {section_summaries}
        
        And their relationships:
        {relationships}
        
        Provide a structured summary that a PI would use to guide their review.
        """
        
        return {
            'document_analysis': self.llm.complete(
                prompt.format(
                    section_summaries="\n".join(
                        f"{s['section_title']}: {s['summary']}" 
                        for s in section_summaries
                    ),
                    relationships=relationship_summary['relationship_analysis']
                )
            )
        }
    
    def _infer_section_title(self, text: str) -> str:
        """Infer section title from content"""
        common_sections = [
            "Abstract", "Introduction", "Methods",
            "Results", "Discussion", "Conclusion"
        ]
        
        first_line = text.split('\n')[0].strip()
        
        for section in common_sections:
            if section.lower() in first_line.lower():
                return section
                
        return first_line[:50] + "..."
