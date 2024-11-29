from dotenv import load_dotenv
load_dotenv()
from llama_index.llms.nvidia import NVIDIA
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from typing import List, Dict

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
        self.node_parser = SentenceSplitter.from_defaults(
            chunk_size=512,
            chunk_overlap=128
        )
    
    def analyze_document(self, document_text: str) -> dict:
        """
        Create a true summary tree where each level summarizes its children.
        """
        if self.verbose:
            print(f"Starting document analysis. Text length: {len(document_text)} characters")
            
        # 1. First split into base sections
        if self.verbose:
            print("Creating Document object...")
        doc = Document(text=document_text)
        
        if self.verbose:
            print("Starting node parsing...")
        base_nodes = self.node_parser.get_nodes_from_documents([doc])
        
        if self.verbose:
            print(f"Node parsing complete. Generated {len(base_nodes)} nodes")
        
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
        
        for i, node in enumerate(nodes):
            if self.verbose:
                print(f"Summarizing base section {i+1}/{len(nodes)}...")
            
            prompt = """
            Provide a brief, focused summary of this academic paper section in 100 words or less:
            1. Main Points (2-3 key ideas)
            2. Evidence/Methods (if applicable)
            3. Key Findings
            
            Section Text: {text}
            
            Be concise and focus on essential information only.
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
        if self.verbose:
            print(f"Analyzing relationships between {len(section_summaries)} sections...")
            
        prompt = """
        In 150 words or less, analyze the key relationships between these sections:
        1. Logical Flow
        2. Main Argument Development
        3. Key Dependencies
        
        Section Summaries:
        {summaries}
        
        Focus on essential connections only.
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
        """Create high-level document summary structured for review guidance"""
        if self.verbose:
            print("Generating final document-level summary...")
            
        prompt = """
        Create a comprehensive summary to guide the paper's review.
        Focus on:
        1. Core Contribution
           - Main claims/findings
           - Significance in field
           - Novel elements
        
        2. Research Approach
           - Key methodological choices
           - Data/evidence types
           - Analysis strategies
        
        3. Paper Structure
           - How arguments are built
           - Evidence flow
           - Key dependencies between sections
        
        4. Expected Standards
           - Critical elements to verify
           - Potential weak points to examine
           - Required supporting evidence
        
        Using these section summaries:
        {section_summaries}
        
        And their relationships:
        {relationships}
        
        Provide a structured review-oriented summary.
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
