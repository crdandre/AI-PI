from typing import List, Dict
import dspy
import logging

logger = logging.getLogger(__name__)

class SectionInfo(dspy.Signature):
    """Structured output for section identification"""
    section_type = dspy.OutputField(desc="Section type (e.g., 'Abstract', 'Introduction', 'Methods', etc.)")
    text = dspy.OutputField(desc="The text content of the section")

class SimpleSectionIdentifier:
    """Identifies academic paper sections using LLM."""
    
    def __init__(self, engine: dspy.LM):
        self.engine = engine
        self.predictor = dspy.Predict(SectionInfo)
    
    def process_document(self, text: str) -> List[Dict]:
        """Process document and identify sections using a single LLM call."""
        prompt = f"""Analyze this academic paper and break it into its main sections.
        
        Guidelines:
        - Return EACH section as a separate item
        - Identify each major section based on content and headings (like Abstract, Introduction, Methods, Results, etc.)
        - Include the full text of each section
        - Maintain the original order of sections
        - Do not summarize - return the actual text from each section
        
        Paper text:
        TESTING

        Return each section with its type and complete content."""
        
        try:
            with dspy.context(lm=self.engine):
                sections = self.predictor(prompt=prompt)
                # Convert to list if single section is returned
                if isinstance(sections.section_type, str):
                    return [{
                        'type': sections.section_type,
                        'text': sections.text
                    }]
                # Handle multiple sections
                return [
                    {
                        'type': section_type,
                        'text': section_text
                    }
                    for section_type, section_text in zip(sections.section_type, sections.text)
                ]
        except Exception as e:
            logger.error(f"Error identifying sections: {str(e)}")
            return []

if __name__ == "__main__":
    # Example usage
    import os
    from dotenv import load_dotenv
    load_dotenv()
    from pathlib import Path
    
    # Set up basic logging
    logging.basicConfig(level=logging.INFO)
    
    # Use specific file path
    paper_path = Path("/home/christian/projects/agents/ai_pi/processed_documents/test_full_text.txt")  
    with open(paper_path, 'r', encoding='utf-8') as f:
        paper_text = f.read()
    
    # Initialize and run
    lm = dspy.LM(
        'openrouter/google/gemini-2.0-flash-exp:free',
        api_base="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.01,
    )
    identifier = SimpleSectionIdentifier(engine=lm)
    sections = identifier.process_document(text=paper_text)
    
    # Print results
    print(sections)

