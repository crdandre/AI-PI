from typing import List, Dict
import dspy
import logging
import json

logger = logging.getLogger(__name__)

class SectionMatchStrings(dspy.Signature):
    """Structured output for a pair of strings to
    use as section boundaries"""
    start = dspy.OutputField(desc="The first three words of a section, marking the start for text matching. This excludes the section name itself, i.e. IntroductionThe study was... --> The study was")
    end = dspy.OutputField(desc="The last three words of a section, marking the end boundary for text matching")

class SectionInfo(dspy.Signature):
    """Structured output for a single section"""
    section_type = dspy.OutputField(desc="Section type (e.g., 'Abstract', 'Introduction', 'Methods', etc.)")
    match_strings = dspy.OutputField(desc="The boundary strings for this section", type=SectionMatchStrings)

class DocumentSections(dspy.Signature):
    """Input/Output signature for document section identification"""
    text = dspy.InputField(desc="The academic paper text to analyze")
    sections = dspy.OutputField(desc="List of sections in the document. Parseable JSON.", type=List[SectionInfo])

class SingleContextSectionIdentifier:
    """Identifies academic paper sections using LLM."""
    
    def __init__(self, engine: dspy.LM):
        self.engine = engine
        self.predictor = dspy.ChainOfThought(DocumentSections)
    
    def process_document(self, text: str) -> List[Dict]:
        """Process document and identify sections using a single LLM call."""
        try:
            with dspy.context(lm=self.engine):
                prompt = f"""Analyze this academic paper and identify its main sections, even if the formatting is inconsistent.
                    
                    Requirements:
                    - Each section should have a section_type (like Abstract, Introduction, Methods, Results). Abstract subsections don't count, nor do any other section names such as "Limitations" etc. Only stick to main well-known section names - think about what you know about likely academic paper section names and assess+include accordingly. References DO count, and should be included.
                    - Each section should contain match strings (start, end), containing the first and last five-ish words in each section for text matching
                    - Ensure the output is parseable as json (i.e. clean extra newline characters, spaces, etc.!!!!!!!
                    - Maintain the original order of sections
                    - Do not deviate from the wording given in the input text
                    - Identify sections even when they lack clear spacing or formatting
                    - For sections without clear boundaries, use context clues like topic changes
                    - If a section header is merged with content (e.g., "IntroductionThe study..."), separate them
                    - Maintain chronological order of sections as they appear
                    
                    Paper text:
                    {text}"""
                
                result = self.predictor(text=prompt)
                
                # Handle both string and list responses
                if isinstance(result.sections, str):
                    # Remove markdown code block syntax if present
                    cleaned_json = result.sections.replace('```json\n', '').replace('\n```', '').strip()
                    sections = json.loads(cleaned_json)
                else:
                    sections = result.sections

                # Ensure we're working with a list
                if not isinstance(sections, list):
                    logger.error(f"Unexpected sections format: {type(sections)}")
                    return []

                # Process each section
                processed_sections = []
                for section in sections:
                    if isinstance(section, dict) and 'section_type' in section and 'match_strings' in section:
                        processed_sections.append({
                            'type': section['section_type'],
                            'match_strings': {
                                'start': section['match_strings']['start'],
                                'end': section['match_strings']['end']
                            }
                        })
                    else:
                        logger.warning(f"Skipping malformed section: {section}")

                return processed_sections
                
        except Exception as e:
            logger.error(f"Error identifying sections: {str(e)}")
            logger.exception("Full traceback:")
            return []

if __name__ == "__main__":
    # Example usage
    import os
    from dotenv import load_dotenv
    load_dotenv()
    from pathlib import Path
    
    import logging
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Use specific file path
    paper_path = Path("/home/christian/projects/agents/ai_pi/examples/testwocomments/testwocomments_corrected.md")  
    with open(paper_path, 'r', encoding='utf-8') as f:
        paper_text = f.read()
        
    # openrouter_model = 'openrouter/google/gemini-2.0-flash-exp:free'
    # openrouter_model = 'openrouter/google/learnlm-1.5-pro-experimental:free'
    # openrouter_model = 'openrouter/anthropic/claude-3.5-haiku'
    # openrouter_model = 'openrouter/deepseek/deepseek-chat'
    openrouter_model = 'openrouter/openai/gpt-4o'
    
    # Initialize and run
    lm = dspy.LM(
        openrouter_model,
        api_base="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.01,
    )
    identifier = SingleContextSectionIdentifier(engine=lm)
    sections = identifier.process_document(text=paper_text)
    
    print(sections)
    
    # Print results
    import json
    print(json.dumps(sections, indent=4))

