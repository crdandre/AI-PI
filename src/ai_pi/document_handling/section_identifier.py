import dspy
import logging
import json
import re
from typing import List, Dict, Union
from json.decoder import JSONDecodeError

from big_dict_energy.lm_setup import LMConfig, LMForTask, TaskConfig
from big_dict_energy.utils.text_utils import normalize_unicode
from big_dict_energy.utils.logging import setup_logging

class HeadingInfo(dspy.Signature):
    """Structured output for a single heading"""
    level = dspy.OutputField(desc="Heading level (number of # characters)")
    text = dspy.OutputField(desc="The heading text")
    section_type = dspy.OutputField(desc="Classified section type (e.g., 'Abstract', 'Introduction', etc.)")

class DocumentStructure(dspy.Signature):
    """Input/Output signature for document structure identification"""
    text = dspy.InputField(desc="The academic paper text to analyze")
    headings = dspy.OutputField(desc="List of headings in the document with their levels and classifications", type=List[HeadingInfo])

class SectionMatchStrings(dspy.Signature):
    """Structured output for a pair of strings to use as section boundaries"""
    start = dspy.OutputField(desc="The first few words of a section, marking the start for text matching")
    end = dspy.OutputField(desc="The last few words of a section, marking the end boundary for text matching")

class SectionInfo(dspy.Signature):
    """Input/Output signature for section identification"""
    text = dspy.InputField(desc="The section text to analyze")
    start_text = dspy.OutputField(desc="The first 10-15 words of the section")
    end_text = dspy.OutputField(desc="The last 10-15 words of the section")

class DocumentSections(dspy.Signature):
    """Input/Output signature for document section identification"""
    text = dspy.InputField(desc="The academic paper text to analyze")
    sections = dspy.OutputField(desc="List of sections in the document", type=List[SectionInfo])

class ExtractSectionBoundaries(dspy.Signature):
    """Signature for extracting exact boundary text from a section"""
    section_text = dspy.InputField(desc="The full text of the academic paper section")
    start_text = dspy.OutputField(desc="The exact first 10-15 words from the section's beginning")
    end_text = dspy.OutputField(desc="The exact last 10-15 words from the section's end")

def _clean_and_parse_json(json_str: str) -> List[Dict]:
    """Helper function to clean and parse potentially malformed JSON from LLM output."""
    try:
        # First try direct parsing
        return json.loads(json_str)
    except JSONDecodeError:
        try:
            # If the input is a Prediction object, try to get the JSON from the headings attribute
            if "Prediction" in json_str and "headings" in json_str:
                # Extract the JSON string from the headings attribute
                match = re.search(r'headings=\'(.*?)\'(?=\)|\s*,)', json_str, re.DOTALL)
                if match:
                    json_str = match.group(1)
                    
            # Remove any markdown code block syntax and escape characters
            cleaned = json_str.replace('```json', '').replace('```', '').strip()
            cleaned = cleaned.replace('\\n', '\n')  # Handle escaped newlines
            
            # Try to parse again
            return json.loads(cleaned)
        except JSONDecodeError as e:
            logger.error(f"Failed to parse JSON after cleaning: {str(e)}")
            logger.debug(f"Problematic JSON string: {cleaned}")
            return []

class SectionTypes:
    """Dynamic section type management"""
    
    # Base sections that can be extended
    DEFAULT_SECTIONS = {
        'Abstract': ['abstract', 'summary'],
        'Introduction': ['introduction', 'background'],
        'Methods': ['methods', 'methodology', 'materials and methods', 'experimental'],
        'Results': ['results', 'findings', 'observations'],
        'Discussion': ['discussion', 'interpretation'],
        'Conclusions': ['conclusions', 'concluding remarks'],
        'References': ['references', 'bibliography', 'works cited'],
        'Other': ['other']
    }
    
    def __init__(self, custom_sections: Dict[str, List[str]] = None):
        self.sections = self.DEFAULT_SECTIONS.copy()
        if custom_sections:
            self.sections.update(custom_sections)
    
    def normalize_section_type(self, heading: str) -> str:
        """Match heading to canonical section type"""
        heading_lower = heading.lower().strip()
        for canonical, variants in self.sections.items():
            if heading_lower in [v.lower() for v in variants]:
                return canonical
        return 'Other'
    
    def get_main_sections(self) -> List[str]:
        """Get list of main section types (excluding 'Other')"""
        return [s for s in self.sections.keys() if s != 'Other']

class SingleContextSectionIdentifier:
    """Identifies academic paper sections using LLM in two passes:
    1. Identify document structure (headings and their levels)
    2. Extract main sections with their boundary strings
    """
    
    def __init__(self, lm: Union[LMConfig, dspy.LM, TaskConfig] = None, 
                 custom_sections: Dict[str, List[str]] = None):
        # Use section_review task configuration by default
        self.lm = LMForTask.SECTION_IDENTIFICATION.get_lm() if lm is None else (
            lm.create_lm() if isinstance(lm, (LMConfig, TaskConfig)) else lm
        )
        self.section_types = SectionTypes(custom_sections)
        # Use predictor type from task config
        predictor_type = LMForTask.SECTION_IDENTIFICATION.get_predictor_type()
        self.structure_predictor = getattr(dspy, predictor_type.value)(DocumentStructure)
        self.boundary_predictor = getattr(dspy, predictor_type.value)(ExtractSectionBoundaries)
        self.logger = logging.getLogger('section_identifier')

    def _identify_document_structure(self, text: str) -> List[Dict]:
        """First pass: Identify all headings and their levels with line numbers."""
        try:
            lines = text.splitlines()
            headings = []
            
            # First get all headings with their line numbers
            for i, line in enumerate(lines):
                if line.strip().startswith('#'):
                    level = len(line) - len(line.lstrip('#'))
                    heading_text = line.lstrip('#').strip()
                    headings.append({
                        'level': level,
                        'text': heading_text,
                        'line_number': i
                    })
            
            # Use LLM to classify the headings
            with dspy.context(lm=self.lm):
                prompt = f"""You are analyzing headings from an academic paper.
                    
                    For each heading, output a JSON object with:
                    - level: number of # characters
                    - text: the heading text
                    - section_type: classify as one of {self.section_types.get_main_sections() + ['Other']}
                    
                    Focus on identifying main sections. Subsections should be classified as "Other".
                    
                    Headings to analyze:
                    {json.dumps([{
                        'level': h['level'], 
                        'text': h['text']
                    } for h in headings], indent=2)}"""
                
                result = self.structure_predictor(text=prompt)
                
                try:
                    # If result is a Prediction object, get the headings attribute
                    if hasattr(result, 'headings'):
                        result_str = result.headings
                    else:
                        result_str = str(result)
                    
                    # Parse the JSON
                    parsed_result = _clean_and_parse_json(result_str)
                    
                    if isinstance(parsed_result, list):
                        classified_headings = parsed_result
                    elif isinstance(parsed_result, dict) and 'headings' in parsed_result:
                        classified_headings = parsed_result['headings']
                    else:
                        logger.error(f"Unexpected result format: {parsed_result}")
                        return headings
                    
                    # Merge classifications with our heading info
                    for i, heading in enumerate(headings):
                        if i < len(classified_headings):
                            heading['section_type'] = classified_headings[i].get('section_type', 'Other')
                        else:
                            heading['section_type'] = 'Other'
                    
                    return headings
                    
                except Exception as e:
                    logger.error(f"Failed to parse LLM response: {e}")
                    logger.debug(f"Raw LLM response: {result}")
                    return headings
            
        except Exception as e:
            self.logger.error(f"Error identifying document structure: {str(e)}")
            self.logger.exception("Full traceback:")
            return []

    def process_document(self, text: str) -> List[Dict]:
        """Process document using heading positions to determine section boundaries."""
        try:
            # Normalize Unicode in input text
            text = normalize_unicode(text)
            lines = text.splitlines()
            
            # First pass: get document structure with line numbers
            headings = self._identify_document_structure(text)
            
            # Filter for main section headings and sort by line number
            main_sections = self.section_types.get_main_sections()
            main_headings = [h for h in headings 
                           if h['section_type'] in main_sections]
            main_headings.sort(key=lambda x: x['line_number'])
            
            # Process each main section
            processed_sections = []
            for i, heading in enumerate(main_headings):
                # Get section boundaries
                start_line = heading['line_number'] + 1  # Start after heading
                if i < len(main_headings) - 1:
                    end_line = main_headings[i + 1]['line_number']
                else:
                    end_line = len(lines)  # Last section goes to end of file
                
                # Get section text and normalize Unicode
                section_text = normalize_unicode('\n'.join(lines[start_line:end_line]).strip())
                
                with dspy.context(lm=self.lm):
                    # Replace the text prompt with the signature-based approach
                    result = self.boundary_predictor(section_text=section_text)
                    
                    try:
                        section_info = {
                            'section_type': heading['section_type'],
                            'match_strings': {
                                'start': normalize_unicode(str(result.start_text).strip()),
                                'end': normalize_unicode(str(result.end_text).strip())
                            },
                            'text': normalize_unicode(section_text)
                        }
                        processed_sections.append(section_info)
                        
                    except Exception as e:
                        logger.warning(f"Failed to process section {heading['section_type']}: {str(e)}")
                        logger.debug("Exception details:", exc_info=True)
                        continue
                    
            return processed_sections
                
        except Exception as e:
            self.logger.error(f"Error processing document: {str(e)}")
            self.logger.exception("Full traceback:")
            return []

if __name__ == "__main__":
    import datetime
    from dotenv import load_dotenv
    load_dotenv()
    from pathlib import Path
    
    # Setup logging using the centralized configuration
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    logger = setup_logging(log_dir, timestamp, "section_identifier")
    
    # Use specific file path
    paper_path = Path("/home/christian/projects/agents/ai_pi/processed_documents/ScolioticFEPaper_v7_20250107_010732/ScolioticFEPaper_v7/ScolioticFEPaper_v7.md")
    with open(paper_path, 'r', encoding='utf-8') as f:
        paper_text = f.read()

    identifier = SingleContextSectionIdentifier()
    sections = identifier.process_document(text=paper_text)
    
    logger.info("Document processing complete. Printing results...")
    print(json.dumps(sections, indent=4, ensure_ascii=False))

