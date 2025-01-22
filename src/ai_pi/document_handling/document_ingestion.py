"""
Extracts:
- Document Info, Sections
- Existing comments
- Images(+Image Paths), Tables

Output:
"""

from lxml import etree
import zipfile
from typing import Dict, List, Optional, TypedDict, Union
from pathlib import Path
from datetime import datetime
import pypandoc
import logging
import json

from ai_pi.document_handling.marker_extract_from_pdf import PDFTextExtractor
from ai_pi.document_handling.section_identifier import SingleContextSectionIdentifier
from dspy_workflow_builder.utils.text_utils import normalize_unicode
from dspy_workflow_builder.utils.logging import setup_logging

logger = logging.getLogger(__name__)

class Revision(TypedDict):
    id: str
    type: str  # 'insertion', 'deletion', 'formatting'
    text: str
    author: Optional[str]
    date: Optional[str]
    position: Dict[str, int] 
    referenced_text: str
    parent_id: Optional[str]
    formatting: Optional[Dict]


class Comment(TypedDict):
    id: str
    text: str
    author: str
    date: str
    original_text: str  # The exact text that was commented
    match_string: str   # Expanded context for matching
    resolved: bool
    replies: List['Comment']
    related_revision_id: Optional[str]


class TableData(TypedDict):
    id: str
    content: List[List[str]]
    position: Dict[str, int]
    caption: Optional[str]


def get_expanded_range(start: int, end: int, full_text: str, context_window: int = None) -> tuple[int, int]:
    """
    Helper function to expand range to sentence boundaries.
    If the text is a heading (ends with no period), expands to section breaks.
    """
    text_len = len(full_text)
    
    # Sentence ending punctuation
    sentence_ends = {'.', '!', '?', '\n\n'}
    
    # Expand start to previous sentence boundary
    expanded_start = start
    while expanded_start > 0:
        if expanded_start < text_len and full_text[expanded_start - 1] in sentence_ends:
            break
        expanded_start -= 1
    
    # Skip leading whitespace
    while expanded_start < end and full_text[expanded_start].isspace():
        expanded_start += 1
    
    # Expand end to next sentence boundary
    expanded_end = end
    while expanded_end < text_len:
        if full_text[expanded_end] in sentence_ends:
            expanded_end += 1
            break
        expanded_end += 1
    
    # If no sentence boundary found (might be a heading), look for section breaks
    if expanded_end == text_len or (expanded_start == 0 and expanded_end == end):
        expanded_start = max(0, start - 1)
        expanded_end = min(text_len, end + 1)
        while expanded_start > 0 and full_text[expanded_start-1] != '\n\n':
            expanded_start -= 1
        while expanded_end < text_len and full_text[expanded_end] != '\n\n':
            expanded_end += 1
    
    referenced_text = full_text[expanded_start:expanded_end]
    
    # Adjust expanded_end based on cleaned text length
    expanded_end = expanded_start + len(referenced_text)
    
    return expanded_start, expanded_end


def prepare_comment_text(comment: Dict) -> str:
    """Format comment data into a single text string."""
    text = (
        f"[COMMENT ID: {comment['id']}]\n"
        f"Position: chars {comment['position']['start']}-{comment['position']['end']}\n"
        f"Context: \"{comment['referenced_text']}\"\n"
        f"Comment: {comment['text']}\n"
        f"Author: {comment['author']}\n"
        f"Date: {comment['date']}\n"
        f"Status: {'Resolved' if comment.get('resolved') else 'Unresolved'}"
    )
    
    if comment.get('related_revision_id'):
        text += f"\nRelated Revision: {comment['related_revision_id']}"
    
    if comment['replies']:
        text += "\nReplies:"
        for reply in comment['replies']:
            reply_text = (
                f"\n  - Reply by {reply['author']} on {reply['date']}:"
                f"\n    {reply['text']}"
                f"\n    Status: {'Resolved' if reply.get('resolved') else 'Unresolved'}"
            )
            text += reply_text
    return text


def prepare_revision_text(revision: Dict) -> str:
    """Format revision data into a single text string."""
    text = (
        f"[REVISION ID: {revision['id']}]\n"
        f"Position: chars {revision['position']['start']}-{revision['position']['end']}\n"
        f"Context: \"{revision['referenced_text']}\"\n"
        f"Change Type: {revision['type']}\n"
        f"Modified Text: {revision['text']}\n"
        f"Author: {revision['author']}\n"
        f"Date: {revision['date']}"
    )
    
    if revision.get('formatting'):
        formatting_str = ', '.join(revision['formatting'].keys())
        text += f"\nFormatting Changes: {formatting_str}"
        
    if revision.get('parent_id'):
        text += f"\nParent Revision: {revision['parent_id']}"
        
    return text


def extract_tables(tree: etree.Element, namespace: Dict[str, str], position_counter: int) -> List[TableData]:
    """Extract tables from the document."""
    # Placeholder for table extraction
    return []


def extract_section_text(full_text: str, start_match: str, end_match: str, section_type: str = None) -> Union[str, List[Dict[str, str]]]:
    """Helper function to extract text between start and end match strings."""
    try:
        start_idx = full_text.index(start_match)
        end_idx = full_text.index(end_match) + len(end_match)
        text = full_text[start_idx:end_idx].strip()
        return text
    except ValueError:
        logging.warning(f"Could not find match strings for section {section_type}")
        return ""


def get_comment_context(text: str, start_pos: int, end_pos: int, context_chars: int = 100) -> str:
    """Get surrounding context for a comment, up to context_chars in each direction."""
    text_len = len(text)
    context_start = max(0, start_pos - context_chars)
    context_end = min(text_len, end_pos + context_chars)
    return text[context_start:context_end]


def extract_document_history(file_path: str, write_to_file: bool = False) -> Union[Dict, str]:
    """Extract document history including images and tables."""
    logger = logging.getLogger('document_ingestion')
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    paper_title = Path(file_path).stem
    base_dir = Path('processed_documents').resolve()
    output_dir = base_dir / f"{paper_title}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    document_history = {
        'document_id': str(file_path),
        'metadata': {
            'last_modified': None,
            'contributors': []
        },
        'sections': [],
        'comments': [],
        'revisions': [],
        'tables': []
    }
    
    # First attempt: Direct PDF conversion
    pdf_path = output_dir / f"{paper_title}.pdf"
    full_text = ""
    try:
        # Try direct conversion to PDF first
        pypandoc.convert_file(
            str(Path(file_path).absolute()),
            "pdf",
            outputfile=str(pdf_path.absolute()),
            extra_args=[
                "--pdf-engine=xelatex",
                "-V", "mainfont=DejaVu Sans",
                "-V", "mathfont=DejaVu Math TeX Gyre"
            ]
        )
        
        pdf_extractor = PDFTextExtractor()
        markdown_path = pdf_extractor.extract_pdf(str(pdf_path))
        
        if not markdown_path or not Path(markdown_path).exists():
            raise ValueError("PDF extraction failed - no valid markdown path returned")

        with open(markdown_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()
            if not markdown_text.strip():
                raise ValueError("Extracted markdown is empty")
            full_text = markdown_text
            document_history['markdown'] = markdown_text
            
    except Exception as e:
        logger.warning(f"Direct PDF conversion failed: {str(e)}. Trying two-step conversion...")
        
        try:
            # Fallback: Two-step conversion through markdown
            temp_md = output_dir / f"{paper_title}_temp.md"
            pypandoc.convert_file(
                str(Path(file_path).absolute()),
                "markdown_strict",
                outputfile=str(temp_md.absolute()),
                extra_args=[
                    "--wrap=none",
                    "--columns=1000",
                    "--atx-headers"
                ]
            )

            pypandoc.convert_file(
                str(temp_md),
                "pdf",
                outputfile=str(pdf_path.absolute()),
                extra_args=[
                    "--pdf-engine=xelatex",
                    "-V", "mainfont=DejaVu Sans",
                    "-V", "geometry:margin=1in",
                    "--wrap=none",
                    "--columns=1000"
                ]
            )
            
            pdf_extractor = PDFTextExtractor()
            markdown_path = pdf_extractor.extract_pdf(str(pdf_path))
            
            if not markdown_path:
                raise ValueError("PDF extraction failed in fallback approach")

            with open(markdown_path, 'r', encoding='utf-8') as f:
                markdown_text = f.read()
                if not markdown_text.strip():
                    raise ValueError("Extracted markdown is empty")
                full_text = markdown_text
                document_history['markdown'] = markdown_text
                
        except Exception as fallback_error:
            logger.error(f"Both conversion approaches failed. Final error: {str(fallback_error)}")
            raise ValueError(f"Document conversion failed: {str(fallback_error)}")

    if not full_text:
        logger.error("No text was extracted from the document")
        raise ValueError("No text was extracted from the document")
        
    with zipfile.ZipFile(file_path, 'r') as docx:
        document_xml = docx.read('word/document.xml')
        tree = etree.fromstring(document_xml)
        namespace = {
            'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
            'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
        }

        # Find all comment reference marks and their referenced text
        comment_references = {}
        current_comment_id = None
        current_text = []
        context_window = 200  # Increased from 50 to 200 characters
        
        for element in tree.iter():
            if element.tag == f'{{{namespace["w"]}}}commentRangeStart':
                current_comment_id = element.get(f'{{{namespace["w"]}}}id')
                current_text = []
            
            elif element.tag == f'{{{namespace["w"]}}}t' and current_comment_id:
                # Get surrounding text nodes for context
                prev_text = []
                next_text = []
                
                # Look for previous siblings
                current = element
                while current is not None and len(''.join(prev_text)) < context_window:
                    if current.getprevious() is not None:
                        current = current.getprevious()
                    elif current.getparent() is not None:
                        current = current.getparent()
                    else:
                        break
                    if current.tag == f'{{{namespace["w"]}}}t':
                        prev_text.insert(0, current.text if current.text else '')
                
                # Look for next siblings
                current = element
                while current is not None and len(''.join(next_text)) < context_window:
                    if current.getnext() is not None:
                        current = current.getnext()
                    elif current.getparent() is not None:
                        current = current.getparent().getnext()
                    else:
                        break
                    if current is not None and current.tag == f'{{{namespace["w"]}}}t':
                        next_text.append(current.text if current.text else '')
                
                # Combine context with current text
                full_context = (
                    ''.join(prev_text).strip() + 
                    ' ' + (element.text if element.text else '').strip() + 
                    ' ' + ''.join(next_text).strip()
                ).strip()
                current_text.append(full_context)
            
            elif element.tag == f'{{{namespace["w"]}}}commentRangeEnd':
                comment_id = element.get(f'{{{namespace["w"]}}}id')
                if comment_id == current_comment_id:
                    comment_references[comment_id] = ' '.join(current_text).strip()
                    current_comment_id = None
                    current_text = []

        # Reset position counter for revisions
        position_counter = 0

        # Extract revisions with enhanced metadata
        for element in tree.iter():
            if element.tag == '{%s}ins' % namespace['w'] or element.tag == '{%s}del' % namespace['w']:
                revision_type = 'insertion' if 'ins' in element.tag else 'deletion'
                text = ''.join(element.itertext())
                
                # Extract author and date if available
                author = element.get('{%s}author' % namespace['w'])
                date = element.get('{%s}date' % namespace['w'])
                
                # Extract formatting information
                formatting = {}
                for child in element:
                    if child.tag == '{%s}rPr' % namespace['w']:  # Run properties
                        for prop in child:
                            formatting[prop.tag.split('}')[-1]] = True

                start = position_counter
                end = position_counter + len(text)
                expanded_start, expanded_end = get_expanded_range(start, end, full_text)
                
                revision = {
                    'id': f'rev_{len(document_history["revisions"])}',
                    'type': revision_type,
                    'text': text,
                    'author': author,
                    'date': date,
                    'position': {
                        'start': start,
                        'end': end,
                        'expanded_start': expanded_start,
                        'expanded_end': expanded_end
                    },
                    'referenced_text': full_text[expanded_start:expanded_end],
                    'formatting': formatting,
                    'parent_id': None
                }
                
                document_history['revisions'].append(revision)
                if author:
                    document_history['metadata']['contributors'].add(author)
                position_counter += len(text)

        # Extract comments with correct positions
        try:
            comments_xml = docx.read('word/comments.xml')
            comments_tree = etree.fromstring(comments_xml)
            
            comment_counter = 0
            for comment in comments_tree.findall('.//w:comment', namespace):
                original_id = comment.get(f'{{{namespace["w"]}}}id')
                original_text = comment_references.get(original_id, '')
                
                # Find position of original text in full document
                text_pos = full_text.find(original_text)
                if text_pos != -1:
                    expanded_context = get_comment_context(
                        full_text, 
                        text_pos, 
                        text_pos + len(original_text)
                    )
                else:
                    expanded_context = original_text
                
                comment_data = {
                    'id': str(comment_counter),
                    'text': ''.join(comment.itertext()),
                    'author': comment.get('{%s}author' % namespace['w']),
                    'date': comment.get('{%s}date' % namespace['w']),
                    'original_text': original_text,
                    'match_string': expanded_context,
                    'resolved': comment.get('{%s}resolved' % namespace['w']) == 'true',
                    'replies': [],
                    'related_revision_id': None
                }
                
                # Update parent reference if this is a reply
                parent_comment_id = comment.get('{%s}parentId' % namespace['w'])
                if parent_comment_id:
                    # Find parent comment and append this as reply
                    for existing_comment in document_history['comments']:
                        if existing_comment['id'] == parent_comment_id:
                            existing_comment['replies'].append(comment_data)
                            break
                else:
                    document_history['comments'].append(comment_data)
                    comment_counter += 1
                
                if comment_data['author']:
                    document_history['metadata']['contributors'].add(comment_data['author'])

        except KeyError:
            # No comments.xml file exists
            pass

        # Update metadata
        document_history['metadata']['last_modified'] = None
        document_history['metadata']['contributors'] = list(document_history['metadata']['contributors'])
        
        # After extracting full_text from PDF, identify sections
        section_identifier = SingleContextSectionIdentifier()
        try:
            sections = section_identifier.process_document(full_text)
            logger.info(f"Type of sections returned: {type(sections)}")
            
            # If sections is a dict, try to extract the list
            if isinstance(sections, dict):
                logger.info("Sections is a dict, trying to extract list")
                if 'sections' in sections:
                    sections = sections['sections']
                    logger.info(f"Extracted sections list: {sections}")
            
            if not isinstance(sections, list):
                logger.error(f"Still not a list after extraction: {type(sections)}")
                document_history['sections'] = []
                return document_history
            
            # Process sections
            processed_sections = []
            
            for section in sections:
                try:
                    # Normalize all text fields in the section
                    processed_section = {
                        'section_type': section['section_type'],
                        'match_strings': {
                            'start': normalize_unicode(section['match_strings']['start']),
                            'end': normalize_unicode(section['match_strings']['end'])
                        },
                        'text': normalize_unicode(section.get('text', ''))
                    }
                    processed_sections.append(processed_section)
                    
                except Exception as e:
                    logger.error(f"Error processing section {section.get('section_type', 'unknown')}: {e}")
                    continue
            
            # Create the document history object
            document_history = {
                'document_id': str(file_path),
                'metadata': {
                    'last_modified': document_history['metadata']['last_modified'],
                    'contributors': document_history['metadata']['contributors']
                },
                'sections': processed_sections,  # Use the processed sections
                'comments': document_history['comments'],
                'revisions': document_history['revisions'],
                'tables': extract_tables(tree, namespace, position_counter)
            }

            # Write to file if requested
            if write_to_file:
                output_file = output_dir / f"{paper_title}_processed.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(document_history, f, indent=4)
                print(f"File written to: {output_file.absolute()}")
            
            return document_history

        except Exception as e:
            logger.error(f"Error in section processing: {str(e)}")
            logger.exception("Full traceback:")  # Add full traceback
            document_history['sections'] = []
        
        # Extract tables and images
        document_history['tables'] = extract_tables(tree, namespace, position_counter)

        # Create clean document structure
        document_history = {
            'document_id': document_history['document_id'],
            'metadata': {
                'last_modified': document_history['metadata']['last_modified'],
                'contributors': document_history['metadata']['contributors']
            },
            'sections': document_history['sections'],
            'comments': document_history['comments'],
            'revisions': document_history['revisions'],
            'tables': document_history['tables']
        }

        # Write to file if requested
        if write_to_file:
            output_file = output_dir / f"{paper_title}_processed.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(document_history, f, indent=4)
            print(f"File written to: {output_file.absolute()}")
        
        # After processing sections, normalize all text fields recursively
        def normalize_text_fields(obj):
            if isinstance(obj, dict):
                return {k: normalize_text_fields(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [normalize_text_fields(item) for item in obj]
            elif isinstance(obj, str):
                return normalize_unicode(obj)
            return obj

        # Apply normalization before returning or writing to file
        document_history = normalize_text_fields(document_history)
        
        return document_history

if __name__ == "__main__":
    import json
    from pathlib import Path
    from datetime import datetime
    
    # Setup logging using the centralized configuration
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    logger = setup_logging(log_dir, timestamp, "document_ingestion")
    
    logger.info("Starting document extraction")
    document_history = extract_document_history("examples/Manuscript_Draft_PreSBReview_Final.docx", write_to_file=False)
    
    logger.info("Document processing complete. Printing results...")
    print(json.dumps(document_history, indent=4))
    
