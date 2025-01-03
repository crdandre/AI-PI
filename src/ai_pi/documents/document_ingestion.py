"""
Extracts the process of revision from first draft to final draft,
including comments, formatting changes, and their relationships.

Currently removes references section. If it's desired to retain this information,
changes would be needed to index the references and format them with appropriate
context.
"""

from lxml import etree
import zipfile
from typing import Dict, List, Optional, TypedDict, Union
from pathlib import Path
import re
from src.ai_pi.documents.section_identifier import SingleContextSectionIdentifier
import dspy

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
    position: Dict[str, int]
    referenced_text: str
    resolved: bool
    replies: List['Comment']
    related_revision_id: Optional[str]

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
    
    # Clean citations from the referenced text before returning
    referenced_text = full_text[expanded_start:expanded_end]
    referenced_text = clean_citations(referenced_text)
    
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

def clean_citations(text: str) -> str:
    """Remove citations and references section from text while preserving readability."""
    # Remove the References section and everything after it
    if 'References' in text:
        text = text.split('References')[0]
    # Remove parenthetical citations like (Smith et al., 2020)
    text = re.sub(r'\([^()]*?\d{4}[^()]*?\)', '', text)
    # Remove numbered citations like [1] or [1,2,3]
    text = re.sub(r'\[\d+(?:,\s*\d+)*\]', '', text)
    # Remove multiple spaces created by citation removal
    text = re.sub(r'\s+', ' ', text)
    # Fix spacing around punctuation
    text = re.sub(r'\s+([.,;:])', r'\1', text)
    
    return text.strip()

def extract_document_history(file_path: str, lm: dspy.LM = None, write_to_file: bool = False) -> Union[Dict, str]:
    """
    Extract document history and format as text, optionally saving to file.
    
    Args:
        file_path: Path to the Word document
        lm: Language model for section identification
        write_to_file: Whether to write the processed content to a file (default: False)
    
    Returns:
        If write_to_file is True: Dictionary containing document history
        If write_to_file is False: Formatted string containing document history
    """
    document_history = {}
    
    with zipfile.ZipFile(file_path, 'r') as docx:
        document_xml = docx.read('word/document.xml')
        tree = etree.fromstring(document_xml)
        namespace = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

        # Build full text with formatting
        full_text = []
        current_formatting = {}
        
        for element in tree.iter():
            if element.tag == f'{{{namespace["w"]}}}p':  # Paragraph
                # Check if this is a heading
                style_element = element.find(f'.//{{{namespace["w"]}}}pStyle', namespace)
                if style_element is not None:
                    style = style_element.get(f'{{{namespace["w"]}}}val')
                    if style == 'Heading1':  # Only preserve Heading1
                        current_formatting['heading'] = 1
                        full_text.append(f"\n# ")  # Markdown-style heading
                    else:
                        current_formatting.pop('heading', None)
                else:
                    current_formatting.pop('heading', None)
                    full_text.append('\n')  # New paragraph
                    
            elif element.tag == f'{{{namespace["w"]}}}r':  # Run
                # Check for text formatting
                rPr = element.find(f'.//{{{namespace["w"]}}}rPr', namespace)
                if rPr is not None:
                    if rPr.find(f'.//{{{namespace["w"]}}}b', namespace) is not None:
                        current_formatting['bold'] = True
                    if rPr.find(f'.//{{{namespace["w"]}}}i', namespace) is not None:
                        current_formatting['italic'] = True
                
            elif element.tag == f'{{{namespace["w"]}}}t':  # Text
                text = element.text if element.text else ''
                if current_formatting.get('bold'):
                    text = f"**{text}**"
                if current_formatting.get('italic'):
                    text = f"*{text}*"
                full_text.append(text)
                current_formatting = {}  # Reset formatting after applying
        
        full_text = ''.join(full_text)
        # Remove multiple newlines
        full_text = re.sub(r'\n{3,}', '\n\n', full_text)
        
        # Clean citations from the full text
        full_text = clean_citations(full_text)

        # Initialize document history
        document_history = {
            'document_id': file_path,
            'revisions': [],
            'comments': [],
            'metadata': {
                'last_modified': None,
                'revision_count': 0,
                'comment_count': 0,
                'contributors': set()
            }
        }

        # First, find all comment reference marks and their positions
        comment_positions = {}
        position_counter = 0
        
        for element in tree.iter():
            # Track text content for position counting
            if element.tag == f'{{{namespace["w"]}}}t':
                position_counter += len(element.text if element.text else '')
            
            # Find comment start and end markers
            elif element.tag == f'{{{namespace["w"]}}}commentRangeStart':
                comment_id = element.get(f'{{{namespace["w"]}}}id')
                if comment_id not in comment_positions:
                    comment_positions[comment_id] = {'start': position_counter}
            
            elif element.tag == f'{{{namespace["w"]}}}commentRangeEnd':
                comment_id = element.get(f'{{{namespace["w"]}}}id')
                if comment_id in comment_positions:
                    comment_positions[comment_id]['end'] = position_counter

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
                for child in element.iter():
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
                original_id = comment.get('{%s}id' % namespace['w'])
                new_id = str(comment_counter)
                
                position = comment_positions.get(original_id, {'start': 0, 'end': 0})
                
                start = position['start']
                end = position['end']
                expanded_start, expanded_end = get_expanded_range(start, end, full_text)
                
                comment_data = {
                    'id': new_id,
                    'text': ''.join(comment.itertext()),
                    'author': comment.get('{%s}author' % namespace['w']),
                    'date': comment.get('{%s}date' % namespace['w']),
                    'position': {
                        'start': start,
                        'end': end,
                        'expanded_start': expanded_start,
                        'expanded_end': expanded_end
                    },
                    'referenced_text': full_text[expanded_start:expanded_end],
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
        document_history['metadata']['revision_count'] = len(document_history['revisions'])
        document_history['metadata']['comment_count'] = len(document_history['comments'])
        document_history['metadata']['contributors'] = list(document_history['metadata']['contributors'])
        
        # After extracting full_text, identify sections
        section_identifier = SingleContextSectionIdentifier(engine=lm)
        sections = section_identifier.process_document(full_text)
        
        # Add the actual text to each section
        for section in sections:
            start_match = section['match_strings']['start']
            end_match = section['match_strings']['end']
            section['text'] = extract_section_text(full_text, start_match, end_match)
        
        # # Add sections to document_history
        document_history['sections'] = sections
        
        # Update formatting to include section information
        formatted_text = []
        formatted_text.append("=== DOCUMENT METADATA ===")
        formatted_text.append(f"Document ID: {document_history['document_id']}")
        formatted_text.append(f"Last Modified: {document_history['metadata']['last_modified']}")
        formatted_text.append(f"Contributors: {', '.join(document_history['metadata']['contributors'])}\n")
        
        formatted_text.append("=== DOCUMENT CONTENT ===")
        formatted_text.append(full_text)
        formatted_text.append("\n")
        
        if document_history['comments']:
            formatted_text.append("=== COMMENTS ===")
            for comment in document_history['comments']:
                formatted_text.append(prepare_comment_text(comment))
                formatted_text.append("\n")
        
        if document_history['revisions']:
            formatted_text.append("=== REVISIONS ===")
            for revision in document_history['revisions']:
                formatted_text.append(prepare_revision_text(revision))
                formatted_text.append("\n")
        
        formatted_text.append("=== DOCUMENT SECTIONS ===")
        for section in sections:
            formatted_text.append(f"\n=== {section['type'].upper()} ===")
            formatted_text.append(section['text'])
        
        formatted_text = "\n".join(formatted_text)
        
        if write_to_file:
            # Create processed_documents directory if it doesn't exist
            output_dir = Path('processed_documents')
            output_dir.mkdir(exist_ok=True)
            
            # Generate base filename from input file
            base_filename = Path(file_path).stem
            output_file = output_dir / f"{base_filename}_processed.txt"
            
            # Write formatted text to file
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(formatted_text)
            
            print(f"File written to: {output_file.absolute()}")
            return document_history
        
        return formatted_text
    

def extract_section_text(full_text: str, start_match: str, end_match: str) -> str:
    """Helper function to extract text between start and end match strings."""
    try:
        start_idx = full_text.index(start_match)
        end_idx = full_text.index(end_match) + len(end_match)
        return full_text[start_idx:end_idx].strip()
    except ValueError:
        return ""

#Testing Usage
if __name__ == "__main__":
    import os, json
    
    openrouter_model = 'openrouter/openai/gpt-4o'
    
    # Initialize and run
    lm = dspy.LM(
        openrouter_model,
        api_base="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        temperature=0.01,
    )
    
    document_history = extract_document_history("examples/ScolioticFEPaper_v7.docx", write_to_file=True, lm=lm)
    
    # Print summary of processed file
    output_dir = Path('processed_documents')
    processed_file = output_dir / f"ScolioticFEPaper_v7_processed2.txt"
    print(f"\nProcessed file saved as: {processed_file.absolute()}")
    
    print(json.dumps(document_history, indent=4))
