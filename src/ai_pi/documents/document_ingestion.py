"""
Extracts:
- Document Info, Sections
- Existing comments
- Images, Tables
"""

from lxml import etree
import zipfile
from typing import Dict, List, Optional, TypedDict, Union
from pathlib import Path
import re
from src.ai_pi.documents.section_identifier import SingleContextSectionIdentifier
import dspy
import io
from PIL import Image
from datetime import datetime
import os


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


class TableData(TypedDict):
    id: str
    content: List[List[str]]
    position: Dict[str, int]
    caption: Optional[str]


class ImageData(TypedDict):
    id: str
    path: str  # Changed from 'data' to 'path'
    format: str
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
    
    # Clean citations from the referenced text before returning
    referenced_text = full_text[expanded_start:expanded_end]
    #referenced_text = clean_citations(referenced_text)
    
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

def clean_xml_and_b64(text: str) -> str:
    """Remove XML tags, base64 content, and other non-content markup from text."""
    # Remove any XML/HTML-like tags and their contents
    text = re.sub(r'<[^>]+>.*?</[^>]+>', '', text)
    
    # Remove standalone XML/HTML-like tags
    text = re.sub(r'<[^>]+/?>', '', text)
    
    # Remove base64 encoded data and long strings of encoded characters
    text = re.sub(r'[A-Za-z0-9+/=\n]{50,}', '', text)
    
    # Remove EndNote and similar reference manager tags
    text = re.sub(r'ADDIN\s+[A-Z.]+(?:\s*\{[^}]*\})?', '', text)
    text = re.sub(r'SEQ\s+(?:Table|Figure)\s*\\[^"]*', '', text)
    
    # Clean up citation brackets while preserving simple numeric citations
    citations = re.findall(r'\[[\d\s,]+\]', text)
    text = re.sub(r'\[.*?\]', '', text)
    
    # Add back simple numeric citations
    if citations:
        text = text.strip() + ' ' + ' '.join(citations)
    
    # Remove multiple spaces, newlines and trim
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def extract_tables(tree: etree.Element, namespace: Dict[str, str], position_counter: int) -> List[TableData]:
    """Extract tables from the document."""
    tables = []
    
    for element in tree.iter(f'{{{namespace["w"]}}}tbl'):
        rows = []
        for row in element.iter(f'{{{namespace["w"]}}}tr'):
            cells = []
            for cell in row.iter(f'{{{namespace["w"]}}}tc'):
                cell_text = clean_xml_and_b64(''.join(cell.itertext()).strip())
                cells.append(cell_text)
            rows.append(cells)
        
        # Find and clean caption if exists
        caption = None
        prev = element.getprevious()
        if prev is not None and prev.tag == f'{{{namespace["w"]}}}p':
            caption_text = ''.join(prev.itertext())
            if caption_text.lower().startswith('table'):
                caption = clean_xml_and_b64(caption_text)

        table_data = {
            'id': f'table_{len(tables)}',
            'content': rows,
            'position': {'start': position_counter, 'end': position_counter},
            'caption': caption
        }
        tables.append(table_data)
    
    return tables


def extract_images(docx: zipfile.ZipFile, tree: etree.Element, namespace: Dict[str, str], 
                  position_counter: int, output_dir: Path) -> List[ImageData]:
    """
    Extract images from the document and save to filesystem.
    Images are saved to: output_dir/images/image_{n}.jpeg
    """
    images = []
    
    # Create images directory
    images_dir = output_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Read relationships file
        rels = docx.read('word/_rels/document.xml.rels')
        rels_tree = etree.fromstring(rels)
        
        # Create relationship ID to image path mapping
        image_rels = {
            rel.get('Id'): rel.get('Target')
            for rel in rels_tree.iter()
            if 'image' in rel.get('Type', '')
        }
        
        # Process images in document
        for element in tree.iter(f'{{{namespace["w"]}}}drawing'):
            blip = element.find(
                './/a:blip',
                {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main'}
            )
            
            if blip is not None:
                rid = blip.get(f'{{{namespace["r"]}}}embed')
                if rid in image_rels:
                    image_path = image_rels[rid]
                    try:
                        # Extract image data
                        image_data = docx.read(f'word/{image_path}')
                        
                        # Convert to JPEG with white background
                        try:
                            # Open image using PIL
                            img = Image.open(io.BytesIO(image_data))
                            
                            # Create a white background image
                            white_bg = Image.new('RGB', img.size, (255, 255, 255))
                            
                            # If image has transparency (RGBA or P mode)
                            if img.mode in ('RGBA', 'P'):
                                # Paste the image onto white background using alpha channel as mask
                                if img.mode == 'P':
                                    img = img.convert('RGBA')
                                white_bg.paste(img, mask=img.split()[3])
                                img = white_bg
                            else:
                                # If no transparency, just convert to RGB
                                img = img.convert('RGB')
                            
                            # Generate image filename
                            image_id = f'image_{len(images)}'
                            image_filename = f'{image_id}.jpeg'
                            image_filepath = images_dir / image_filename
                            
                            # Save as JPEG file
                            img.save(image_filepath, format='JPEG', quality=85)
                            
                        except Exception as e:
                            print(f"Error converting image to JPEG: {e}")
                            continue
                        
                        # Find caption
                        caption = None
                        next_elem = element.getnext()
                        if next_elem is not None and next_elem.tag == f'{{{namespace["w"]}}}p':
                            caption_text = ''.join(next_elem.itertext())
                            if caption_text.lower().startswith('figure'):
                                caption = clean_xml_and_b64(caption_text)

                        image_info = {
                            'id': image_id,
                            'path': str(image_filepath.relative_to(output_dir)),  # Store relative path
                            'format': 'jpeg',
                            'position': {'start': position_counter, 'end': position_counter},
                            'caption': caption
                        }
                        images.append(image_info)
                        
                    except KeyError:
                        continue
    
    except KeyError:
        pass
    
    return images


def extract_section_text(full_text: str, start_match: str, end_match: str, section_type: str = None) -> Union[str, List[Dict[str, str]]]:
    """Helper function to extract text between start and end match strings."""
    try:
        start_idx = full_text.index(start_match)
        end_idx = full_text.index(end_match) + len(end_match)
        text = full_text[start_idx:end_idx].strip()
        return text
    except ValueError:
        return ""


def extract_document_history(file_path: str, lm: dspy.LM = None, write_to_file: bool = False, include_formatting: bool = True) -> Union[Dict, str]:
    """Extract document history including images and tables."""
    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    paper_title = Path(file_path).stem
    output_dir = Path('processed_documents') / f"{paper_title}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    document_history = {}
    
    with zipfile.ZipFile(file_path, 'r') as docx:
        document_xml = docx.read('word/document.xml')
        tree = etree.fromstring(document_xml)
        namespace = {
            'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
            'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'
        }

        # Build full text with formatting
        full_text = []
        current_formatting = {}
        
        for element in tree.iter():
            if element.tag == f'{{{namespace["w"]}}}p':  # Paragraph
                # Always add paragraph breaks
                full_text.append('\n')
                
                if include_formatting:
                    # Check if this is a heading
                    style_element = element.find(f'.//{{{namespace["w"]}}}pStyle', namespace)
                    if style_element is not None:
                        style = style_element.get(f'{{{namespace["w"]}}}val')
                        if style == 'Heading1' or any(section_name.lower() in ''.join(element.itertext()).lower() 
                                                     for section_name in ['abstract', 'introduction', 'methods', 
                                                                        'results', 'discussion', 'conclusion']):
                            current_formatting['heading'] = 1
                            full_text[-1] = f"\n# "  # Replace last newline with markdown heading
                    
            elif element.tag == f'{{{namespace["w"]}}}r':  # Run
                # Check for text formatting only if enabled
                if include_formatting:
                    rPr = element.find(f'.//{{{namespace["w"]}}}rPr', namespace)
                    if rPr is not None:
                        if rPr.find(f'.//{{{namespace["w"]}}}b', namespace) is not None:
                            current_formatting['bold'] = True
                        if rPr.find(f'.//{{{namespace["w"]}}}i', namespace) is not None:
                            current_formatting['italic'] = True
                
            elif element.tag == f'{{{namespace["w"]}}}t':  # Text
                text = element.text if element.text else ''
                if include_formatting:
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
        #full_text = clean_citations(full_text)

        # Initialize document history
        document_history = {
            'document_id': file_path,
            'metadata': {
                'last_modified': None,
                'contributors': set()
            },
            'sections': [],
            'comments': [],
            'revisions': []
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
        document_history['metadata']['last_modified'] = None
        document_history['metadata']['contributors'] = list(document_history['metadata']['contributors'])
        
        # After extracting full_text, identify sections
        section_identifier = SingleContextSectionIdentifier(engine=lm)
        sections = section_identifier.process_document(full_text)
        
        # Add the actual text to each section
        for section in sections:
            start_match = section['match_strings']['start']
            end_match = section['match_strings']['end']
            section['text'] = extract_section_text(
                full_text, 
                start_match, 
                end_match,
                section_type=section['type']
            )
        
        # # Add sections to document_history
        document_history['sections'] = sections
        
        # Extract tables and images
        document_history['tables'] = extract_tables(tree, namespace, position_counter)
        document_history['images'] = extract_images(docx, tree, namespace, position_counter, output_dir)

        # Create clean document structure
        document_history = {
            'document_id': document_history['document_id'],
            'metadata': {
                'last_modified': document_history['metadata']['last_modified'],
                'contributors': document_history['metadata']['contributors']
            },
            'sections': [{
                'type': section['type'],
                'text': section['text'],
                'match_strings': section['match_strings']
            } for section in sections],
            'comments': document_history['comments'],
            'revisions': document_history['revisions'],
            'tables': document_history['tables'],
            'images': document_history['images']
        }

        # Write to file if requested
        if write_to_file:
            output_file = output_dir / f"{paper_title}_processed.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(document_history, f, indent=4)
            print(f"File written to: {output_file.absolute()}")
        
        return document_history

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
    
    document_history = extract_document_history("examples/ScolioticFEPaper_v7.docx", write_to_file=False, lm=lm, include_formatting=True)
    # document_history = extract_document_history("examples/NormativeFEPaper_v7.docx", write_to_file=False, lm=lm, include_formatting=True)
    
    # Print summary of processed file
    output_dir = Path('processed_documents')
    processed_file = output_dir / f"ScolioticFEPaper_v7_processed2.txt"
    
    print(json.dumps(document_history, indent=4))
    
