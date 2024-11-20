"""
Extracts the process of revision from first draft to final draft,
including comments, formatting changes, and their relationships.
"""

from docx import Document
from lxml import etree
import zipfile
from datetime import datetime
from typing import Dict, List, Optional, TypedDict

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
    
    return expanded_start, expanded_end

def extract_document_history(file_path: str) -> Dict:
    with zipfile.ZipFile(file_path, 'r') as docx:
        document_xml = docx.read('word/document.xml')
        tree = etree.fromstring(document_xml)
        namespace = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

        # Build full document text first
        full_text = []
        for element in tree.iter():
            if element.tag == f'{{{namespace["w"]}}}t':
                full_text.append(element.text if element.text else '')
        full_text = ''.join(full_text)

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
            
            for comment in comments_tree.findall('.//w:comment', namespace):
                comment_id = comment.get('{%s}id' % namespace['w'])
                position = comment_positions.get(comment_id, {'start': 0, 'end': 0})
                
                start = position['start']
                end = position['end']
                expanded_start, expanded_end = get_expanded_range(start, end, full_text)
                
                comment_data = {
                    'id': comment_id,
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
                
                # Check if this is a reply to another comment
                parent_comment_id = comment.get('{%s}parentId' % namespace['w'])
                if parent_comment_id:
                    # Find parent comment and append this as reply
                    for existing_comment in document_history['comments']:
                        if existing_comment['id'] == parent_comment_id:
                            existing_comment['replies'].append(comment_data)
                            break
                else:
                    document_history['comments'].append(comment_data)
                
                if comment_data['author']:
                    document_history['metadata']['contributors'].add(comment_data['author'])

        except KeyError:
            # No comments.xml file exists
            pass

        # Update metadata
        document_history['metadata']['revision_count'] = len(document_history['revisions'])
        document_history['metadata']['comment_count'] = len(document_history['comments'])
        document_history['metadata']['contributors'] = list(document_history['metadata']['contributors'])
        
        return document_history
    

#Testing Usage
if __name__ == "__main__":
    document_history = extract_document_history("examples/ScolioticFEPaper_v7.docx")
    import json
    print(json.dumps(document_history, indent=4, default=str))
