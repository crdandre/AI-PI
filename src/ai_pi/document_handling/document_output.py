"""
This file takes the output from reviewer_agent.py and creates a Word doc
given the original document and the review output, yielding a new doc which
looks as if a reviewer left comments and suggested revisions. The idea is to
leave the user able to parse through each and address them individually using
this output.
"""
import docx
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from fuzzywuzzy import fuzz

def enable_track_changes(doc):
    """Enable track changes in the document."""
    # Create track revisions tag
    tag = OxmlElement('w:trackRevisions')
    
    # Get the settings element
    settings = doc.settings._element
    settings.append(tag)

def add_high_level_review(doc, high_level_review):
    """Add high-level review as a new section at the start of the document."""
    # Add title
    doc.add_heading('Scientific Review Summary', level=1)
    
    # Add overall assessment
    doc.add_heading('Overall Assessment', level=2)
    doc.add_paragraph(high_level_review.get('overall_assessment', 'No overall assessment provided.'))
    
    # Add key strengths
    doc.add_heading('Key Strengths', level=2)
    strengths = high_level_review.get('key_strengths', [])
    if strengths:
        for strength in strengths:
            doc.add_paragraph(strength.strip(), style='List Bullet')
    else:
        doc.add_paragraph('No key strengths specified.')
    
    # Add key weaknesses
    doc.add_heading('Key Weaknesses', level=2)
    weaknesses = high_level_review.get('key_weaknesses', [])
    if weaknesses:
        for weakness in weaknesses:
            doc.add_paragraph(weakness.strip(), style='List Bullet')
    else:
        doc.add_paragraph('No key weaknesses specified.')
    
    # Add recommendations
    doc.add_heading('Key Recommendations', level=2)
    recommendations = high_level_review.get('recommendations', [])
    if recommendations:
        for rec in recommendations:
            doc.add_paragraph(rec.strip(), style='List Bullet')
    else:
        doc.add_paragraph('No recommendations provided.')
    
    # Add communication review section
    if 'communication_review' in high_level_review:
        doc.add_heading('Writing and Communication Review', level=2)
        comm_review = high_level_review['communication_review']
        
        # Writing assessment (as regular paragraph)
        doc.add_paragraph(comm_review.get('writing_assessment', 'No writing assessment provided.'))
        
        # Narrative strengths
        if comm_review.get('narrative_strengths'):
            doc.add_heading('Narrative Strengths', level=3)
            for strength in comm_review['narrative_strengths']:
                clean_text = strength.strip().lstrip('•').strip()
                doc.add_paragraph(clean_text, style='List Bullet')
        
        # Areas for improvement
        if comm_review.get('narrative_weaknesses'):
            doc.add_heading('Areas for Improvement', level=3)
            for weakness in comm_review['narrative_weaknesses']:
                clean_text = weakness.strip().lstrip('•').strip()
                doc.add_paragraph(clean_text, style='List Bullet')
        
        # Style recommendations
        if comm_review.get('style_recommendations'):
            doc.add_heading('Style Recommendations', level=3)
            for rec in comm_review['style_recommendations']:
                clean_text = rec.strip().lstrip('•').strip()
                doc.add_paragraph(clean_text, style='List Bullet')
    
    # Add page break before the original document
    doc.add_page_break()

def output_commented_document(input_doc_path, document_review_items, output_doc_path, match_threshold=90, verbose=False):
    """Process the document and add AI review comments and suggestions."""
    # Create new document for output
    doc = docx.Document()
    
    # Add high-level review from final_review if it exists
    if 'final_review' in document_review_items:
        high_level_review = {
            'overall_assessment': document_review_items['final_review']['overall_assessment'],
            'key_strengths': document_review_items['final_review']['key_strengths'],
            'key_weaknesses': document_review_items['final_review']['key_weaknesses'],
            'recommendations': document_review_items['final_review']['recommendations']
        }
        add_high_level_review(doc, high_level_review)
    
    # Load and append original document
    original_doc = docx.Document(input_doc_path)
    for element in original_doc.element.body:
        doc.element.body.append(element)
    
    enable_track_changes(doc)
    
    print(f"Processing document with {len(doc.paragraphs)} paragraphs")
    
    # Flatten section reviews into lists
    all_match_strings = []
    all_comments = []
    all_revisions = []
    
    for section_review in document_review_items.get('section_reviews', []):
        review = section_review.get('review', {})
        all_match_strings.extend(review.get('match_strings', []))
        all_comments.extend(review.get('comments', []))
        all_revisions.extend(review.get('revisions', []))
    
    print("\nLooking for these matches:", all_match_strings)
    
    matches_found = 0
    # Keep original matches in a list that won't be modified
    all_matches = list(zip(all_match_strings, all_comments, all_revisions))
    
    # Track which matches have been successfully processed
    processed_matches = set()
    
    # Iterate through each paragraph in the document
    for i, paragraph in enumerate(doc.paragraphs):
        text = paragraph.text.strip()
        normalized_text = ' '.join(text.split())
        
        # Try each match that hasn't been processed yet
        for match, comment, revision in all_matches:
            # Skip if we've already processed this match
            if match in processed_matches:
                continue
                
            normalized_match = ' '.join(match.split())
            
            # Try exact match first with more context
            full_context = normalized_text
            if normalized_match in full_context:
                match_ratio = 100
                match_location = full_context.index(normalized_match)
                text = full_context
                match = normalized_match
                if verbose:
                    print(f"Exact match found for: '{match}'")
            else:
                # Enhanced fuzzy matching with better context handling
                match_ratio = fuzz.token_set_ratio(normalized_match, full_context)
                if match_ratio >= match_threshold:
                    try:
                        # Use larger context windows for better matching
                        window_size = len(normalized_match) * 2
                        windows = []
                        
                        for i in range(max(0, len(text) - window_size + 1)):
                            window_text = text[i:i + window_size]
                            window_score = fuzz.token_set_ratio(normalized_match, window_text)
                            windows.append((window_text, i, window_score))
                        
                        # Sort windows by score and get the best match
                        best_window = max(windows, key=lambda x: x[2])
                        match_location = best_window[1]
                        
                        # Verify the match position
                        context_before = text[max(0, match_location-50):match_location]
                        context_after = text[match_location:match_location+len(normalized_match)+50]
                        
                        if verbose:
                            print(f"Context before: '{context_before}'")
                            print(f"Matched text: '{text[match_location:match_location+len(normalized_match)]}'")
                            print(f"Context after: '{context_after}'")
                            
                    except Exception as e:
                        if verbose:
                            print(f"Error in fuzzy matching: {str(e)}")
                        continue
                else:
                    continue
            
            try:
                # Split the paragraph into parts
                before_match = text[:match_location]
                after_match = text[match_location + len(match):]
                
                # Clear the paragraph's content
                paragraph.clear()
                
                # Add text before the match
                if before_match:
                    paragraph.add_run(before_match)
                
                # Add the matched text with comment and revision
                if revision and revision.strip():  # Only process revision if it has content
                    # Create deletion run and add comment to it
                    del_run = paragraph.add_run()
                    
                    # Create run properties
                    rPr = OxmlElement('w:rPr')
                    
                    # Create deletion element
                    del_element = OxmlElement('w:del')
                    del_element.set(qn('w:author'), 'AIPI')
                    del_element.set(qn('w:date'), '2024-03-21T12:00:00Z')
                    
                    # Create text element for deletion
                    del_text = OxmlElement('w:t')
                    del_text.set(qn('xml:space'), 'preserve')
                    del_text.text = match
                    
                    # Build the XML structure
                    del_run._element.append(rPr)
                    del_run._element.append(del_element)
                    del_element.append(del_text)
                    
                    # Add comment to the deletion run
                    del_run.add_comment(f"{comment} (Match confidence: {match_ratio}%)", author="AIPI", initials="AI")
                    
                    # Create insertion run
                    ins_run = paragraph.add_run()
                    rPr = OxmlElement('w:rPr')
                    ins_element = OxmlElement('w:ins')
                    ins_element.set(qn('w:author'), 'AIPI')
                    ins_element.set(qn('w:date'), '2024-03-21T12:00:00Z')
                    ins_text = OxmlElement('w:t')
                    ins_text.set(qn('xml:space'), 'preserve')
                    ins_text.text = revision
                    ins_run._element.append(rPr)
                    ins_run._element.append(ins_element)
                    ins_element.append(ins_text)
                else:
                    # Just add comment without revision
                    match_run = paragraph.add_run(match)
                    match_run.add_comment(comment, author="AIPI", initials="AI")
                
                # Add text after the match
                if after_match:
                    paragraph.add_run(after_match)
                
                # Mark this match as processed
                processed_matches.add(match)
                matches_found += 1
                
                print(f"Found match: '{match}' with confidence {match_ratio}%")
                print(f"Added comment: '{comment}'")
                if revision and revision.strip():
                    print(f"Added revision: '{revision}'")
                    
            except Exception as e:
                print(f"Error processing match '{match}': {str(e)}")
                continue
    
    # Report unmatched strings
    unmatched = set(m[0] for m in all_matches) - processed_matches
    if unmatched:
        print("\nWarning: The following matches were not found in the document:")
        for match in unmatched:
            print(f"- '{match}'")
    
    print(f"\nTotal matches found: {matches_found}")
    print(f"Saving document to {output_doc_path}")
    doc.save(output_doc_path)

if __name__ == "__main__":
    # Test data with high-level review
    test_review_items = {
        'match_strings': [
            'The current study will build upon our previously published work on pediatric patient-specific FE modeling and growth', 
            'Vertebral growth was modeled through adaptation of a region-specific orthotropic thermal expansion method described by Balasubramanian', 
            'and adjusted linearly according to Risser', 
        ],
        'revisions': [
            'The current study builds upon our previously published work on pediatric patient-specific FE modeling and growth, which has shown promising results in predicting scoliotic curve progression.',
            'Vertebral growth was modeled using a region-specific orthotropic thermal expansion method, as described by Balasubramanian et al. (reference), which has been validated in previous studies.',
            'The value for was initially set to 0.4 MPa-1, as reported by Shi et al. (reference), and then adjusted linearly based on the Risser sign, with a clear explanation of how the adjustment was made.',
        ], 
        'comments': [
            'Added context to clarify the significance of the current study.', 
            'Added reference and context to support the methodology.', 
            'Improved clarity and added reference to support the methodology.'
        ],
        'high_level_review': {
            'overall_assessment': 'This manuscript presents a novel approach to modeling scoliotic progression using finite element analysis. While the methodology is sound, there are several areas where additional validation and clarity would strengthen the scientific contribution.',
            'key_strengths': [
                'Innovative integration of growth modeling with FE analysis',
                'Clear theoretical foundation based on established biomechanical principles',
                'Practical clinical applications for predicting curve progression'
            ],
            'key_weaknesses': [
                'Limited validation against clinical data',
                'Assumptions in growth model need more justification',
                'Statistical analysis of model accuracy could be more robust'
            ],
            'recommendations': [
                'Include validation against a larger clinical dataset',
                'Provide detailed sensitivity analysis for growth parameters',
                'Add statistical confidence intervals for prediction accuracy',
                'Clarify the clinical implications of the model\'s accuracy limits'
            ],
            'communication_review': {
                'writing_assessment': "The paper presents complex ideas clearly, but lacks smooth transitions between sections. Technical concepts are well-explained for the target audience.",
                'narrative_strengths': ["Clear problem statement", "Effective use of examples"],
                'narrative_weaknesses': ["Section transitions need work", "Methods section assumes too much background knowledge"],
                'style_recommendations': ["Add transition paragraphs between major sections", "Include more context for technical terms"]
            }
        }
    }
    
    # Test paths - adjust these to your actual file locations
    input_path = "examples/ScolioticFEPaper_v7.docx"
    output_path = "examples/test_output_with_review.docx"
    
    try:
        output_commented_document(input_path, test_review_items, output_path, verbose=True)
        print(f"Successfully created reviewed document with high-level summary at {output_path}")
    except Exception as e:
        print(f"Error processing document: {str(e)}")

