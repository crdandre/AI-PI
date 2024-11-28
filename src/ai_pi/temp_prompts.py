def generate_review_prompt(doc_context, prior_review_context):
    """
    Generates a prompt for the reviewer agent to create a high-level review
    
    Args:
        doc_context (str): The text content of the current document
        prior_review_context (str): Context from previous reviews
        
    Returns:
        str: A formatted prompt for the review agent
    """
    return f"""Please review this academic document in the style indicated by the prior review context.
        
    Prior Review Context:
    {prior_review_context}

    Document to Review:
    {doc_context}

    Using the style, tone, and focus areas shown in the prior review context, please generate a comprehensive review that includes:
    1. Overall assessment of the work
    2. Major strengths
    3. Key weaknesses
    4. High-level suggestions for improvement
    5. Writing feedback

    Focus particularly on the types of issues that were commonly raised in prior reviews.
    """

def generate_comments_and_revisions_prompt(review, doc_context, prior_review_context):
    """
    Generates a prompt for the reviewer agent to create specific comments and revision suggestions
    
    Args:
        review (str): The high-level review generated previously
        doc_context (str): The text content of the current document
        prior_review_context (str): Context from previous reviews
        
    Returns:
        str: A formatted prompt for generating specific comments and revisions
    """
    return f"""Based on the high-level review and prior review patterns, please generate specific comments and revision suggestions for this document.

    Prior Review Context:
    {prior_review_context}

    High-level Review:
    {review}

    Document Text:
    {doc_context}

    For each issue identified in the review, please:
    1. Locate relevant sections in the document text
    2. Generate specific comments explaining the issue
    3. Propose concrete revisions using REVISION_START/END blocks that include:
    - Location in the text
    - Original text
    - Suggested changes
    - Explanation of why the change improves the document

    Match the tone and style of feedback shown in the prior review context.
    Focus on similar types of issues that were commonly addressed in previous reviews.
    """

context_agent_directive = "Find the comments left for the prior documents. Capture the essence of when the feedback was given, by whom (provide names), and for what reason. The idea is to be able to map that type of feedback to another new document using specific details. Capture the tone of the feedback and related personality traits as well."

revise_agent_directive = """
    You are EXACTING! The pinnacle of precision. 
    
    Can you generate a list of revisions using this context? For each item, provide three things (IN THIS ORDER): a match string which contains text from the original document which can be used to place comments (CRUCIAL IT IS ACTUAL TEXT VERBATIM FROM THE DOCUMENT!!!!!!!), a comment on that match string, and a suggested revision. 
    
    The format is: list[list[str]],
    Where the lowest level list contains (these are placeholders) ["*match_string*", "*comment*", "*revision*"] this format of data for each entry. 
    
    Only include the necessary information for the Word doc. I.e. only the exact match string, no additional tags.
    
    After you're done assembling this, go back and think.....slowly....reason about whether the suggested match strings are exact matches in the document and correct them if not.
"""

call_review_directive = "Can you generate a short review of this paper using this context? Suggest high level changes if there should be any."