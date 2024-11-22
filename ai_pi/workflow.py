"""
Orchestrates execution of this workflow:
- Document upload and processing (document_preprocessing.py)
- Context embedding (embed_documents.py)
- Gathering relevant context from the uploaded documents and any other added datastores to capture the key directives in reviewing a document (context_agent.py)
- Creating comment and revision suggestions given the relevant context (reviewer_agent.py)
- Outputting a .docx with the comments and revisions added (document_output.py)
"""

from ai_pi.context_agent import ContextAgent, DSPyContextAgent
from ai_pi.reviewer_agent import ReviewerAgent
from ai_pi.document_preprocessing import extract_document_history

context_agent = ContextAgent(similarity_top_k=20, verbose=True)
reviewer_agent = ReviewerAgent()

# Get formatted text directly
doc_context = extract_document_history("examples/ScolioticFEPaper_v7.docx")

prior_review_context = context_agent.query("Find the comments left for the prior documents. Capture the essence of when the feedback was given, by whom (provide names), and for what reason. The idea is to be able to map that type of feedback to another new document using specific details. Capture the tone of the feedback and related personality traits as well.")

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


review_prompt = generate_review_prompt(doc_context, prior_review_context)

review = reviewer_agent.call_review(
    context=review_prompt,
    question=review_prompt + "Can you generate a review using this context?"
)

print(review)

comments_and_revisions_prompt = generate_comments_and_revisions_prompt(
    review, doc_context, prior_review_context
)

comments_and_revisions = reviewer_agent.call_revise(
    context=comments_and_revisions_prompt,
    question=comments_and_revisions_prompt + "Can you generate a list of revisions using this context? For each item, provide three things: a match string in the original doc, a comment on that match string, and a suggested revision."
)

print(comments_and_revisions)