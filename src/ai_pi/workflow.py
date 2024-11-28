"""
Orchestrates execution of this workflow:
- Document upload and processing (document_preprocessing.py)
- Context embedding (embed_documents.py)
- Gathering relevant context from the uploaded documents and any other added datastores to capture the key directives in reviewing a document (context_agent.py)
- Creating comment and revision suggestions given the relevant context (reviewer_agent.py)
- Outputting a .docx with the comments and revisions added (document_output.py)
"""

from ai_pi.context_agent import ContextAgent
from ai_pi.reviewer_agent import ReviewerAgent
from ai_pi.document_preprocessing import extract_document_history
from ai_pi.document_output import output_commented_document
from ai_pi.temp_prompts import *

context_agent = ContextAgent(similarity_top_k=20, verbose=True)
reviewer_agent = ReviewerAgent()

# Get formatted text directly
input_doc_path = "examples/ScolioticFEPaper_v7.docx"
doc_context = extract_document_history(input_doc_path)

prior_review_context = context_agent.query(context_agent_directive)

review_prompt = generate_review_prompt(doc_context, prior_review_context)

review = reviewer_agent.call_review(
    context=review_prompt,
    question=review_prompt + call_review_directive
)

print(review)

comments_and_revisions_prompt = generate_comments_and_revisions_prompt(
    review, doc_context, prior_review_context
)

comments_and_revisions = reviewer_agent.call_revise(
    context=comments_and_revisions_prompt,
    question=comments_and_revisions_prompt + revise_agent_directive
)

document_review_items = {
    "match_strings": [item[0] for item in comments_and_revisions.revision_list],
    "comments": [item[1] for item in comments_and_revisions.revision_list],
    "revisions": [item[2] for item in comments_and_revisions.revision_list],
}


output_path = output_commented_document(
    input_doc_path,
    document_review_items,
    "reviewed_document.docx"
)

print(document_review_items)
