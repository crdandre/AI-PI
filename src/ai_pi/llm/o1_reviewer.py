"""
2nd approach to reviewing that doesn't require internal CoT setup

This module provides an alternative approach to scientific paper reviewing that maintains
compatibility with the original reviewer.py output format.

Input/Output Format Description:

Section Review:
- Input:
  - section_text: Full text content of the section being reviewed
  - section_type: Type of section (e.g., Methods, Results, Discussion)
  - context: Additional paper context and metadata

Output of form:
return {
    'overall_assessment': str(final_review.overall_assessment),
    'key_strengths': [str(s) for s in (final_review.key_strengths if isinstance(final_review.key_strengths, list) else [final_review.key_strengths])],
    'key_weaknesses': [str(w) for w in (final_review.key_weaknesses if isinstance(final_review.key_weaknesses, list) else [final_review.key_weaknesses])],
    'recommendations': [str(r) for r in (final_review.recommendations if isinstance(final_review.recommendations, list) else [final_review.recommendations])],
    'communication_review': {
        'writing_assessment': str(comm_review.writing_assessment),
        'narrative_strengths': [str(s) for s in (comm_review.narrative_strengths if isinstance(comm_review.narrative_strengths, list) else [comm_review.narrative_strengths])],
        'narrative_weaknesses': [str(w) for w in (comm_review.narrative_weaknesses if isinstance(comm_review.narrative_weaknesses, list) else [comm_review.narrative_weaknesses])],
        'style_recommendations': [str(r) for r in (comm_review.style_recommendations if isinstance(comm_review.style_recommendations, list) else [comm_review.style_recommendations])]
    }
}
"""

prompt = ""
