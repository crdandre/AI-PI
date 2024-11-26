"""
This file contains the code for an agent which, given relevant context
from prior reviews and any other guidelines for the review process, can
create suggestions for comments and revisions, simulating the style of feedback
a PI would given when reviewing a paper

For now, some manual token management for fitting within context length...
"""
import os
import dspy

class ReviewerAgent:
    def __init__(
        self,
        llm=dspy.LM(
            "nvidia_nim/meta/llama3-70b-instruct",
            api_key=os.environ['NVIDIA_API_KEY'],
            api_base=os.environ['NVIDIA_API_BASE']
        ),
    ):
        self.llm = llm
        dspy.configure(lm=llm)
        self.call_review = dspy.ChainOfThought("question -> answer")
        self.call_revise = dspy.ChainOfThought(
            "question -> revision_list: list[list[str]]"
        )