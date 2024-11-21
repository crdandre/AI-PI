"""
This file contains the code for an agent which, given relevant context
from prior reviews and any other guidelines for the review process, can
create suggestions for comments and revisions, simulating the style of feedback
a PI would given when reviewing a paper

For now, some manual token management for fitting within context length...
"""
from llama_index.llms.nvidia import NVIDIA
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from typing import Optional
import tiktoken

def truncate_text(text: str, token_limit: int) -> str:
    """Truncate text to stay within token limit, with safety margin"""
    # Initialize tokenizer (using same one as GPT-3.5/4)
    encoding = tiktoken.get_encoding("cl100k_base")
    
    # Add safety margin for response tokens (roughly 1000-1500 tokens)
    safe_limit = max(token_limit - 1500, 100)  # Ensure we don't go negative
    
    # Get tokens
    tokens = encoding.encode(text)
    
    # If we're under the safe limit, return original text
    if len(tokens) <= safe_limit:
        return text
        
    # Otherwise truncate tokens and decode
    truncated_tokens = tokens[:safe_limit]
    return encoding.decode(truncated_tokens)

class ReviewerAgent:
    def __init__(self, 
            llm=NVIDIA(model="meta/llama3-70b-instruct"),
            max_iterations=20,
            verbose=False,
            max_tokens=6000  # Total token limit including response
        ):
        self.llm = llm
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.max_tokens = max_tokens
        
        self._setup_review_agent()
        self._setup_revision_agent()
        
    def _reflect(self, text: str) -> str:
        """Simple function to return the input text"""
        return text
        
    def _setup_review_agent(self):
        """Setup agent for generating comprehensive reviews"""
        review_system_prompt = """You are an academic reviewer analyzing documents. 
        When analyzing a document, reflect deeply on its content and provide your analysis directly.
        
        Follow these steps in your mind:
        1. Analyze the content systematically
        2. Compare against patterns from prior reviews
        3. Generate specific, actionable feedback
        
        When you want to express your thoughts, use the reflect tool with a single string argument 
        containing your analysis."""

        reflect_tool = FunctionTool.from_defaults(
            fn=self._reflect,
            name="reflect",
            description="Express your analysis as a single text string"
        )

        self.review_agent = ReActAgent.from_tools(
            tools=[reflect_tool],
            system_prompt=review_system_prompt,
            llm=self.llm,
            max_iterations=self.max_iterations,
            verbose=self.verbose
        )
        
    def _setup_revision_agent(self):
        """Setup agent for generating specific revision suggestions"""
        revision_system_prompt = """You are an academic editor suggesting specific revisions.
        When analyzing a document, reflect deeply on potential improvements and provide your suggestions directly.
        
        Follow these steps in your mind:
        1. Identify areas needing improvement
        2. Consider multiple revision options
        3. Suggest the most effective changes
        
        When you want to express your thoughts, use the reflect tool with a single string argument 
        containing your suggestions."""

        reflect_tool = FunctionTool.from_defaults(
            fn=self._reflect,
            name="reflect",
            description="Express your suggestions as a single text string"
        )

        self.revision_agent = ReActAgent.from_tools(
            tools=[reflect_tool],
            system_prompt=revision_system_prompt,
            llm=self.llm,
            max_iterations=self.max_iterations,
            verbose=self.verbose
        )
        
    def generate_review(
        self,
        prompt: str,
        context: Optional[str] = None
    ) -> str:
        """Generate a review, truncating input if needed"""
        # Combine prompt and context
        if context:
            # Truncate context first if needed
            context_limit = self.max_tokens // 2  # Use half tokens for context
            truncated_context = truncate_text(context, context_limit)
            combined_text = f"{prompt}\n\nContext: {truncated_context}"
        else:
            combined_text = prompt
            
        # Final truncation to ensure we stay within limits
        truncated_text = truncate_text(
            text=combined_text, 
            token_limit=self.max_tokens
        )
        
        return self.review_agent.chat(truncated_text)
    
    def generate_comments_and_revisions(
        self,
        prompt,
    ):
        return self.revision_agent.chat(prompt)
    

        
    