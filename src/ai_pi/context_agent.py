from dotenv import load_dotenv
load_dotenv()
from llama_index.llms.nvidia import NVIDIA
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import AgentRunner, ReActAgent, ReActAgentWorker

from ai_pi.embed_documents import ContextStorageInterface

import os
import dspy

class ContextAgent:
    def __init__(self,
            llm=NVIDIA(model="meta/llama3-70b-instruct"),
            max_iterations=20,
            similarity_top_k=10,
            store_embeddings=False,
            verbose=False,
        ):
        self.llm = llm
        self.max_iterations = max_iterations
        self.similarity_top_k = similarity_top_k
        self.verbose=verbose
        
        # Initialize context interface and query engines
        self.context_interface = ContextStorageInterface(store_embeddings=store_embeddings)
        self.summary_query_engine = self.context_interface.summary_index.as_query_engine(
            similarity_top_k=self.similarity_top_k
        )
        self.vector_query_engine = self.context_interface.vector_index.as_query_engine(
            similarity_top_k=self.similarity_top_k
        )
        
        # Initialize tools and agent
        self._setup_tools()
        self._setup_agent()
    
    def _setup_tools(self):
        self.summary_tool = QueryEngineTool.from_defaults(
            query_engine=self.summary_query_engine,
            name="summary_tool",
            description=(
                "Useful for summarization questions related to the general messaging or take-home messages from comments in the embedded papers"
            ),
        )
        
        self.vector_tool = QueryEngineTool.from_defaults(
            query_engine=self.vector_query_engine,
            name="vector_tool",
            description=(
                "Useful for retrieving specific context from the embedded papers and related comments."
            ),
        )
    
    def _setup_agent(self):
        self.agent = ReActAgent.from_tools(
            tools=[self.summary_tool, self.vector_tool],
            llm=self.llm,
            max_iterations=self.max_iterations,
            verbose=self.verbose
        )
    
    def query(self, prompt):
        """
        Query the context using the agent with any prompt
        
        Args:
            prompt (str): The question or prompt to ask about the embedded documents
            
        Returns:
            Response from the agent's chat
        """
        return self.agent.chat(prompt)
    
    
    
class DSPyContextAgent:
    """
    Same as above, but with DSPy
    Need to figure out how to set things like memory, max iters, etc
    """
    def __init__(self,
            llm=dspy.LM(
                "nvidia_nim/meta/llama3-70b-instruct",
                api_key=os.environ['NVIDIA_API_KEY'],
                api_base=os.environ['NVIDIA_API_BASE']
            ),
            similarity_top_k=10,
            store_embeddings=False,
        ):
        self.llm = llm
        self.similarity_top_k = similarity_top_k
        
        # Initialize context interface and query engines
        self.context_interface = ContextStorageInterface(store_embeddings=store_embeddings)
        self.summary_query_engine = self.context_interface.summary_index.as_query_engine(
            similarity_top_k=self.similarity_top_k
        )
        self.vector_query_engine = self.context_interface.vector_index.as_query_engine(
            similarity_top_k=self.similarity_top_k
        )
        
        self.tools = []
        
        # Initialize tools and agent
        dspy.configure(lm=llm)
        self._setup_tools()
        self._setup_agent()
    
    def _setup_tools(self):
        self.summary_tool = QueryEngineTool.from_defaults(
            query_engine=self.summary_query_engine,
            name="summary_tool",
            description=(
                "Useful for summarization questions related to the general messaging or take-home messages from comments in the embedded papers"
            ),
        )
        
        self.vector_tool = QueryEngineTool.from_defaults(
            query_engine=self.vector_query_engine,
            name="vector_tool",
            description=(
                "Useful for retrieving specific context from the embedded papers and related comments."
            ),
        )
        
        self.tools = [self.summary_tool, self.vector_tool]
            
    def _setup_agent(self):
        self.agent = dspy.ReAct("question -> answer", tools=self.tools)
        
    def query(self, prompt):
        """
        Query the context using the agent with any prompt
        
        Args:
            prompt (str): The question or prompt to ask about the embedded documents
            
        Returns:
            Response from the agent's chat
        """
        return {
            "response": self.agent(question=prompt),
            "history": self.llm.inspect_history(n=3)
        }