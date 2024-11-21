from dotenv import load_dotenv
load_dotenv()
from llama_index.llms.nvidia import NVIDIA
from llama_index.core import Settings
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import ReActAgent

from ai_pi.embed_documents import ContextStorageInterface

Settings.llm = NVIDIA(model="meta/llama3-70b-instruct")

context_interface = ContextStorageInterface(
    store_embeddings=False
)

#if top k is less than 10, it seems not to find comments? that must mean that
#document body vectors are more similar in terms of wording
# query_engine = context_interface.index.as_query_engine(similarity_top_k=10)

# question = "Can you describe details about the comments left about this document?"
# query_response = query_engine.query(
#     question
# )
# retriever_response = context_interface.retrieve(question)
# print(retriever_response[:100])

# #Chat
# query_engine = context_interface.index.as_chat_engine(similarity_top_k=10)

# import warnings
# warnings.simplefilter("ignore")
# print("Your bot is ready to talk! Type your messages here or send 'stop'")
# while True:
#     query = input("<|user|>\n")
#     if query == 'stop':
#         break
#     response = query_engine.chat(query)
#     print("\n", response, "\n")

summary_query_engine = context_interface.summary_index.as_query_engine(similarity_top_k=10)
vector_query_engine = context_interface.vector_index.as_query_engine(similarity_top_k=10)

summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    description=(
        "Useful for summarization questions related to the general messaging or take-home messages from comments in the embedded papers"
    ),
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    description=(
        "Useful for retrieving specific context from the embedded papers and related comments."
    ),
)

llm = NVIDIA(model="meta/llama3-70b-instruct")
agent = ReActAgent.from_tools([summary_tool, vector_tool], llm=llm, verbose=True)

response = agent.chat("What comments are left on this document, and can you explain them? I.e. author, content, etc.")