from dotenv import load_dotenv
load_dotenv()
from llama_index.llms.nvidia import NVIDIA
from llama_index.core.tools import QueryEngineTool
from llama_index.core.agent import AgentRunner, ReActAgent, ReActAgentWorker

from ai_pi.embed_documents import ContextStorageInterface

max_iterations = 20
similarity_top_k = 10

context_interface = ContextStorageInterface(
    store_embeddings=False
)

#if top k is less than 10, it seems not to find comments? that must mean that
summary_query_engine = context_interface.summary_index.as_query_engine(similarity_top_k=similarity_top_k)
vector_query_engine = context_interface.vector_index.as_query_engine(similarity_top_k=similarity_top_k)

summary_tool = QueryEngineTool.from_defaults(
    query_engine=summary_query_engine,
    name="summary_tool",
    description=(
        "Useful for summarization questions related to the general messaging or take-home messages from comments in the embedded papers"
    ),
)

vector_tool = QueryEngineTool.from_defaults(
    query_engine=vector_query_engine,
    name="vector_tool",
    description=(
        "Useful for retrieving specific context from the embedded papers and related comments."
    ),
)

llm = NVIDIA(model="meta/llama3-70b-instruct")
agent = ReActAgent.from_tools(
    tools=[summary_tool, vector_tool],
    llm=llm,
    max_iterations=max_iterations,
    verbose=True
)

task_prompt = "Can you summarize the theme addressed by the embedded documents, and give a short general overview?"
agent.chat(task_prompt)


"""
TODO: Figure out interactivity with task-wise agent execution
"""
# # Interactive Agent
# react_agent_step_engine = ReActAgentWorker.from_tools(
#     tools=[summary_tool, vector_tool],
#     llm=llm,
#     max_iterations=max_iterations,
#     verbose=True
# )
# react_interactive_agent = AgentRunner(react_agent_step_engine)

# #create task
# task = react_interactive_agent.create_task(task_prompt)
# print(task)

# #execute step
# step_output = react_interactive_agent.run_step(task.task_id)

# # if step_output is done, finalize response
# if step_output.is_last:
#     response = agent.finalize_response(task.task_id)

# # list tasks
# task.list_tasks()

# # get completed steps
# task.get_completed_steps(task.task_id)

# print(str(response))