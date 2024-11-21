# AI-PI
Can't pin your PI down to review a manuscript?
No problem! The AI-PI will take care of it!

Jokes aside this is a tool to streamline the review process. Using prior materials from your PI's review process, you can more likely take care of the things they'd always find and comment on, freeing up cognitive load to focus on the less-obvious review points.

## Setup

### API Access
- OpenAI:
- NVIDIA:
    - Create a cloud account name, i.e. "christians-nvidia-cloud"
    - Generate an API Key and store in `.env` as `NVIDIA_API_KEY`

## Approach
1. LlamaIndex-based Agentic RAG
    - ReActAgent has inbuilt implementation in LlamaIndex, so starting with this
    - At least one data collection to query (Prior reviewed docs as context)
        - Many have separate documents, i.e. MyPaper_v{1-n}, it'd be nice to combine these into one coherent revision history
        - Relating revisions to comments would be even better!
        - Adding collections (indices) will change the tool format, etc. - take note of this.
    - Could add a more general database of review pointers for academic papers based on field, or a broader database of reviewed documents with comments to better understand comment/response structure (sometimes comments can be sparse/spread out)
    - {SOON} Should have autonomous and interactive modes to guide the agent like STORM
2. Workflow
    - Ingestion
    - Context
    - Review
        - Must know *where* to place the comments rather than just supply general comments, and apply both a comment and a suggested revision to this particular text
        - This may necessitate working with input context other than the whole document to be reviewed, especially if llama compatibility is desired or any other 8192 token context model. 
            - What approaches allow for this? Should this document be embedded as well? I want to separate this from the prior reviews used as context.
    - Output
        - [how to have a ReAct agent output intermediate steps (which have the revision comments in them)](https://github.com/run-llama/llama_index/issues/15952)
2. Frontend
    - Chat Interface
    - Allows upload of .docx to embed as "PI Upload"
    - Cyberpunk Einstein Feel
    - Outputs .docx with comments and suggested revisions which the user can open in Word and approve as if they were emailed a reviewed document by a person


## Improvement Notes
### Give this a try to optimize the RAG parameters and data storage types?
- https://github.com/KruxAI/ragbuilder

### Use DSPy/TextGrad to optimize prompts where applicable?
    - Not sure yet how this may conflict with agents

### Full local use option, dockerized option, etc.

### Other Agent-Calling Methods
- [FunctionCallingAgent vs ReActAgent within LlamaIndex](https://github.com/run-llama/llama_index/issues/15685)
- DSPy [example](https://medium.com/@leighphil4/dspy-rag-with-llamaindex-programming-llms-over-prompting-1b12d12cbc43)
- Other frameworks
- Different problem-solving methods such as tree-search and topics discussed [here](https://www.youtube.com/watch?v=MXPYbjjyHXc)
- [Agents From Scratch](https://learnbybuilding.ai/tutorials/dspy-agents-from-scratch)