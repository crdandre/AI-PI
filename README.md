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


## Scratchpad
1. First get the structure, themes, overall feel of the paper
2. Given this, scrutinize each section wrt a set of requirements that involve detected PI preferences, best writing practices, guidelines from a journal, etc.
3. Use any suggestions for improvement generated from the review phase to generate actionable changes to the document with explanations for the changes made
4. Produce an output .docx with these comments and suggested revisions, able to be opened in Word and parsed through for addressing each comment individually, as well as one-click approved just like it'd be if a human reviewer suggested the changes

The idea is that this not only saves the PI time, but also allows for a more personalized review process that can be tailored to the PI's preferences and the specific needs of the manuscript - it's low friction and not adversarial to the author and the PI, it seeks to extend them both.

### Ingestion
-Embed raw chunks
- Use LLM to detect and summarize sections of the paper, and build a knowledge tree where parent summaries can be used to navigate to the relevant raw text of the paper (i.e. not necessarily the raw chunks, but rather the text detected by the LLM as bein ga "section" of the paper or subsection within a main section - variable tree depth is OK)
- Embed both the raw chunks and this summary tree

^This can be done independently or as part of the new document processing (user can supply their own context either before or during processing of a current document).

### New Document Processing
- Parse new doc
- Generate hierarchical summary of the new doc (or for MVP just a simple summary)
- Embed the new doc
- Create a context-friendly summary that can be used to scrutinize things like the coherence of a section relative to the whole paper
- Give a reviewer agent this summary and the ability to retrieve relevant raw text (using both the summary tree and the raw chunks)
- This generates a list of comments and suggested revisions
- Generate structured data: match strings, comments, suggested changes (i.e. removals and additions)
- Output .docx with comments and suggested revisions


## Learned Concepts
1. Knowledge storage formats
    - Flat
    - Tree
    - Graph
2. information storage
    - Embeddings stored in vector db (flat, so far - others may be possible)
    - Large chunk to small chunk tree (llamaindex HierarchicalNodeParser)
    - Summary to raw text tree (unsure of existing implementation)

### MVP Progression
- 11/29: All code working, but feedback both isn't being output correctly (interferes with document text) and the feedback itself is not of a high enough quality to be useful.
    - Next step: what is missing? Overall feel of the paper? Narrative/structure level feedback given as a report? It's giving all sentence/clarity feedback at the moment, but not addressing the overall quality of the science/message/narrative/methods, etc.
    - Should be able to tune a "threshold" for how much feedback to give (i.e. how perfectionist the reviewer is, to keep the focus on the highest prio points)
- 12/2: Review quality is improving with CoT configs, considering using o1 as well. Working on doc formatting. Need to expand review process to hit style, clarity, and decisions about whether to comment, revise, both, or none for a given review point (but still comment and revise sufficiently - tune this propensity)
    - add [MegaParse](https://github.com/quivrhq/megaparse) ?
    - need CoT/ReAct to actually show reasoning steps
        - use Storm's role+dialogue based approach and dspy implementations to create depth of reasoning
    - need to assess whether o1 works well with the arbitrary sections of the paper and whether this helps or hinders proper feedback? (or if using non-o1 is better here)
    - review sample database: [scrape openreview.net?](https://openreview.net/forum?id=2z4U9reLm9&referrer=%5Bthe%20profile%20of%20Subhashini%20Venugopalan%5D(%2Fprofile%3Fid%3D~Subhashini_Venugopalan2))