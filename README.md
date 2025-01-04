# AI-PI
Can't pin your PI down to review a manuscript?
No problem! The AI-PI will take care of it!

Jokes aside, this is a tool to aid in the review process and connect more directly to word documents which seem to remain quite common in manuscript draft preparation.

My mom added a contribution in `moms_addition`. I'm not sure what it does, but it's evidently crucial to correct function of the framework. Remove at your own peril.

This design of this framework was inspired primarily by [STORM](https://github.com/stanford-oval/storm/tree/main/knowledge_storm) and [MMAPIS](https://github.com/fjiangAI/MMAPIS/tree/main?tab=readme-ov-file#how-to-run).

## Setup
1. Fill in `OPENROUTER_API_KEY` in `.env.example` and rename as `.env`
2. `python3 -m venv .venv && pip install -e .`
3. If installing the frontend, `cd frontend`, `npm install` or install with your package manager of choice, `npm run dev`

## Usage
### CLI
`python -m src.ai_pi.workflow -i path/to/input_file.docx -o path/to/output_folder`

### GUI
(from the repo root directory) 
- Start API : `uvicorn src.ai_pi.main:app --reload --host localhost --port 8001`
- Start Frontend: `cd frontend && npm run dev`







