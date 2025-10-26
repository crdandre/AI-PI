# AI-PI
[Demo](https://www.youtube.com/watch?v=ArPSZDJEqUg)

<img height="300" alt="image" src="https://github.com/user-attachments/assets/170eb32f-3857-4b67-87a0-3e9ac98b305c" />

Can't pin your PI down to review a manuscript?
No problem! The AI-PI will take care of it!

Jokes aside, this is a tool to aid in the review process and connect more directly to word documents which seem to remain quite common in manuscript draft preparation.

You input a .docx of your abstract/manuscript, and this framework outputs a .docx with comments that ideally contain helpful feedback. I realize the current models utilized are outdated, will get around to updating that sooner or later, as I plan to eventually reinforce this with a collection of actual manuscript reviews that I've collected as well. You're welcome to try the same if you get to it before me!

My mom added a contribution in `moms_addition`. I'm not sure what it does, but it's evidently crucial to correct function of the framework. Remove at your own peril.

This design of this framework was inspired primarily by [STORM](https://github.com/stanford-oval/storm/tree/main/knowledge_storm) and [MMAPIS](https://github.com/fjiangAI/MMAPIS/tree/main?tab=readme-ov-file#how-to-run).

## Setup
1. Fill in `OPENROUTER_API_KEY` in `.env.example` and rename as `.env`
2. `python3 -m venv .venv && pip install -e .`
3. If installing the frontend, `cd frontend`, `npm install` or install with your package manager of choice, `npm run dev`

apt-get installs:
1. pandoc
2. texlive-xetex

## Usage
### CLI
`python -m src.ai_pi.workflow -i path/to/input_file.docx -o path/to/output_folder`

### GUI
(from the repo root directory) 
- Start API : `uvicorn src.ai_pi.main:app --reload --host localhost --port 8001`
- Start Frontend: `cd frontend && npm run dev`






