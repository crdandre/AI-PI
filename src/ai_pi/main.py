from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from ai_pi.workflow2 import PaperReview
from llama_index.llms.openai import OpenAI
import dspy
import os
from typing import Dict
import uuid
from fastapi import HTTPException

app = FastAPI()

# Add CORS middleware before any routes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Add a dictionary to store uploaded files temporarily
uploaded_files = {}

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)) -> Dict[str, str]:
    try:
        # Create temp directory if it doesn't exist
        os.makedirs("temp", exist_ok=True)
        
        file_id = str(uuid.uuid4())
        input_path = f"temp/{file_id}_{file.filename}"
        
        # Read the file content
        content = await file.read()
        
        # Write to temp file
        with open(input_path, "wb") as buffer:
            buffer.write(content)
        
        # Store file info
        uploaded_files[file_id] = {
            "input_path": input_path,
            "original_filename": file.filename,
            "status": "uploaded"
        }
        
        return {
            "fileId": file_id,
            "filename": file.filename,
            "status": "uploaded",
            "filePath": input_path
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/process")
async def process_document(request: Dict[str, str]):
    file_id = request.get("fileId")
    model = request.get("model", "gpt-4o-mini")
    
    if not file_id or file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_info = uploaded_files[file_id]
    input_path = file_info["input_path"]
    original_filename = file_info["original_filename"]
    output_path = f"temp/reviewed_{file_id}_{original_filename}"
    
    try:
        # Initialize LLMs with the provided model
        llm = OpenAI(model=model)
        lm = dspy.LM(
            model,
            api_base="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            temperature=1.0,
            max_tokens=9999
        )
        
        # Process document
        paper_review = PaperReview(llm=llm, lm=lm, verbose=True, reviewer_class="Predict")
        paper_review.review_paper(input_path, output_path)
        
        # Instead of returning the file, store its path and return the ID
        uploaded_files[file_id]["output_path"] = output_path
        uploaded_files[file_id]["status"] = "processed"
        
        return {
            "fileId": file_id,
            "status": "processed",
            "filename": f"reviewed_{original_filename}"
        }
    except Exception as e:
        uploaded_files[file_id]["status"] = "error"
        uploaded_files[file_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents/{file_id}")
async def get_document(file_id: str):
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    file_info = uploaded_files[file_id]
    
    # Always return the processed file if it exists
    if file_info["status"] == "processed":
        file_path = file_info["output_path"]
    else:
        raise HTTPException(status_code=400, detail="File has not been processed yet")
    
    # Verify file exists
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Processed file not found on server")
    
    return FileResponse(
        file_path,
        media_type='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        filename=f"reviewed_{file_id}_{file_info['original_filename']}"
    )

# Add a status endpoint
@app.get("/api/documents/{file_id}/status")
async def get_document_status(file_id: str):
    if file_id not in uploaded_files:
        raise HTTPException(status_code=404, detail="File not found")
    
    return {
        "status": uploaded_files[file_id]["status"],
        "error": uploaded_files[file_id].get("error")
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/")
async def root():
    return {"message": "Welcome to AI-Pi API. Visit /docs for API documentation."}

"""
maybe result should be the raw .docx? idk how to address this yet in document_output
maybe downloading should be a separate call, and this should only produce a site-viewable doc (not a docx necessarily)

that would mean making the document_output call differently, and rendering the word doc on the website? are there better ways to do this?
"""