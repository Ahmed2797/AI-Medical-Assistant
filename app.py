from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import logging

# Import your RAG logic from main.py
from main import get_medical_answer 

# Setup basic logging to see what's happening in the terminal
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Medical AI Assistant API",
    description="A RAG-powered interface for medical document querying.",
    version="1.0.0"
)

# CORS configuration: Allows your frontend to talk to your backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic model for incoming JSON requests
class QueryRequest(BaseModel):
    query: str

# Mount static files (CSS/JS) and setup HTML templates
# Ensure these folders exist in your directory!
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# --- ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    """
    Serves the main chat interface (index.html).
    """
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/status")
async def get_status():
    """
    Health check to verify the API is running.
    """
    return {"status": "online", "system": "Medical RAG v1.0"}

@app.post("/ask")
async def ask_question(request: QueryRequest):
    """
    Receives a question, calls the RAG chain, and returns the medical answer.
    """
    try:
        # Log the incoming query for debugging
        logger.info(f"Received query: {request.query}")

        # 1. Call your function from main.py
        # It expects a string and returns a string (based on our previous step)
        result_string = get_medical_answer(request.query)
        
        # 2. Return the JSON response
        return {"answer": result_string}
        
    except Exception as e:
        # Detailed error logging in your terminal
        logger.error(f"Error processing RAG request: {str(e)}")
        # Generic error message for the user for security
        raise HTTPException(status_code=500, detail="An error occurred while processing the medical query.")

if __name__ == "__main__":
    # Start the server on port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)

## uvicorn app:app --reload
