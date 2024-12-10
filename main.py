from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List
import torch
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, SearchRequest, ScoredPoint
from featurize import featurize_text  # Import featurization function

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
model = torch.load('model.pth')
model.eval()  # Set to evaluation mode

# Qdrant Configuration
QDRANT_HOST = "https://fc94b6ab-f5e6-4b45-8e7c-51ed48367a37.us-east4-0.gcp.cloud.qdrant.io:6333"
QDRANT_API_KEY = "j31ExXzFJ5xD77C1OFwnx21Aa5gFUSpSEQhCO4bHxIaGfSd6TNXRVQ"
qdrant_client = QdrantClient(url=QDRANT_HOST, api_key=QDRANT_API_KEY)
QDRANT_COLLECTION_NAME = "star_charts" 

class QuestionRequest(BaseModel):
    question: str

@app.get("/favicon.ico")
async def favicon():
    return RedirectResponse(url="/static/favicon.ico")

@app.get("/")
def root():
    return {"message": "Welcome to the ROS2 RAG system!"}

@app.post("/ask")
def ask_question(request: QuestionRequest):
    """
    Endpoint to answer a question based on the most relevant context.
    """
    try:
        # Convert question to vector using featurize_text
        question_vector = featurize_text(request.question)
        
        # Search for relevant context in Qdrant
        search_results: List[ScoredPoint] = qdrant_client.search(
            collection_name=QDRANT_COLLECTION_NAME,
            query_vector=question_vector.tolist(),
            limit=1
        )

        if not search_results:
            raise HTTPException(status_code=404, detail="No relevant context found for the question.")

        # Extract the most relevant context
        relevant_context = search_results[0].payload.get("content", "")

        # Generate an answer using the loaded model
        with torch.no_grad():
            # Prepare inputs as expected by your model
            model_inputs = {
                "question": request.question,
                "context": relevant_context
            }
            
            # Get the model's prediction
            answer = model(**model_inputs)
            
        return {
            "question": request.question,
            "relevant_context": relevant_context,
            "answer": answer
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate an answer: {str(e)}")

# Start the server with: uvicorn main:app --reload