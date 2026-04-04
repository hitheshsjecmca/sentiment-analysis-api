from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

# Request body structure
class TextRequest(BaseModel):
    text: str

# Load model once
sentiment_pipeline = pipeline("sentiment-analysis")

@app.get("/")
def home():
    return {"message": "Sentiment API running 🚀"}

@app.post("/analyze")
def analyze(request: TextRequest):
    result = sentiment_pipeline(request.text)[0]
    return {
        "text": request.text,
        "sentiment": result["label"],
        "confidence": result["score"]
    }