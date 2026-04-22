from fastapi import FastAPI
from app.schemas import TextRequest
from app.model import analyze_sentiment, analyze_emotion

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Sentiments & Emotion API 🚀"}

@app.post("/analyze")
def analyze(request: TextRequest):
    sentiment = analyze_sentiment(request.text)
    emotions = analyze_emotion(request.text)

    return {
        "text": request.text,
        "sentiment": sentiment,
        "emotions": emotions
    }