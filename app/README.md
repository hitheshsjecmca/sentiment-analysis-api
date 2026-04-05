# Sentiment & Emotion Analysis API

A REST API built using FastAPI and Hugging Face Transformers to analyze text sentiment and emotions.

## Features
- Sentiment Analysis (Positive / Negative)
- Emotion Detection (Top 3 emotions)
- REST API with FastAPI
- Swagger UI for testing

## Tech Stack
- Python
- FastAPI
- Uvicorn
- Hugging Face Transformers

## Run Locally
pip install -r requirements.txt
python -m uvicorn app.main:app --reload

## API Endpoint
POST /analyze

Example Request:
{
  "text": "I love coding"
}