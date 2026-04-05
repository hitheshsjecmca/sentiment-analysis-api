from transformers import pipeline

# Load sentiment model
sentiment_pipeline = pipeline("sentiment-analysis")

# Try loading emotion model safely
try:
    emotion_pipeline = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )
except Exception as e:
    emotion_pipeline = None
    print("Emotion model failed to load:", e)


def analyze_sentiment(text: str):
    result = sentiment_pipeline(text)[0]
    return {
        "sentiment": result["label"],
        "confidence": result["score"]
    }


def analyze_emotion(text: str):
    results = emotion_pipeline(text)

    emotions = {}
    for item in results:
        emotions[item["label"]] = round(item["score"], 3)

    # Sort and take top 3  
    sorted_emotions = dict(
        sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:3]
    )

    return sorted_emotions