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
    print("DEBUG: Emotion function called")

    if emotion_pipeline is None:
        return {"error": "Emotion model not available"}

    try:
        results = emotion_pipeline(text)
        print("DEBUG RESULTS:", results)

        emotion_dict = {}

        # Case 1: multiple emotions (list of dicts)
        if isinstance(results, list) and isinstance(results[0], dict):
            for item in results:
                emotion_dict[item["label"]] = item["score"]

        # Case 2: nested list (rare case)
        elif isinstance(results, list) and isinstance(results[0], list):
            for item in results[0]:
                emotion_dict[item["label"]] = item["score"]

        else:
            return {"error": "Unexpected format"}

        return emotion_dict

    except Exception as e:
        return {"error": str(e)}