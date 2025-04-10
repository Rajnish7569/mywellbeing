from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="AI Mental Health Assistant")

# Load Emotion Classification Model
emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

# Input Schema
class TextInput(BaseModel):
    text: str

@app.post("/ai/analyze-recommendation", tags=["AI Assistant"], summary="Analyze Emotion & Get Recommendation")
def analyze_and_recommend(input_text: TextInput):
    result = emotion_classifier(input_text.text)
    top_emotion = sorted(result[0], key=lambda x: x['score'], reverse=True)[0]['label']

    coaching = {
        "joy": "Keep doing what makes you happy! Maybe share your joy with someone else today.",
        "sadness": "It's okay to feel sad. Try writing down your feelings or talking to someone you trust.",
        "anger": "Take a deep breath. A short walk or some mindfulness can help you calm down.",
        "fear": "Try to identify what's making you feel fearful and challenge the thought gently.",
        "love": "Nurture the relationships that make you feel loved. Send a kind message to someone.",
        "surprise": "Embrace the unexpected! Write down your experience and how it made you feel.",
    }

    recommendation = coaching.get(top_emotion, "Try to reflect on your emotions and take care of your mental space.")

    return {
        "emotion_scores": result,
        "top_emotion": top_emotion,
        "recommendation": recommendation
    }

# Hide the old routes so they don't show separately in Swagger
@app.post("/analyze-emotion", include_in_schema=False)
def analyze_emotion(input_text: TextInput):
    return analyze_and_recommend(input_text)

@app.post("/ai/recommendation", include_in_schema=False)
def get_recommendation(input_text: TextInput):
    return analyze_and_recommend(input_text)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
