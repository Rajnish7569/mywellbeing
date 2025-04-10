from fastapi import FastAPI
from pydantic import BaseModel
from emotion_analysis import analyze_emotion
from mental_health_bot import get_mental_health_reply

app = FastAPI(title="AI Mental Health Assistant")

class TextInput(BaseModel):
    text: str

@app.post("/analyze-emotion", tags=["AI Assistant"])
def emotion_endpoint(input_text: TextInput):
    result = analyze_emotion(input_text.text)
    return {"emotion_scores": result}

@app.post("/ai/recommendation", tags=["AI Assistant"])
def recommendation_endpoint(input_text: TextInput):
    result = analyze_emotion(input_text.text)
    top_emotion = sorted(result[0], key=lambda x: x['score'], reverse=True)[0]['label']
    recommendation = get_gemini_advice(top_emotion)  # Using Gemini
    return {
        "emotion": top_emotion,
        "recommendation": recommendation
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
