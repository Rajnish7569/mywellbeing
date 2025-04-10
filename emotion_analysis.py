from transformers import pipeline

emotion_classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    return_all_scores=True
)

def analyze_emotion(text: str):
    return emotion_classifier(text)

def get_recommendation(text: str):
    result = analyze_emotion(text)
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
