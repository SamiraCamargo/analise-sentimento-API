from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

# Carregar modelo e vetorizador
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")


class ReviewRequest(BaseModel):
    review: str


@app.get("/")
def home():
    return {"mensagem": "API de análise de sentimento de reviews de produtos funcionando!"}


@app.post("/predict")
def predict_sentiment(request: ReviewRequest):
    review_text = request.review

    # Transformar o texto usando o mesmo vectorizer do treino
    review_vector = vectorizer.transform([review_text])

    # Fazer a predição
    prediction = model.predict(review_vector)[0]

    return {
        "review": review_text,
        "sentimento_previsto": prediction
    }