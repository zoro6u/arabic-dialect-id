import re
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_ID = "zoro6u/marbert-arabic-dialect-id"

app = FastAPI(title="Arabic Dialect Identification API", version="0.1.0")

def clean(text: str) -> str:
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[\u064B-\u0652]", "", text)
    return re.sub(r"\s+", " ", text).strip()

@app.on_event("startup")
def load_model():
    app.state.tok = AutoTokenizer.from_pretrained(MODEL_ID)
    app.state.model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()
    app.state.labels = app.state.model.config.id2label

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=1000)

class Prediction(BaseModel):
    dialect: str
    confidence: float
    top3: list[dict]

@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID}

@app.post("/predict", response_model=Prediction)
def predict(req: PredictRequest):
    text = clean(req.text)
    inputs = app.state.tok(text, return_tensors="pt", truncation=True, max_length=64)
    with torch.no_grad():
        probs = torch.softmax(app.state.model(**inputs).logits, dim=-1)[0]
    top = torch.topk(probs, 3)
    top3 = [{"dialect": app.state.labels[i.item()], "confidence": round(p.item(), 4)}
            for p, i in zip(top.values, top.indices)]
    return Prediction(dialect=top3[0]["dialect"], confidence=top3[0]["confidence"], top3=top3)