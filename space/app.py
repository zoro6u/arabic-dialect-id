import re
import gradio as gr
import spaces
from transformers import pipeline

MODEL_ID = "zoro6u/marbert-arabic-dialect-id"
clf = pipeline("text-classification", model=MODEL_ID, top_k=18)

NAMES = {
    "OM": "Oman", "SD": "Sudan", "SA": "Saudi Arabia", "KW": "Kuwait", "QA": "Qatar",
    "LB": "Lebanon", "JO": "Jordan", "SY": "Syria", "IQ": "Iraq", "MA": "Morocco",
    "EG": "Egypt", "PL": "Palestine", "YE": "Yemen", "BH": "Bahrain", "DZ": "Algeria",
    "AE": "UAE", "TN": "Tunisia", "LY": "Libya",
}

def clean(text):
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"#", " ", text)
    text = re.sub(r"[\u064B-\u0652]", "", text)
    return re.sub(r"\s+", " ", text).strip()

@spaces.GPU
def predict(text):
    if not text.strip():
        return {}
    results = clf(clean(text))[0]
    return {f"{NAMES[r['label']]} ({r['label']})": r["score"] for r in results}

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=3, label="Arabic text", rtl=True,
                      placeholder="اكتب جملة بأي لهجة عربية"),
    outputs=gr.Label(num_top_classes=5, label="Dialect"),
    title="Arabic Dialect Identification",
    description="Fine-tuned MARBERTv2 on QADI (18 countries). Test macro-F1: 0.614. "
                "Model: zoro6u/marbert-arabic-dialect-id",
    examples=[
        ["شنو الاخبار يا زول كيفك"],
        ["ايه الاخبار يا معلم عامل ايه"],
        ["وش السالفة يا رجال"],
        ["شو الاخبار كيفك"],
        ["واش راك لاباس"],
    ],
)

demo.launch()