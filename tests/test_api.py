import pytest
from fastapi.testclient import TestClient
from api.main import app

@pytest.fixture(scope="module")
def client():
    # الـ with بيشغّل الـ startup (تحميل الموديل) مرة واحدة لكل الاختبارات
    with TestClient(app) as c:
        yield c

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"

def test_predict_sudanese(client):
    r = client.post("/predict", json={"text": "شنو الاخبار يا زول كيفك"})
    assert r.status_code == 200
    body = r.json()
    assert body["dialect"] == "SD"
    assert body["confidence"] > 0.5
    assert len(body["top3"]) == 3

def test_predict_egyptian(client):
    r = client.post("/predict", json={"text": "ايه الاخبار يا معلم عامل ايه"})
    assert r.json()["dialect"] == "EG"

def test_empty_text_rejected(client):
    r = client.post("/predict", json={"text": ""})
    assert r.status_code == 422

def test_mentions_are_cleaned(client):
    # نفس الجملة مع منشن وهاشتاق لازم تدّي نفس النتيجة
    a = client.post("/predict", json={"text": "شنو الاخبار يا زول كيفك"}).json()
    b = client.post("/predict", json={"text": "@someone شنو الاخبار يا زول كيفك #السودان"}).json()
    assert a["dialect"] == b["dialect"]