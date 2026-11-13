from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_ingest_and_ask_roundtrip():
    r = client.post("/ingest", json={
        "text": "Velto Studios starts every engagement with a paid two-week pilot.",
        "source": "test.md",
    })
    assert r.status_code == 200
    assert r.json()["chunks"] >= 1

    r = client.post("/ask", json={"question": "how does an engagement start?"})
    assert r.status_code == 200
    body = r.json()
    assert "pilot" in body["answer"]
    assert body["citations"]
    assert body["citations"][0]["source"] == "test.md"


def test_ask_empty_index_still_answers():
    r = client.post("/ask", json={"question": "something with no matches zzzqqq"})
    assert r.status_code == 200
    assert "answer" in r.json()


def test_ingest_rejects_empty():
    r = client.post("/ingest", json={"text": "   ", "source": "x"})
    assert r.status_code == 422
