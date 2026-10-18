import json
import threading
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient

from stationwatch.app import create_app
from stationwatch.config import Settings
from stationwatch.features import Features
from stationwatch.gemini import GeminiReviewer
from stationwatch.storage import EventStore


def test_authentication_media_review_and_cross_origin_writes(tmp_path):
    store = EventStore(tmp_path)
    store.add({"id": "test", "captured_at": "2026-10-17T00:00:00Z", "segment": "test.mp4"}, b"jpeg")
    pipeline = Mock()
    pipeline.lock = threading.Lock()
    pipeline.status.return_value = {"running": False, "mode": "baseline"}
    with TestClient(create_app(pipeline, store, "a-long-private-dashboard-token")) as client:
        assert client.get("/").status_code == 200
        for path in [
            "/api/status",
            "/api/events",
            "/api/stream",
            "/api/events/test/snapshot",
            "/api/events/test/video",
        ]:
            assert client.get(path).status_code == 401
        assert client.post("/api/events/test/reviewed").status_code == 401
        assert client.post("/api/login", json={"token": "wrong"}).status_code == 401
        response = client.post("/api/login", json={"token": "a-long-private-dashboard-token"})
        assert response.status_code == 200
        assert "HttpOnly" in response.headers["set-cookie"]
        assert "SameSite=strict" in response.headers["set-cookie"]
        assert client.get("/api/status").status_code == 200
        assert client.get("/api/events/test/snapshot").content == b"jpeg"
        assert client.get("/api/events/test/video").status_code == 409
        assert (
            client.post("/api/events/test/reviewed", headers={"Origin": "https://other.example"}).status_code
            == 403
        )
        assert client.post("/api/events/test/reviewed").status_code == 200
        assert client.get("/api/events").json()[0]["reviewed"] is True
        assert client.get("/api/events/no-such-event/snapshot").status_code == 404
        assert client.post("/api/events/no-such-event/reviewed").status_code == 404
    pipeline.close.assert_called_once()


@pytest.mark.parametrize(
    "content",
    [
        {"decision": "uncertain", "confidence": 1.3, "reason": "Bad confidence"},
        {"decision": "inattentive", "confidence": 0.9},
        {"decision": "maybe", "confidence": 0.9, "reason": "Bad decision"},
    ],
)
def test_malformed_reviews_fail_closed(content):
    reviewer = GeminiReviewer(Settings(), "test-key")
    reviewer.client.close()
    reviewer.client = httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200,
                json={
                    "candidates": [
                        {"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(content)}]}}
                    ]
                },
            )
        )
    )
    try:
        with pytest.raises(RuntimeError, match="invalid or blocked"):
            reviewer.review(b"jpeg", Features(90, 120, 0.4, 0.5, 1, 0.8, 0.5, 3))
    finally:
        reviewer.close()


def test_gemini_request_contains_image_and_geometry_without_key_in_url():
    settings = Settings()
    reviewer = GeminiReviewer(settings, "test-key")
    reviewer.client.close()

    def respond(request):
        assert "test-key" not in str(request.url)
        assert request.headers["x-goog-api-key"] == "test-key"
        body = json.loads(request.content)
        assert body["contents"][0]["parts"][1]["inlineData"]["mimeType"] == "image/jpeg"
        assert "Machine center" in body["contents"][0]["parts"][0]["text"]
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "finishReason": "STOP",
                        "content": {
                            "parts": [
                                {
                                    "text": json.dumps(
                                        {"decision": "uncertain", "confidence": 0.3, "reason": "Occluded"}
                                    )
                                }
                            ]
                        },
                    }
                ]
            },
        )

    reviewer.client = httpx.Client(
        headers={"x-goog-api-key": "test-key"}, transport=httpx.MockTransport(respond)
    )
    try:
        assert reviewer.review(b"jpeg", Features(90, 120, 0.4, 0.5, 1, 0.8, 0.5, 3)).decision == "uncertain"
    finally:
        reviewer.close()
