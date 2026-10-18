import json
from email.parser import BytesParser
from email.policy import default

import cv2
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from worker_webpage import MAX_UPLOAD_BYTES, create_app, prepare_image


def photo(width=640, height=480):
    ok, encoded = cv2.imencode(".png", np.zeros((height, width, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def result(choice="on_task", confidence=0.9):
    probabilities = {label: 0.05 for label in ["on_task", "off_task"]}
    probabilities[choice] = 0.95
    return {
        "model": "jev-test",
        "answers": {"task_status": {
            "type": "choice", "choice": choice, "confidence": confidence, "probabilities": probabilities,
        }},
    }


@pytest.mark.parametrize("choice,confidence,expected", [
    ("on_task", 0.9, "on_task"), ("off_task", 0.9, "off_task"),
    ("on_task", 0.4, "on_task"), ("off_task", 0.4, "off_task"),
])
def test_upload_calls_jev_with_image_and_task(choice, confidence, expected):
    def respond(request):
        assert str(request.url) == "https://api.typesafe.ai/v1/systemone"
        assert request.headers["Authorization"] == "Bearer private-key"
        assert request.headers["TypeSafe-Danger-Zone-Images"] == "true"
        message = BytesParser(policy=default).parsebytes(
            f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode() + request.content
        )
        parts = list(message.iter_parts())
        assert parts[0].get_param("name", header="content-disposition") == "request"
        body = json.loads(parts[0].get_payload(decode=True))
        assert body["state"]["assigned_task"] == "Assemble parts"
        asset = body["state"]["photo"]
        assert body["assets"][asset]["part"] == "photo"
        assert set(body["questions"]["task_status"]["criteria"]) == {"on_task", "off_task"}
        assert "consecutive" not in body["questions"]["task_status"]["instructions"]
        assert parts[1].get_param("name", header="content-disposition") == "photo"
        image = cv2.imdecode(np.frombuffer(parts[1].get_payload(decode=True), np.uint8), cv2.IMREAD_COLOR)
        assert image.shape[:2] == (480, 640)
        return httpx.Response(200, json=result(choice, confidence))

    with TestClient(create_app("private-key", transport=httpx.MockTransport(respond))) as client:
        assert client.get("/").status_code == 200
        setup = client.get("/api/setup")
        assert setup.json()["configured"] is True
        assert "private-key" not in setup.text
        response = client.post("/api/classify", params={"task": "Assemble parts"}, content=photo(),
                               headers={"content-type": "image/png"})
        assert response.status_code == 200
        assert response.json()["decision"] == expected
        assert response.json()["model_choice"] == choice
        assert response.json()["confidence"] == confidence
        assert set(response.json()["probabilities"]) == {"on_task", "off_task"}
        assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("width,height,expected", [
    (2400, 1800, (1176, 1568)), (8064, 6048, (1176, 1568)),
    (20, 1000, (1000, 280)), (10, 10, (280, 280)),
])
def test_images_are_prepared_for_jev_limits(width, height, expected):
    encoded = prepare_image(photo(width, height))
    assert encoded.startswith(b"\xff\xd8\xff")
    decoded = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == expected


def test_invalid_uploads_and_cross_origin_requests_never_call_jev():
    def unexpected(request):
        pytest.fail("Invalid upload reached Jev")

    with TestClient(create_app("key", transport=httpx.MockTransport(unexpected))) as client:
        endpoint = "/api/classify?task=Assemble"
        assert client.post(endpoint, content=b"not an image", headers={"content-type": "image/png"}).status_code == 415
        assert client.post(endpoint, content=b"\x89PNG\r\n\x1a\n", headers={"content-type": "image/png"}).status_code == 422
        assert client.post(endpoint, content=photo(), headers={"content-type": "text/plain"}).status_code == 415
        assert client.post(endpoint, content=b"x" * (MAX_UPLOAD_BYTES + 1), headers={"content-type": "image/png"}).status_code == 413
        assert client.post(endpoint, content=photo(), headers={"content-type": "image/png", "origin": "https://other.example"}).status_code == 403
        assert client.post("/api/classify?task=%20%20", content=photo(), headers={"content-type": "image/png"}).status_code == 422
        assert client.post("/api/classify", content=photo(), headers={"content-type": "image/png"}).status_code == 422


def test_missing_key_serves_page_with_setup_instructions():
    with TestClient(create_app("")) as client:
        assert client.get("/").status_code == 200
        assert client.get("/api/setup").json()["configured"] is False
        response = client.post("/api/classify?task=Assemble", content=photo(), headers={"content-type": "image/png"})
        assert response.status_code == 503
        assert "TYPESAFE_API_KEY" in response.text


@pytest.mark.parametrize("status", [401, 403, 429, 529, 500])
def test_provider_errors_do_not_become_classifications_or_leak_secrets(status):
    with TestClient(create_app("private-key", transport=httpx.MockTransport(
        lambda request: httpx.Response(status, text="private-key secret-provider-body")
    ))) as client:
        response = client.post("/api/classify?task=Assemble", content=photo(), headers={"content-type": "image/png"})
        assert response.status_code in {502, 503}
        assert "decision" not in response.json()
        assert "private-key" not in response.text and "secret-provider-body" not in response.text


@pytest.mark.parametrize("bad_answer", [
    {"choice": "maybe"}, {"confidence": float("inf")}, {"probabilities": {"on_task": 1.0}},
    {"choice": "uncertain"},
    {"probabilities": {"on_task": -1.0, "off_task": 1.0, "uncertain": 1.0}},
    {"probabilities": {"on_task": 0.2, "off_task": 0.8}},
])
def test_malformed_provider_responses_fail_closed(bad_answer):
    payload = result()
    payload["answers"]["task_status"].update(bad_answer)
    # JSON cannot encode infinity in a valid HTTP response, so use its JSON spelling explicitly.
    raw = json.dumps(payload)
    with TestClient(create_app("key", transport=httpx.MockTransport(
        lambda request: httpx.Response(200, content=raw, headers={"content-type": "application/json"})
    ))) as client:
        response = client.post("/api/classify?task=Assemble", content=photo(), headers={"content-type": "image/png"})
        assert response.status_code == 502


def test_provider_timeout_returns_retryable_error():
    def timeout(request):
        raise httpx.ReadTimeout("private-key", request=request)

    with TestClient(create_app("key", transport=httpx.MockTransport(timeout))) as client:
        response = client.post("/api/classify?task=Assemble", content=photo(), headers={"content-type": "image/png"})
        assert response.status_code == 504
        assert "private-key" not in response.text
