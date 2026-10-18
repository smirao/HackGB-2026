import asyncio
import json
from email.parser import BytesParser
from email.policy import default
from stat import S_IMODE

import cv2
import httpx
import numpy as np
import pytest
from dotenv import dotenv_values

from worker_video import get_api_key, watch


def test_prompt_saves_key_and_later_run_uses_it_without_environment(tmp_path, monkeypatch, capsys):
    env_path = tmp_path / ".env"
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr("worker_video.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("worker_video.getpass.getpass", lambda prompt: " test-secret ")
    assert get_api_key(env_path) == "test-secret"
    assert dotenv_values(env_path)["TYPESAFE_API_KEY"] == "test-secret"
    assert S_IMODE(env_path.stat().st_mode) == 0o600
    monkeypatch.setattr("worker_video.getpass.getpass", lambda prompt: pytest.fail("Unexpected key prompt"))
    monkeypatch.setattr("worker_video.sys.stdin.isatty", lambda: False)
    assert get_api_key(env_path) == "test-secret"
    output = capsys.readouterr()
    assert "test-secret" not in output.out + output.err


def test_exported_key_is_saved_while_other_dotenv_settings_are_preserved(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("# Local settings\nOTHER_SETTING='keep me'\nTYPESAFE_API_KEY='old-key'\n")
    monkeypatch.setenv("TYPESAFE_API_KEY", "new-key-with-$-and-'-quote")
    assert get_api_key(env_path) == "new-key-with-$-and-'-quote"
    saved = dotenv_values(env_path, interpolate=False)
    assert saved["TYPESAFE_API_KEY"] == "new-key-with-$-and-'-quote"
    assert saved["OTHER_SETTING"] == "keep me"
    assert "# Local settings" in env_path.read_text()


def test_missing_key_without_terminal_has_clear_setup_error(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr("worker_video.sys.stdin.isatty", lambda: False)
    with pytest.raises(RuntimeError, match="Run this script in a terminal"):
        get_api_key(env_path)
    assert not env_path.exists()


@pytest.mark.parametrize("key", ["", "secret\nINJECTED=value"])
def test_invalid_prompted_key_is_not_saved(tmp_path, monkeypatch, key):
    env_path = tmp_path / ".env"
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr("worker_video.sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("worker_video.getpass.getpass", lambda prompt: key)
    with pytest.raises(RuntimeError, match="nonempty, single-line"):
        get_api_key(env_path)
    assert not env_path.exists()


class FakeCapture:
    def __init__(self, frames, fps=10):
        self.frames = iter(frames)
        self.fps = fps
        self.released = False

    def isOpened(self):
        return True

    def set(self, *args):
        pass

    def get(self, prop):
        assert prop == cv2.CAP_PROP_FPS
        return self.fps

    def read(self):
        frame = next(self.frames, None)
        return frame is not None, frame

    def release(self):
        self.released = True


def answer(choice="on_task", confidence=0.9):
    probabilities = {label: 0.05 for label in ("on_task", "off_task")}
    probabilities[choice] = 0.95
    return {"model": "jev-test", "answers": {"task_status": {
        "type": "choice", "choice": choice, "confidence": confidence, "probabilities": probabilities,
    }}}


def test_video_samples_are_sent_while_previous_requests_are_pending(tmp_path, monkeypatch, capsys):
    video = tmp_path / "feed.mp4"
    video.touch()
    capture = FakeCapture([np.full((300, 300, 3), i * 40, np.uint8) for i in range(6)])
    monkeypatch.setattr("worker_video.cv2.VideoCapture", lambda source: capture)
    active = 0
    peak = 0
    images = []

    async def respond(request):
        nonlocal active, peak
        assert request.headers["authorization"] == "Bearer test-key"
        message = BytesParser(policy=default).parsebytes(
            f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode() + request.content
        )
        parts = list(message.iter_parts())
        body = json.loads(parts[0].get_payload(decode=True))
        assert body["state"]["assigned_task"] == "Assemble parts"
        jpeg = parts[1].get_payload(decode=True)
        image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        images.append(round(float(image.mean())))
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.25)
        active -= 1
        return httpx.Response(200, json=answer("off_task", confidence=0.4))

    asyncio.run(watch(str(video), "Assemble parts", "test-key", interval=0.2,
                      transport=httpx.MockTransport(respond)))
    output = capsys.readouterr()
    assert images == [0, 80, 160]  # Frames at 0.0, 0.2, and 0.4 seconds.
    assert peak >= 2
    assert output.out.count("off_task confidence=0.400 model_choice=off_task") == 3
    assert "0.00s" in output.out and "0.40s" in output.out
    assert capture.released


def test_slow_provider_has_bounded_concurrency_and_reports_skips(tmp_path, monkeypatch, capsys):
    video = tmp_path / "feed.mp4"
    video.touch()
    capture = FakeCapture([np.zeros((300, 300, 3), np.uint8) for _ in range(5)], fps=20)
    monkeypatch.setattr("worker_video.cv2.VideoCapture", lambda source: capture)
    calls = 0

    async def respond(request):
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.4)
        return httpx.Response(429, text="secret-provider-body")

    asyncio.run(watch(str(video), "Assemble parts", "test-key", interval=0.05, max_in_flight=1,
                      transport=httpx.MockTransport(respond)))
    output = capsys.readouterr()
    assert calls == 1
    assert "SKIPPED:" in output.err and "ERROR: Jev is busy" in output.err
    assert "secret-provider-body" not in output.err
    assert not output.out
    assert capture.released


def test_live_feed_failure_releases_camera(monkeypatch):
    capture = FakeCapture([])
    monkeypatch.setattr("worker_video.cv2.VideoCapture", lambda source: capture)
    with pytest.raises(RuntimeError, match="Video feed stopped"):
        asyncio.run(watch(0, "Assemble parts", "test-key"))
    assert capture.released


def test_real_video_uses_one_second_default(tmp_path, capsys):
    video = tmp_path / "feed.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 4, (300, 300))
    assert writer.isOpened()
    try:
        for i in range(5):
            writer.write(np.full((300, 300, 3), i * 40, np.uint8))
    finally:
        writer.release()
    calls = 0

    def respond(request):
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=answer())

    asyncio.run(watch(str(video), "Assemble parts", "test-key", transport=httpx.MockTransport(respond)))
    output = capsys.readouterr()
    assert calls == 2
    assert "0.00s" in output.out and "1.00s" in output.out
    assert output.out.count("on_task confidence=0.900") == 2
    assert not output.err
