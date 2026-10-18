import threading
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest

from camera_stream import Camera, create_app


def test_browser_page_and_multiple_mjpeg_clients_share_the_camera():
    image = np.full((120, 160, 3), 80, np.uint8)
    ok, encoded = cv2.imencode('.jpg', image)
    assert ok
    jpeg = encoded.tobytes()
    camera = SimpleNamespace(stop=threading.Event(), get_frame=Mock(return_value=jpeg))
    app = create_app(camera)
    with app.test_client() as first, app.test_client() as second:
        page = first.get('/')
        assert page.status_code == 200 and b'/video_feed' in page.data
        streams = [client.get('/video_feed', buffered=False) for client in (first, second)]
        try:
            for response in streams:
                assert response.status_code == 200
                assert response.headers['Content-Type'] == 'multipart/x-mixed-replace; boundary=frame'
                assert response.headers['Cache-Control'] == 'no-store'
                chunk = next(response.response)
                header, content = chunk.split(b'\r\n\r\n', 1)
                assert header.startswith(b'--frame\r\nContent-Type: image/jpeg\r\n')
                assert f'Content-Length: {len(jpeg)}'.encode() in header
                assert content == jpeg + b'\r\n'
                decoded = cv2.imdecode(np.frombuffer(content[:-2], np.uint8), cv2.IMREAD_COLOR)
                assert decoded.shape == image.shape
        finally:
            for response in streams:
                response.close()
    assert camera.get_frame.call_count == 2


def test_capture_failure_ends_stream_instead_of_repeating_stale_frames():
    stop = threading.Event()
    jpeg = b'test-jpeg'

    def get_frame(previous):
        if previous is None:
            return jpeg
        stop.set()
        return None

    camera = SimpleNamespace(stop=stop, get_frame=get_frame)
    with create_app(camera).test_client() as client:
        response = client.get('/video_feed', buffered=False)
        iterator = iter(response.response)
        assert jpeg in next(iterator)
        with pytest.raises(StopIteration):
            next(iterator)
        response.close()


def test_camera_publishes_jpeg_and_releases_device_after_capture_error(monkeypatch):
    capture = Mock()
    capture.isOpened.return_value = True
    release = threading.Event()
    image = np.zeros((120, 160, 3), np.uint8)
    first = True

    def read():
        nonlocal first
        if first:
            first = False
            return True, image
        release.wait(2)
        raise cv2.error('Disconnected')

    capture.read.side_effect = read
    monkeypatch.setattr('camera_stream.cv2.VideoCapture', lambda index: capture)
    camera = Camera()
    try:
        jpeg = camera.get_frame()
        assert jpeg.startswith(b'\xff\xd8')
        release.set()
        camera.thread.join(timeout=2)
        assert camera.stop.is_set()
        assert camera.get_frame(jpeg) is None
    finally:
        release.set()
        camera.release()
    capture.release.assert_called_once()


def test_unavailable_camera_is_released(monkeypatch):
    capture = Mock()
    capture.isOpened.return_value = False
    monkeypatch.setattr('camera_stream.cv2.VideoCapture', lambda index: capture)
    with pytest.raises(RuntimeError, match='Could not open camera'):
        Camera()
    capture.release.assert_called_once()
