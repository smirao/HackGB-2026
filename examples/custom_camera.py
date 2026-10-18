"""Adapt a future camera script without changing the monitoring pipeline.

Set camera_backend='custom' and camera_factory='examples.custom_camera:create_camera'
in config.json. Run from the repository root so this module can be imported.
"""


class YourCamera:
    def __init__(self, settings):
        # Initialize your camera/stream here, once.
        raise NotImplementedError("Connect your camera script here")

    def read(self):
        # Return a NEW uint8 numpy array with shape (height, width, 3), BGR channel order.
        # Block until the next frame, with a finite timeout; raise RuntimeError on failure.
        # Do not launch an extra capture loop or run ML here.
        raise NotImplementedError

    def close(self):
        # Release camera/stream resources. Must also be safe after a read failure.
        pass


def create_camera(settings):
    return YourCamera(settings)
