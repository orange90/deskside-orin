from pathlib import Path

import cv2
import numpy as np

import candidate_main

app = candidate_main.app
marker = Path(__file__).with_name("first-attempt")
if not marker.exists():
    marker.touch()
    def injected_failure():
        raise RuntimeError("injected camera worker failure")
    candidate_main.capture_frames = injected_failure
else:
    class SyntheticCapture:
        def __init__(self, *args):
            self.image = np.full((720, 1280, 3), 100, np.uint8)

        def set(self, *args):
            return True

        def read(self):
            return True, self.image

        def release(self):
            pass

    cv2.VideoCapture = SyntheticCapture
