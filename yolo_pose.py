"""YOLO26m-Pose TensorRT FP16 inference without a PyTorch runtime dependency."""
import hashlib
import time
from pathlib import Path

import cv2
import numpy as np

MODEL_NAME = 'YOLO26m-Pose'
MODEL_PATH = 'models/yolo26m-pose.engine'
MODEL_SHA256 = '074860e9a8e7d94c857e6b803832229caf9d518739127bcf1097e473b32a57b5'


def letterbox(image):
    height, width = image.shape[:2]
    if min(height, width) <= 0:
        raise ValueError('Empty camera crop')
    gain = min(640 / height, 640 / width)
    resized_width, resized_height = round(width*gain), round(height*gain)
    left = round((640-resized_width)/2 - .1)
    top = round((640-resized_height)/2 - .1)
    canvas = np.full((640, 640, 3), 114, dtype=np.uint8)
    canvas[top:top+resized_height, left:left+resized_width] = cv2.resize(image, (resized_width, resized_height))
    tensor = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32)
    tensor /= 255.0
    return tensor, (gain, left, top)


def decode(output, shape, transform):
    # Pinned exported head: cx, cy, width, height, person score, 17*(x,y,confidence).
    if output.shape != (1, 56, 8400):
        raise ValueError(f'Unexpected YOLO26 pose head: {output.shape}')
    predictions = output[0].T
    predictions = predictions[(predictions[:, 4] >= .28) & np.isfinite(predictions).all(axis=1)]
    if not len(predictions):
        return np.empty((0, 5), np.float32), np.empty((0, 17, 3), np.float32)
    xywh = predictions[:, :4].copy()
    xywh[:, :2] -= xywh[:, 2:4]/2
    keep = np.asarray(cv2.dnn.NMSBoxes(xywh.tolist(), predictions[:, 4].tolist(), .279999, .45), dtype=np.int64).reshape(-1)[:300]
    boxes = xywh[keep].copy()
    boxes[:, 2:] += boxes[:, :2]
    joints = predictions[keep, 5:].reshape(-1, 17, 3).copy()
    gain, left, top = transform
    height, width = shape[:2]
    boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]]-left)/gain, 0, width)
    boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]]-top)/gain, 0, height)
    joints[:, :, 0] = np.clip((joints[:, :, 0]-left)/gain, 0, width)
    joints[:, :, 1] = np.clip((joints[:, :, 1]-top)/gain, 0, height)
    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    return np.column_stack((boxes, predictions[keep, 4]))[valid], joints[valid]


class YoloPose:
    def __init__(self, root):
        from trt_model import TensorRT
        path = Path(root)/MODEL_PATH
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        self.backend = TensorRT(path, (1, 3, 640, 640), metadata_header=True, output_names=['output0'])
        if self.backend.metadata.get('task') != 'pose' or self.backend.metadata.get('kpt_shape') != [17, 3]:
            raise ValueError('Engine is not a compatible 17-keypoint pose model')
        self.info = {'name': MODEL_NAME, 'backend': 'TensorRT FP16', 'engine_sha256': digest, 'input_size': [640, 640]}
        self.last_ms = None

    def __call__(self, image):
        started = time.perf_counter()
        tensor, transform = letterbox(image)
        result = decode(self.backend(tensor)[0], image.shape, transform)
        self.last_ms = (time.perf_counter()-started)*1000
        return result
