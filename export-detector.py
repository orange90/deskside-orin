# Historical RTMDet export only; current production uses rebuild-vision.sh / build-yolo-engine.py.
"""Keep the RTMDet network and move per-class NMS to the application."""
from pathlib import Path
import onnx

root = Path(__file__).resolve().parent
onnx.utils.extract_model(str(root / "models/rtmdet.onnx"),
                         str(root / "models/rtmdet-raw.onnx"),
                         ["input"], ["1062", "1035"])
