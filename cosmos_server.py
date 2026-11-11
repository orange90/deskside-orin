import base64
import io
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).resolve().parent
LOCK = threading.Lock()


@asynccontextmanager
async def lifespan(app):
    checkpoint = ROOT / "models/cosmos"
    app.state.processor = AutoProcessor.from_pretrained(checkpoint, local_files_only=True)
    app.state.model = AutoModelForImageTextToText.from_pretrained(
        checkpoint, dtype=torch.bfloat16, device_map="cuda", local_files_only=True
    ).eval()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health():
    return {"ready": True}


@app.post("/v1/chat/completions")
def generate(body: dict):
    content = []
    images = 0
    try:
        for item in body["messages"][-1]["content"]:
            if item["type"] == "text":
                content.append({"type": "text", "text": item["text"][:500]})
            elif item["type"] == "image_url":
                prefix, data = item["image_url"]["url"].split(",", 1)
                if prefix != "data:image/jpeg;base64" or len(data) > 1_500_000 or images >= 4:
                    raise ValueError("invalid image")
                image = Image.open(io.BytesIO(base64.b64decode(data, validate=True))).convert("RGB")
                image.thumbnail((640, 640))
                content.append({"type": "image", "image": image})
                images += 1
    except (KeyError, IndexError, TypeError, ValueError, OSError) as error:
        raise HTTPException(422, "Only local inline JPEG images and text are accepted") from error
    processor, model = app.state.processor, app.state.model
    with LOCK, torch.inference_mode():
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt", enable_thinking=False
        ).to(model.device)
        generated = model.generate(**inputs, max_new_tokens=max(1, min(48, int(body.get("max_tokens", 48)))),
                                   do_sample=False)
        result = processor.batch_decode(generated[:, inputs.input_ids.shape[1]:],
                                        skip_special_tokens=True)[0].strip()
    return {"choices": [{"message": {"role": "assistant", "content": result}}]}
