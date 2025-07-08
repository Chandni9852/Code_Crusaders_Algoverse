import pathlib
import os
import sys

# ✅ Patch to avoid PosixPath error on Windows
if os.name == 'nt':
    pathlib.PosixPath = pathlib.WindowsPath


FILE = os.path.dirname(os.path.abspath(__file__))
YOLOV5_PATH = os.path.join(FILE, "yolov5")
if YOLOV5_PATH not in sys.path:
    sys.path.append(YOLOV5_PATH)

# Custom scale_coords (since it's missing from utils.general)
def scale_coords(img1_shape, coords, img0_shape):
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad = ((img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2)
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].clamp(min=0)
    return coords

from utils.augmentations import letterbox
from utils.general import non_max_suppression
from models.common import DetectMultiBackend

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import numpy as np
import io
import base64
import cv2

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use your frontend URL in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load YOLOv5 model
model = DetectMultiBackend("weights/best.pt", device="cpu")
model.eval()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".png"):
        return JSONResponse(status_code=400, content={"error": "Only PNG images are allowed."})

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    img = np.array(image)

    # Preprocess
    img_resized = letterbox(img, new_shape=640)[0]
    img_resized = img_resized.transpose((2, 0, 1))
    img_resized = np.ascontiguousarray(img_resized)

    img_tensor = torch.from_numpy(img_resized).to(model.device).float() / 255.0
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    pred = model(img_tensor, augment=False)
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45)[0]

    feedback = []
    if pred is not None and len(pred):
        pred[:, :4] = scale_coords(img_tensor.shape[2:], pred[:, :4], img.shape).round()
        for *xyxy, conf, cls in pred:
            label = model.names[int(cls)]
            x1, y1, x2, y2 = map(int, xyxy)
            color = (0, 0, 255) if label.lower() == 'reversal' else (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            feedback.append(f"{label} detected at x={x1}, y={y1}")

    _, buffer = cv2.imencode('.jpg', img)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return JSONResponse({
        "annotated_image": img_base64,
        "feedback": feedback
    })
