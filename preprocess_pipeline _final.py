# 🔧 Preprocessing Pipeline for Handwriting Detection

import cv2
import numpy as np
from PIL import Image
import torch # type: ignore

# 1. Load image (PIL or filepath)
def load_image(input_img):
    if isinstance(input_img, str):
        img = Image.open(input_img).convert("RGB")
    else:
        img = input_img.convert("RGB")
    return np.array(img)

# 2. Convert to grayscale
def convert_to_grayscale(np_img):
    gray = cv2.cvtColor(np_img, cv2.COLOR_RGB2GRAY)
    return gray

# 3. Resize to model expected size (e.g. 416x416)
def resize_image(gray_img, size=(416, 416)):
    resized = cv2.resize(gray_img, size)
    return resized

# 4. Normalize and convert to tensor for YOLOv5/YOLOv8
def prepare_tensor_input(resized_img):
    img = resized_img.astype(np.float32) / 255.0  # normalize to [0,1]
    img = np.expand_dims(img, axis=0)  # add channel dimension for grayscale
    img = np.expand_dims(img, axis=0)  # batch dim
    tensor = torch.tensor(img)
    return tensor

# 📦 Full pipeline

def preprocess_image(input_img):
    np_img = load_image(input_img)
    gray = convert_to_grayscale(np_img)
    resized = resize_image(gray)
    tensor = prepare_tensor_input(resized)
    return tensor
