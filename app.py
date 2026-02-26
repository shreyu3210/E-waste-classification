"""
E-Waste Classifier — FastAPI Server
=====================================
Endpoints:
  GET  /           → Frontend UI
  POST /predict    → Upload an image file for classification
  POST /predict-frame → Send a base64 webcam frame for classification
"""

import io
import base64
from pathlib import Path

from PIL import Image
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from ultralytics import YOLO

# ─── CONFIG ───
MODEL_PATH = Path(r"D:\Work\EV_LAB\runs\classify\runs\classify\ewaste_binary\weights\best.pt")
IMG_SIZE = 224

# ─── INIT ───
app = FastAPI(title="E-Waste Classifier", version="1.0")

# Load model once at startup
print("Loading model...")
model = YOLO(str(MODEL_PATH))
print(f"✅ Model loaded: {MODEL_PATH.name}")

# Class metadata
CLASS_META = {
    "e-waste":     {"emoji": "⚡", "color": "#ef4444", "desc": "Electronic waste — phones, laptops, PCBs, TVs, batteries"},
    "non-ewaste":  {"emoji": "♻️", "color": "#10b981", "desc": "Non-electronic waste — cardboard, glass, metal, paper, plastic, trash"},
}


def classify_image(img: Image.Image) -> dict:
    """Run classification on a PIL image and return structured results."""
    results = model.predict(source=img, imgsz=IMG_SIZE, device=0, verbose=False)
    r = results[0]
    probs = r.probs

    top1_idx = probs.top1
    top1_conf = probs.top1conf.item()
    top1_name = r.names[top1_idx]

    # Build top-5 list
    top5 = []
    for idx, conf in zip(probs.top5, probs.top5conf.tolist()):
        name = r.names[idx]
        meta = CLASS_META.get(name, {"emoji": "❓", "color": "#888", "desc": ""})
        top5.append({
            "class": name,
            "confidence": round(conf * 100, 1),
            "emoji": meta["emoji"],
            "color": meta["color"],
            "desc": meta["desc"],
        })

    meta = CLASS_META.get(top1_name, {"emoji": "❓", "color": "#888", "desc": ""})
    return {
        "prediction": top1_name,
        "confidence": round(top1_conf * 100, 1),
        "emoji": meta["emoji"],
        "color": meta["color"],
        "desc": meta["desc"],
        "is_ewaste": top1_name == "e-waste" and top1_conf > 0.5,
        "top5": top5,
    }


# ─── ROUTES ───

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the frontend UI."""
    html_path = Path(__file__).parent / "templates" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/predict")
async def predict_upload(file: UploadFile = File(...)):
    """Classify an uploaded image file."""
    contents = await file.read()
    img = Image.open(io.BytesIO(contents)).convert("RGB")
    result = classify_image(img)
    return JSONResponse(content=result)


@app.post("/predict-frame")
async def predict_frame(request: Request):
    """Classify a base64-encoded webcam frame sent as JSON."""
    body = await request.json()
    image_data = body.get("image", "")

    # Strip the data URL prefix (e.g. "data:image/jpeg;base64,...")
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    img_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    result = classify_image(img)
    return JSONResponse(content=result)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
