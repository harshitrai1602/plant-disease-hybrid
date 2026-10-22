from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import io
from PIL import Image
from backend.ml_api.inference import load_model, predict_image

app = FastAPI(title="Plant Disease Hybrid ML API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
MODEL_INFO = {"name": "efficientnet_b0", "path": "models/cnn/efficientnet_b0_best.pth", "loaded": False}

@app.on_event("startup")
def startup():
    global MODEL, MODEL_INFO
    try:
        MODEL = load_model(MODEL_INFO["path"])
        MODEL_INFO["loaded"] = True
        print("Model loaded:", MODEL_INFO["path"])
    except Exception as e:
        print("Model not loaded - running in health-only mode:", e)
        MODEL_INFO["error"] = str(e)

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_INFO["loaded"], "model": MODEL_INFO["name"]}

@app.get("/model-info")
def model_info():
    return MODEL_INFO

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png", "image/jpg", "image/webp"]:
        raise HTTPException(status_code=400, detail="Invalid MIME type - upload JPEG/PNG")
    if file.size and file.size > 10*1024*1024:
        raise HTTPException(status_code=400, detail="File too large - max 10MB")
    try:
        data = await file.read()
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except:
        raise HTTPException(status_code=400, detail="Corrupted or invalid image")
    if MODEL is None:
        return {"crop": "unknown", "disease": "model not loaded - health check only", "confidence": 0.0, "top_predictions": [], "model": MODEL_INFO["name"], "explanation_available": False}
    result = predict_image(MODEL, img)
    return result

@app.post("/predict/explain")
async def predict_explain(file: UploadFile = File(...)):
    result = await predict(file)
    result["explanation_available"] = True
    result["gradcam_url"] = "/gradcam/placeholder - run src/explainability/gradcam.py for heatmap"
    return result
