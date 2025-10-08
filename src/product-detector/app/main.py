from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .service import ProductClassifier
from .schemas import (
    HealthOut, PredictIn, PredictOut,
    BatchPredictIn, BatchPredictOut, BatchPredictOutItem
)

app = FastAPI(title="Product Name Detector (Text) — XLM-R XNLI")

# CORS (عدّل الأصول عند الحاجة)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

clf = ProductClassifier()

@app.get("/product/health", response_model=HealthOut)
def health():
    return HealthOut(model_name=clf.model_name, threshold=clf.threshold)

@app.post("/product/predict", response_model=PredictOut)
def predict(inp: PredictIn):
    out = clf.classify_one(inp.text)
    return PredictOut(**out)

@app.post("/product/predict-batch", response_model=BatchPredictOut)
def predict_batch(inp: BatchPredictIn):
    results = clf.classify_batch(inp.texts)
    return BatchPredictOut(
        results=[BatchPredictOutItem(**r) for r in results]
    )
