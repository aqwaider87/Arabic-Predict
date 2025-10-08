from pydantic import BaseModel, Field
from typing import List, Optional, Any

class HealthOut(BaseModel):
    status: str = "ok"
    model_name: str
    threshold: float

class PredictIn(BaseModel):
    text: str = Field(..., description="النص المراد تصنيفه")

class PredictOut(BaseModel):
    is_product: bool
    confidence: float
    label: str
    raw: Optional[Any] = None  # ضعها None في الإنتاج إذا لا تريد تفاصيل

class BatchPredictIn(BaseModel):
    texts: List[str] = Field(..., min_items=1, description="قائمة نصوص")

class BatchPredictOutItem(BaseModel):
    input: str
    is_product: bool
    confidence: float
    label: str

class BatchPredictOut(BaseModel):
    results: List[BatchPredictOutItem]
