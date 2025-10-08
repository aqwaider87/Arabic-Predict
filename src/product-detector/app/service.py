import os
from pathlib import Path
from typing import Dict, List
from transformers import pipeline, logging as hf_logging
from .utils import normalize_arabic

# Silence transformers logging (user requested no logs/prints)
hf_logging.set_verbosity_error()


class ProductClassifier:
    def __init__(self):
        # Prefer local fine-tuned model directory if present, otherwise fallback to HF hub model
        local_model_dirs = [
            Path("models/product_detector_model/best_model"),
            Path("models/product_detector_model"),
        ]
        detected_local = next((str(p) for p in local_model_dirs if p.exists()), None)
        self.model_name = os.getenv("MODEL_NAME", detected_local or "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")
        self.threshold = float(os.getenv("THRESHOLD", "0.6"))
        self.hypothesis_template = os.getenv("HYPOTHESIS_TEMPLATE", "هذا النص {}.")
        self.label_product = os.getenv("LABEL_PRODUCT", "يمثل اسم منتج")
        self.label_not_product = os.getenv("LABEL_NOT_PRODUCT", "لا يمثل اسم منتج")
        self.max_len = int(os.getenv("MAX_SEQ_LEN", "160"))

        # تحميل الـ pipeline مرة واحدة (caching)
        self.clf = pipeline(
            task="zero-shot-classification",
            model=self.model_name,
        )

        # ثابتة لخفض الـ latency (يجوز تغيّرها لاحقًا حسب نتائج القياس)
        self.candidate_labels = [self.label_product, self.label_not_product]

    def classify_one(self, text: str) -> Dict:
        text = normalize_arabic(text)

        result = self.clf(
            text,
            candidate_labels=self.candidate_labels,
            hypothesis_template=self.hypothesis_template,
        )

        top_label = result["labels"][0]
        top_score = float(result["scores"][0])
        is_prod = (top_label == self.label_product) and (top_score >= self.threshold)

        return {
            "is_product": is_prod,
            "confidence": round(top_score, 4),
            "label": top_label,
            "raw": result  # يمكنك حذفها بالإنتاج
        }

    def classify_batch(self, texts: List[str]) -> List[Dict]:
        # FastAPI/Transformers يدعمون دفعات مباشرة
        normed = [normalize_arabic(t) for t in texts]
        results = self.clf(
            normed,
            candidate_labels=self.candidate_labels,
            hypothesis_template=self.hypothesis_template,
        )
        outputs = []
        for inp, r in zip(texts, results):
            top_label = r["labels"][0]
            top_score = float(r["scores"][0])
            is_prod = (top_label == self.label_product) and (top_score >= self.threshold)
            outputs.append({
                "input": inp,
                "is_product": is_prod,
                "confidence": round(top_score, 4),
                "label": top_label
            })
        return outputs
