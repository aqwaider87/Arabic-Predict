

# Product Classification System
# نظام تصنيف المنتجات


## 📁 File Structure

```
product-detector/
├─ app/
│  ├─ __init__.py
│  ├─ main.py              # نقطة الدخول لـ FastAPI
│  ├─ service.py           # منطق التحميل والتصنيف
│  ├─ schemas.py           # Pydantic models
│  └─ utils.py             # تنظيف النص العربي وإعدادات مساعدة
├─ requirements.txt
├─ .env                    # إعدادات قابلة للتعديل
```

## 🚀 Quick Start

### 1. Installation

python -m venv .venv
source .venv/bin/activate  # على ويندوز: .venv\Scripts\activate
pip install -r requirements.txt

# تأكد من وجود ملف .env (أو متغيرات البيئة)
uvicorn src.product-detector.app.main:app --reload --host 0.0.0.0 --port 8006


curl -X POST http://localhost:8000/product/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"آيفون 15 برو ماكس"}'

curl -X POST http://localhost:8000/product/predict-batch \
  -H "Content-Type: application/json" \
  -d '{"texts":["آيفون 15 برو ماكس","خلينا نطلع مشوار","عطر ديور سوفاج"]}'
