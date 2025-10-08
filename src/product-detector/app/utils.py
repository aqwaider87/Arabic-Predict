import re
from typing import List

_AR_DIAC = re.compile(r"[\u0617-\u061A\u064B-\u0652]")
_AR_TATWEEL = re.compile(r"[\u0640]")
_AR_PUNCT = re.compile(r"[^\w\s#@+\-_/]")  # اسمح ببعض الرموز الشائعة

def normalize_arabic(text: str) -> str:
    """تنظيف مبسّط للنص العربي: إزالة التشكيل والتمطيط وتوحيد الألف/الهمزات."""
    if not isinstance(text, str):
        return ""
    t = text.strip()
    t = _AR_DIAC.sub("", t)
    t = _AR_TATWEEL.sub("", t)
    # توحيد الألف
    t = t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    # توحيد الهاء/ة (اختياري جداً) — علّقها إن ما بدك
    # t = t.replace("ة", "ه")
    # إزالة رموز كثيرة مزعجة
    t = _AR_PUNCT.sub(" ", t)
    t = re.sub(r"\s+", " ", t)
    return t

def batch_chunks(lst: List[str], size: int = 16) -> List[List[str]]:
    for i in range(0, len(lst), size):
        yield lst[i:i+size]
