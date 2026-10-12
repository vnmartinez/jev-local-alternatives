"""Jev-style typed decision with a local zero-shot model (GLiClass, CPU)."""
import time

from gliclass import GLiClassModel, ZeroShotClassificationPipeline
from transformers import AutoTokenizer

MODEL = "knowledgator/gliclass-modern-base-v3.0"
LABELS = ["billing", "bug report", "feature request", "account access"]
TEXTS = [
    "I was charged twice for my subscription this month.",
    "The export button crashes the app when the list is empty.",
    "It would be great to have a dark mode.",
    "I can't log in, the password reset email never arrives.",
]

model = GLiClassModel.from_pretrained(MODEL)
tokenizer = AutoTokenizer.from_pretrained(MODEL)
pipe = ZeroShotClassificationPipeline(model, tokenizer, classification_type="multi-label", device="cpu")

for text, expected in zip(TEXTS, LABELS):
    start = time.perf_counter()
    scores = {r["label"]: r["score"] for r in pipe(text, LABELS, threshold=0.0)[0]}
    ms = (time.perf_counter() - start) * 1000
    best = max(scores, key=scores.get)
    print(f"{ms:6.0f} ms  {best:16} {'ok ' if best == expected else 'MISS'}  {text}")
    print("          " + "  ".join(f"{k}={v:.2f}" for k, v in sorted(scores.items(), key=lambda kv: -kv[1])))
