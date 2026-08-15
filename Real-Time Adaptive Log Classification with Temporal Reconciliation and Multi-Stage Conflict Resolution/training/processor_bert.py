# processor_bert.py

from pathlib import Path

import joblib
from sentence_transformers import SentenceTransformer

try:
    from .schemas import ClassificationSignal
except ImportError:
    from schemas import ClassificationSignal


CLASSIFIER_PATH = Path(__file__).resolve().parent / "models" / "log_classifier.joblib"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
ML_CONFIDENCE_THRESHOLD = 0.50
_embedding_model: SentenceTransformer | None = None
_classification_model = None


def _load_models():
    """Load the existing models once, when this stage is actually invoked."""
    global _embedding_model, _classification_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    if _classification_model is None:
        _classification_model = joblib.load(CLASSIFIER_PATH)
    return _embedding_model, _classification_model


def classify_signal_with_bert(log_message: str) -> ClassificationSignal:
    """Return the best Logistic Regression result and its confidence."""
    embedding_model, classification_model = _load_models()
    embeddings = embedding_model.encode([log_message])
    probabilities = classification_model.predict_proba(embeddings)[0]
    best_index = max(range(len(probabilities)), key=lambda index: float(probabilities[index]))
    confidence = float(probabilities[best_index])
    if confidence < ML_CONFIDENCE_THRESHOLD:
        return ClassificationSignal(
            stage="bert", label="Unclassified", confidence=confidence, matched=False,
            reason=f"Best model confidence {confidence:.6f} is below threshold {ML_CONFIDENCE_THRESHOLD:.2f}.",
        )
    return ClassificationSignal(
        stage="bert", label=str(classification_model.classes_[best_index]), confidence=confidence,
        matched=True, reason="Highest Logistic Regression probability selected.",
    )


def classify_with_bert(log_message: str) -> str:
    """Legacy label-only adapter retained for the CSV compatibility route."""
    return classify_signal_with_bert(log_message).label


if __name__ == "__main__":
    logs = [
        "alpha.osapi_compute.wsgi.server - 12.10.11.1 - API returned 404 not found error",
        "GET /v2/3454/servers/detail HTTP/1.1 RCODE   404 len: 1583 time: 0.1878400",
        "System crashed due to drivers errors when restarting the server",
        "Hey bro, chill ya!",
        "Multiple login failures occurred on user 6454 account",
        "Server A790 was restarted unexpectedly during the process of data transfer"
    ]
    for log in logs:
        label = classify_with_bert(log)
        print(log, "->", label)
