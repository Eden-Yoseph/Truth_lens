"""
TruthLens AI
fake news detection API with multi-model analysis,
calibrated scoring, and robust error handling.
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import requests
import logging
import time
import re
import os

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("truthlens")


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────
@dataclass(frozen=True)
class Config:
    HF_API_TOKEN: str = os.getenv("HF_API_TOKEN", "")
    HF_API_URL: str = "https://api-inference.huggingface.co/models/"
    MAX_TEXT_LENGTH: int = 2000
    REQUEST_TIMEOUT: int = 30
    MAX_RETRIES: int = 3
    RATE_LIMIT_DEFAULT: str = "30 per minute"
    RATE_LIMIT_BULK: str = "5 per minute"


cfg = Config()


# ─────────────────────────────────────────────
# Flask app + extensions
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=[cfg.RATE_LIMIT_DEFAULT],
    storage_uri="memory://",
)


# ─────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────

MODELS: dict[str, dict] = {
    "toxicity": {
        "name": "martin-ha/toxic-comment-model",
        "purpose": "toxicity_detection",
    },
    "sentiment": {
        "name": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "purpose": "sentiment_analysis",
    },
}


# ─────────────────────────────────────────────
# Fake-news linguistic indicators
# ─────────────────────────────────────────────
CLICKBAIT_PHRASES: list[str] = [
    "you won't believe", "doctors hate this", "secret that",
    "they don't want you to know", "shocking truth", "incredible discovery",
    "amazing breakthrough", "miracle cure", "scientists baffled",
    "experts shocked", "unbelievable results", "this will blow your mind",
    "you'll never guess", "what happens next", "number \d+ will shock you",
    "the result will surprise you", "click here to find out",
]
SENSATIONAL_WORDS: list[str] = [
    "amazing", "incredible", "shocking", "unbelievable", "miraculous",
    "revolutionary", "breakthrough", "exclusive", "leaked", "exposed",
    "revealed", "secret",
]
EMOTIONAL_TRIGGERS: list[str] = [
    "outraged", "disgusted", "terrified", "heartbroken", "devastated", "furious",
]

# Pre-compile regex patterns for performance
_CLICKBAIT_RE = re.compile(
    "|".join(CLICKBAIT_PHRASES), flags=re.IGNORECASE
)
_CAPS_WORD_RE = re.compile(r'\b[A-Z]{3,}\b')
_REPEATED_CHAR_RE = re.compile(r'(.)\1{2,}')
_EXCESSIVE_PUNCT_RE = re.compile(r'[!?]{2,}')
_URL_RE = re.compile(
    r'https?://(?:[a-zA-Z0-9$\-_.+!*\'(),]|(?:%[0-9a-fA-F]{2}))+'
)
_MENTION_RE = re.compile(r'@\w+')
_SENTENCE_END_RE = re.compile(r'[.!?]+')


# ─────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────
@dataclass
class AIResult:
    toxicity_score: float = 0.0
    toxicity_label: str = "unknown"
    sentiment_label: str = "neutral"
    sentiment_score: float = 0.0
    # True when the sentiment confidence is extreme (regardless of polarity)
    extreme_sentiment: bool = False
    models_called: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class TextFeatures:
    word_count: int = 0
    sentence_count: int = 0
    avg_word_length: float = 0.0
    exclamation_marks: int = 0
    question_marks: int = 0
    quote_count: int = 0
    all_caps_words: int = 0
    repeated_chars: int = 0
    excessive_punct: int = 0
    url_count: int = 0
    mention_count: int = 0
    clickbait_matches: int = 0
    sensational_word_count: int = 0
    emotional_trigger_count: int = 0


@dataclass
class CredibilityResult:
    score: float           # 0–100, higher = more credible
    label: str             # CREDIBLE | SUSPICIOUS | UNCERTAIN
    confidence: float      # 0–100
    risk_factors: list[str]
    positive_factors: list[str]
    timestamp: str


# ─────────────────────────────────────────────
# HuggingFace API client
# ─────────────────────────────────────────────
class HFClient:
    """Thin, stateless wrapper around the HuggingFace Inference API."""

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        if cfg.HF_API_TOKEN:
            self._session.headers["Authorization"] = f"Bearer {cfg.HF_API_TOKEN}"

    def query(self, model_key: str, text: str) -> Optional[list | dict]:
        """
        Call a HuggingFace classification model.
        Returns the raw JSON on success, None on failure.
        """
        model_name = MODELS[model_key]["name"]
        url = f"{cfg.HF_API_URL}{model_name}"

        for attempt in range(cfg.MAX_RETRIES):
            try:
                resp = self._session.post(
                    url,
                    json={"inputs": text},
                    timeout=cfg.REQUEST_TIMEOUT,
                )
                if resp.status_code == 200:
                    logger.info("HF call OK: model=%s", model_key)
                    return resp.json()

                if resp.status_code == 503:
                    wait = 2 ** attempt
                    logger.warning("Model loading (%s), retry in %ds", model_key, wait)
                    time.sleep(wait)
                    continue

                logger.error(
                    "HF error: model=%s status=%d body=%s",
                    model_key, resp.status_code, resp.text[:200],
                )
                return None

            except requests.Timeout:
                logger.warning("Timeout: model=%s attempt=%d", model_key, attempt + 1)
                if attempt < cfg.MAX_RETRIES - 1:
                    time.sleep(1)
            except requests.RequestException as exc:
                logger.error("Request error: model=%s exc=%s", model_key, exc)
                return None

        return None

    def close(self) -> None:
        self._session.close()


# ─────────────────────────────────────────────
# Text feature extractor
# ─────────────────────────────────────────────
def extract_features(text: str) -> TextFeatures:
    words = text.split()
    lower = text.lower()

    feat = TextFeatures()
    feat.word_count = len(words)
    feat.sentence_count = len(_SENTENCE_END_RE.findall(text)) or 1
    feat.avg_word_length = (
        sum(len(w) for w in words) / len(words) if words else 0.0
    )
    feat.exclamation_marks = text.count("!")
    feat.question_marks = text.count("?")
    feat.quote_count = text.count('"') + text.count("'")
    feat.all_caps_words = len(_CAPS_WORD_RE.findall(text))
    feat.repeated_chars = len(_REPEATED_CHAR_RE.findall(text))
    feat.excessive_punct = len(_EXCESSIVE_PUNCT_RE.findall(text))
    feat.url_count = len(_URL_RE.findall(text))
    feat.mention_count = len(_MENTION_RE.findall(text))
    feat.clickbait_matches = len(_CLICKBAIT_RE.findall(lower))
    feat.sensational_word_count = sum(
        1 for w in SENSATIONAL_WORDS if w in lower
    )
    feat.emotional_trigger_count = sum(
        1 for t in EMOTIONAL_TRIGGERS if t in lower
    )
    return feat


# ─────────────────────────────────────────────
# AI result parser  (BUG FIX: correct nested structure handling)
# ─────────────────────────────────────────────
def _best_label(items: list[dict], target_labels: set[str]) -> tuple[str, float]:
    """
    Return the (label, score) of the highest-scoring item whose label
    is in target_labels, or the single top item if none match.
    Handles both flat [{"label":..,"score":..}] and nested [[{...}]] shapes.
    """
    # Flatten one level if wrapped in outer list
    flat: list[dict] = []
    for item in items:
        if isinstance(item, list):
            flat.extend(item)
        elif isinstance(item, dict):
            flat.append(item)

    if not flat:
        return "unknown", 0.0

    # Sort descending by score
    ranked = sorted(flat, key=lambda x: x.get("score", 0.0), reverse=True)

    for entry in ranked:
        label = str(entry.get("label", "")).lower()
        if any(t in label for t in target_labels):
            return label, float(entry.get("score", 0.0))

    # Fallback: just return top entry
    top = ranked[0]
    return str(top.get("label", "unknown")).lower(), float(top.get("score", 0.0))


def parse_ai_results(
    raw_toxicity: Optional[list | dict],
    raw_sentiment: Optional[list | dict],
) -> AIResult:
    result = AIResult()

    # ── Toxicity ────────────────────────────────────────────
    if raw_toxicity is not None:
        try:
            items = raw_toxicity if isinstance(raw_toxicity, list) else [raw_toxicity]
            label, score = _best_label(items, {"toxic", "fake", "hate", "offensive"})
            result.toxicity_label = label
            result.toxicity_score = score
            result.models_called.append("toxicity")
        except Exception as exc:
            logger.error("Toxicity parse error: %s", exc)
            result.errors.append(f"toxicity parse: {exc}")

    # ── Sentiment ───────────────────────────────────────────
    if raw_sentiment is not None:
        try:
            items = raw_sentiment if isinstance(raw_sentiment, list) else [raw_sentiment]
            label, score = _best_label(items, {"negative", "positive", "neutral"})
            result.sentiment_label = label
            result.sentiment_score = score
            result.models_called.append("sentiment")
            # Extreme = very high confidence in a strongly-valenced label
            result.extreme_sentiment = (
                score > 0.88 and label in ("negative", "positive")
            )
        except Exception as exc:
            logger.error("Sentiment parse error: %s", exc)
            result.errors.append(f"sentiment parse: {exc}")

    return result


# ─────────────────────────────────────────────
# Calibrated credibility scorer
# ─────────────────────────────────────────────
# Each factor contributes a signed delta on a 0–100 scale.
# Weights were chosen to reflect research on fake-news signals; tune as needed.

def score_credibility(feat: TextFeatures, ai: AIResult) -> CredibilityResult:
    score: float = 50.0   # neutral prior
    risk: list[str] = []
    positive: list[str] = []

    # ── Linguistic red flags ─────────────────────────────────
    if feat.clickbait_matches > 0:
        delta = min(feat.clickbait_matches * 12, 30)
        score -= delta
        risk.append(
            f"Clickbait language detected ({feat.clickbait_matches} phrase(s))"
        )

    if feat.sensational_word_count >= 3:
        score -= 8
        risk.append("Heavy use of sensational vocabulary")
    elif feat.sensational_word_count >= 1:
        score -= 3

    if feat.emotional_trigger_count >= 2:
        score -= 7
        risk.append("Multiple emotional trigger words")

    # ── Formatting red flags ────────────────────────────────
    if feat.all_caps_words >= 3:
        score -= 8
        risk.append("Excessive ALL-CAPS usage")
    elif feat.all_caps_words >= 1:
        score -= 3

    if feat.exclamation_marks >= 4:
        score -= 6
        risk.append("Excessive exclamation marks")
    elif feat.exclamation_marks >= 2:
        score -= 2

    if feat.excessive_punct >= 2:
        score -= 5
        risk.append("Repeated multi-punctuation (!! / ??)")

    # ── Content signals ─────────────────────────────────────
    if feat.word_count < 15:
        score -= 10
        risk.append("Too brief for credible reporting")
    elif feat.word_count >= 80:
        score += 6
        positive.append("Substantial article length")

    if feat.sentence_count >= 4:
        score += 4
        positive.append("Well-structured prose")

    if feat.quote_count >= 2:
        score += 8
        positive.append("Contains quoted sources or dialogue")
    elif feat.quote_count == 1:
        score += 3

    if feat.url_count >= 1:
        score += 4
        positive.append("References external links")

    # ── AI model signals ────────────────────────────────────
    if "toxicity" in ai.models_called:
        if ai.toxicity_score >= 0.8:
            score -= 22
            risk.append(
                f"AI classifier: strong toxic/misleading signal ({ai.toxicity_score:.0%})"
            )
        elif ai.toxicity_score >= 0.5:
            score -= 10
            risk.append(
                f"AI classifier: moderate toxic signal ({ai.toxicity_score:.0%})"
            )
        else:
            score += 5
            positive.append("AI classifier: low toxicity signal")

    if "sentiment" in ai.models_called:
        if ai.extreme_sentiment:
            score -= 8
            risk.append(
                f"Extreme {ai.sentiment_label} sentiment ({ai.sentiment_score:.0%} confidence)"
            )

    # ── Clamp and label ─────────────────────────────────────
    score = max(0.0, min(100.0, score))

    if score >= 60:
        label = "CREDIBLE"
    elif score <= 38:
        label = "SUSPICIOUS"
    else:
        label = "UNCERTAIN"

    # Confidence = how many independent signals fired
    signal_count = len(risk) + len(positive)
    confidence = min(95.0, 50.0 + signal_count * 5.0)

    return CredibilityResult(
        score=round(score, 1),
        label=label,
        confidence=round(confidence, 1),
        risk_factors=risk,
        positive_factors=positive,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


# ─────────────────────────────────────────────
# Singleton HF client (reuse TCP connections)
# ─────────────────────────────────────────────
hf = HFClient()


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
SAMPLE_TEXTS = {
    "real": (
        "The UAE government announced new regulations for artificial intelligence "
        "development, focusing on ethical AI practices and data privacy. The policy "
        "initiative, confirmed by the Ministry of AI, aims to position the country as "
        "a global leader in responsible innovation, with phased compliance deadlines "
        "starting in Q1 2025."
    ),
    "fake": (
        "Scientists have discovered that sleeping upsidedown "
        "can increase lifespan by 200%!! You won't believe what "
        "happens next. Doctors HATE this one weird trick! Incredible breakthrough that "
        "they don't want you to know about!!!"
    ),
    "unclear": (
        "A new study suggests that people who eat chocolate daily may have better "
        "memory, though the research sample was small and the funding source was "
        "not disclosed."
    ),
}


@app.route("/")
def index():
    return render_template("index.html", samples=SAMPLE_TEXTS)


@app.route("/samples")
def get_samples():
    return jsonify(SAMPLE_TEXTS)


@app.route("/check", methods=["POST"])
@limiter.limit(cfg.RATE_LIMIT_DEFAULT)
def check_news():
    """Single-text credibility analysis endpoint."""
    body = request.get_json(silent=True)
    if not body or "text" not in body:
        return jsonify({"error": "Request body must contain a 'text' field."}), 400

    text = body["text"].strip()
    if not text:
        return jsonify({"error": "Text field is empty."}), 400
    if len(text) > cfg.MAX_TEXT_LENGTH:
        return jsonify(
            {"error": f"Text exceeds {cfg.MAX_TEXT_LENGTH} character limit."}
        ), 400

    logger.info("Analysing text (len=%d)", len(text))

    # ── AI calls (run both regardless; errors surfaced in AIResult) ──
    raw_tox = hf.query("toxicity", text)
    raw_sent = hf.query("sentiment", text)

    # ── Feature extraction ───────────────────────────────────────────
    features = extract_features(text)

    # ── Parse AI results (BUG-FIXED parser) ─────────────────────────
    ai = parse_ai_results(raw_tox, raw_sent)

    # ── Calibrated scoring ───────────────────────────────────────────
    result = score_credibility(features, ai)

    return jsonify(
        {
            "classification": result.label,
            "credibility_score": result.score,
            "confidence": result.confidence,
            "is_credible": result.label == "CREDIBLE",
            "explanation": (
                ". ".join(result.risk_factors[:3])
                if result.risk_factors
                else "No significant credibility concerns detected."
            ),
            "analysis_details": {
                "risk_factors": result.risk_factors,
                "positive_factors": result.positive_factors,
                "text_stats": {
                    "word_count": features.word_count,
                    "sentence_count": features.sentence_count,
                    "avg_word_length": round(features.avg_word_length, 2),
                },
                "ai": {
                    "models_used": ai.models_called,
                    "toxicity_score": ai.toxicity_score,
                    "toxicity_label": ai.toxicity_label,
                    "sentiment_label": ai.sentiment_label,
                    "sentiment_score": ai.sentiment_score,
                    "parse_errors": ai.errors,
                },
                "timestamp": result.timestamp,
            },
        }
    )


@app.route("/analyze/bulk", methods=["POST"])
@limiter.limit(cfg.RATE_LIMIT_BULK)
def bulk_analyze():
    """Bulk analysis (pattern-only, no AI calls to stay within rate limits)."""
    body = request.get_json(silent=True)
    if not body or not isinstance(body.get("texts"), list):
        return jsonify({"error": "'texts' must be a JSON array."}), 400

    texts: list = body["texts"]
    if len(texts) > 10:
        return jsonify({"error": "Maximum 10 texts per bulk request."}), 400

    results = []
    for idx, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            results.append({"index": idx, "error": "Empty or invalid text."})
            continue

        features = extract_features(text.strip())
        # Bulk uses pattern-only scoring (no AI to stay within API limits)
        empty_ai = AIResult()
        cred = score_credibility(features, empty_ai)

        results.append(
            {
                "index": idx,
                "preview": text[:120].rstrip() + ("…" if len(text) > 120 else ""),
                "classification": cred.label,
                "credibility_score": cred.score,
                "confidence": cred.confidence,
                "risk_count": len(cred.risk_factors),
            }
        )

    return jsonify(
        {
            "results": results,
            "total": len(results),
            "ai_powered": False,
            "note": "Bulk endpoint uses pattern-only scoring. Use /check for AI-enhanced analysis.",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    )


@app.route("/health")
def health():
    return jsonify(
        {
            "status": "healthy",
            "service": "TruthLens AI — Elite Edition",
            "models": list(MODELS.keys()),
            "api_token_configured": bool(cfg.HF_API_TOKEN),
            "rate_limits": {
                "/check": cfg.RATE_LIMIT_DEFAULT,
                "/analyze/bulk": cfg.RATE_LIMIT_BULK,
            },
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
    )


# ─────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("\n🔍 TruthLens AI — Elite Edition")
    print("─" * 50)
    if cfg.HF_API_TOKEN:
        print("HuggingFace token: configured")
    else:
        print(" HuggingFace token: NOT SET")
        print("   Set HF_API_TOKEN env var for AI-enhanced analysis.")
    print(f"Models: {', '.join(MODELS)}")
    print(f" Rate limiting: {cfg.RATE_LIMIT_DEFAULT} (check), {cfg.RATE_LIMIT_BULK} (bulk)")
    print("http://127.0.0.1:5000")
    print("─" * 50 + "\n")

    app.run(debug=False, host="127.0.0.1", port=5000)
