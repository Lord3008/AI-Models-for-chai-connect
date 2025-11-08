from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import re

# Lightweight sentiment analyzer
# pip install vaderSentiment
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception as e:
    SentimentIntensityAnalyzer = None  # will raise at runtime if used without install

router = APIRouter(prefix="/nlp", tags=["nlp"])

# Request/response schemas
class TrustRequest(BaseModel):
    texts: Optional[List[str]] = None  # list of texts
    text: Optional[str] = None         # single text (alternative to texts)
    model_path: Optional[str] = None   # optional path to a TF model (not used by default)

class TrustItem(BaseModel):
    text: str
    cleaned_text: str
    sentiment: Dict[str, float]
    trust_score: float

class TrustResponse(BaseModel):
    items: List[TrustItem]
    summary: Dict[str, Any]

# Simplified text preprocessor adapted from notebook
def text_preprocessor(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # remove emojis, non-ascii, URLs and replace digits with '#'
    emoji_pattern = re.compile("[["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "]+", flags=re.UNICODE)
    non_ascii_pattern = re.compile(r"[^\x00-\x7F]+", flags=re.UNICODE)
    digit_pattern = re.compile(r"[0-9]", flags=re.UNICODE)
    link_pattern = re.compile(r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)", flags=re.UNICODE)

    out = emoji_pattern.sub("", text)
    out = non_ascii_pattern.sub("", out)
    out = digit_pattern.sub("#", out)
    out = link_pattern.sub("", out)
    out = out.strip()
    return out

def _ensure_analyzer():
    if SentimentIntensityAnalyzer is None:
        raise RuntimeError("vaderSentiment not installed. Install with: pip install vaderSentiment")
    return SentimentIntensityAnalyzer()

def compute_trust_from_compound(compound: float, text_len: int) -> float:
    """
    Map compound (-1..1) to base trust (0..1), then blend with length factor.
    - base = (compound + 1) / 2
    - length_factor = min(1.0, text_len / 200)
    - final = 0.75 * base + 0.25 * length_factor
    Returns value between 0 and 1.
    """
    base = max(0.0, min(1.0, (compound + 1.0) / 2.0))
    length_factor = min(1.0, text_len / 200.0)
    final = 0.75 * base + 0.25 * length_factor
    return float(max(0.0, min(1.0, final)))

@router.post("/trust_score", response_model=TrustResponse)
async def trust_score_endpoint(req: TrustRequest):
    """
    Compute trust score(s) for provided text(s).
    Provide either 'text' (single) or 'texts' (list).
    """
    texts = []
    if req.text and isinstance(req.text, str):
        texts = [req.text]
    elif req.texts and isinstance(req.texts, list):
        texts = req.texts
    else:
        raise HTTPException(status_code=400, detail="Provide 'text' or 'texts' in the request body.")

    analyzer = _ensure_analyzer()

    items: List[TrustItem] = []
    compounds = []
    for t in texts:
        cleaned = text_preprocessor(t)
        # If a TF model path is provided you could load and run it here (placeholder)
        # ...existing code...
        sent = analyzer.polarity_scores(cleaned)
        compound = float(sent.get("compound", 0.0))
        trust = compute_trust_from_compound(compound, len(cleaned))
        compounds.append(compound)
        items.append(TrustItem(
            text=t,
            cleaned_text=cleaned,
            sentiment=sent,
            trust_score=round(trust, 4)
        ))

    # summary statistics
    avg_compound = float(sum(compounds) / len(compounds)) if compounds else 0.0
    avg_trust = float(sum(it.trust_score for it in items) / len(items)) if items else 0.0
    summary = {
        "count": len(items),
        "avg_compound": round(avg_compound, 4),
        "avg_trust_score": round(avg_trust, 4)
    }

    return TrustResponse(items=items, summary=summary)