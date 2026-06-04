"""
HERizon Sovereign — Phase 2: FastAPI Server
============================================
REST API wrapper around the Impact Catcher NLP engine.
Exposes endpoints consumed by the Next.js frontend.

Run with:
    uvicorn api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from nlp_engine import ImpactCatcher, ImpactAnalysis

app = FastAPI(
    title="HERizon Impact Catcher API",
    description="Turns invisible labor into structured Relational Equity Data",
    version="1.0.0"
)

# Allow Next.js frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://horizon-sovereign.vercel.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the NLP engine (demo mode — swap to model_path for production)
catcher = ImpactCatcher(demo_mode=True)


# ─── REQUEST / RESPONSE MODELS ───

class AnalyzeRequest(BaseModel):
    text: str
    estimated_minutes: Optional[int] = 15
    source: Optional[str] = "manual"  # "slack", "email", "manual", "nfc"

class BatchAnalyzeRequest(BaseModel):
    texts: list[str]
    source: Optional[str] = "slack"

class AnalyzeResponse(BaseModel):
    primary_category: str
    sub_category: str
    equity_score: float
    confidence: float
    monetary_estimate_usd: float
    sbt_eligible: bool
    key_phrases: list[str]
    flags: list[str]
    source: str

class VaultSummary(BaseModel):
    total_interactions: int
    total_equity_score: float
    average_equity_score: float
    total_monetary_value_usd: float
    sbt_eligible_count: int
    category_breakdown: dict
    top_impact_area: str


# ─── ENDPOINTS ───

@app.get("/health")
def health():
    return {"status": "sovereign", "engine": "DeBERTa-v3 (demo mode)"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_text(req: AnalyzeRequest):
    """
    Analyze a single message for invisible labor.
    Called by the Impact Catcher UI when user pastes text.
    """
    try:
        result: ImpactAnalysis = catcher.analyze(req.text, req.estimated_minutes)
        return AnalyzeResponse(
            primary_category=result.primary_category,
            sub_category=result.sub_category,
            equity_score=result.equity_score,
            confidence=result.confidence,
            monetary_estimate_usd=result.monetary_estimate_usd,
            sbt_eligible=result.sbt_eligible,
            key_phrases=result.key_phrases,
            flags=result.flags,
            source=req.source,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/analyze/batch")
def analyze_batch(req: BatchAnalyzeRequest):
    """
    Analyze a full Slack thread or email chain.
    Returns individual analyses + aggregate vault summary.
    """
    analyses = catcher.analyze_batch(req.texts)
    summary = catcher.aggregate_vault_score(analyses)

    return {
        "analyses": [
            {
                "text_preview": a.raw_text[:80] + "...",
                "category": a.primary_category,
                "equity_score": a.equity_score,
                "sbt_eligible": a.sbt_eligible,
                "monetary_estimate_usd": a.monetary_estimate_usd,
            }
            for a in analyses
        ],
        "vault_summary": summary
    }


@app.post("/analyze/nfc-vouch")
def analyze_nfc_vouch(req: AnalyzeRequest):
    """
    Special endpoint called after an NFC tap validation.
    Source is "nfc" — automatically flags for SBT minting.
    Sets a higher confidence baseline since a human physically vouched.
    """
    result: ImpactAnalysis = catcher.analyze(req.text, req.estimated_minutes)

    # NFC vouches get a confidence boost — a human signed off physically
    boosted_confidence = min(result.confidence + 0.15, 0.98)
    boosted_equity = min(result.equity_score + 0.5, 10.0)

    return {
        "category": result.primary_category,
        "sub_category": result.sub_category,
        "equity_score": boosted_equity,
        "confidence": boosted_confidence,
        "monetary_estimate_usd": result.monetary_estimate_usd,
        "sbt_eligible": True,   # NFC-vouched interactions are ALWAYS SBT-eligible
        "flags": result.flags + ["NFC_VOUCHED", "HUMAN_ATTESTED"],
        "mint_ready": True,
    } 