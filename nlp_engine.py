"""
HERizon Sovereign — Phase 2: The Impact Catcher
================================================
NLP Engine using DeBERTa-v3 to classify women's invisible labor
into structured Relational Equity Data.

Model: microsoft/deberta-v3-base (fine-tuned on our custom dataset)
Task:  Multi-label classification → Impact Category + Equity Score
"""

from dataclasses import dataclass, field
from typing import Optional
import re


# ─────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────

@dataclass
class ImpactAnalysis:
    """The structured output for every analyzed message."""
    raw_text: str
    primary_category: str          # e.g. "Emotional Support"
    sub_category: str              # e.g. "Crisis De-escalation"
    equity_score: float            # 0.0 → 10.0  (Relational Equity Score)
    confidence: float              # Model confidence 0.0 → 1.0
    monetary_estimate_usd: float   # Estimated $ value of the labor
    sbt_eligible: bool             # Should this mint a Soulbound Token?
    key_phrases: list[str] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# IMPACT CATEGORIES (The Taxonomy of Invisible Labor)
# ─────────────────────────────────────────────

IMPACT_TAXONOMY = {
    "Emotional Support": {
        "sub_categories": [
            "Crisis De-escalation",
            "Active Listening",
            "Encouragement & Motivation",
            "Conflict Mediation",
            "Mental Health Check-in",
        ],
        "keywords": [
            "here for you", "how are you", "you've got this",
            "I understand", "that sounds hard", "I hear you",
            "don't worry", "we'll figure it out", "proud of you",
            "you're not alone", "take care of yourself",
        ],
        "base_equity_score": 7.5,
        "hourly_rate_usd": 85.0,   # Professional coaching rate proxy
        "description": "Active emotional labor that sustains team psychological safety",
    },
    "Informational Support": {
        "sub_categories": [
            "Knowledge Transfer",
            "Onboarding Assistance",
            "Technical Mentoring",
            "Process Documentation",
            "Resource Sharing",
        ],
        "keywords": [
            "let me show you", "here's how", "I can explain",
            "documentation", "tutorial", "walkthrough", "guide",
            "tip", "best practice", "you should know", "FYI",
            "resource", "template", "example",
        ],
        "base_equity_score": 8.2,
        "hourly_rate_usd": 120.0,  # Senior consultant rate proxy
        "description": "Knowledge transfer that accelerates team capability",
    },
    "Relational Labor": {
        "sub_categories": [
            "Team Cohesion Building",
            "Inclusion Facilitation",
            "Recognition & Celebration",
            "Networking Bridging",
            "Culture Stewardship",
        ],
        "keywords": [
            "introduce you to", "you two should connect",
            "celebrate", "shoutout", "kudos", "well done",
            "team lunch", "check in with", "include", "welcome",
            "let's make sure everyone", "don't forget",
        ],
        "base_equity_score": 6.8,
        "hourly_rate_usd": 95.0,   # HR/Culture specialist rate proxy
        "description": "Social glue work that makes organizations function",
    },
    "Cognitive Labor": {
        "sub_categories": [
            "Invisible Planning",
            "Anticipatory Work",
            "Administrative Burden-Carrying",
            "Meeting Facilitation",
            "Follow-up & Accountability",
        ],
        "keywords": [
            "I'll organize", "I'll take notes", "I'll follow up",
            "reminder", "agenda", "action items", "I already",
            "I took care of", "I handled", "I scheduled",
            "I coordinated", "I arranged",
        ],
        "base_equity_score": 7.0,
        "hourly_rate_usd": 75.0,   # Project coordinator rate proxy
        "description": "The mental load of keeping everything running",
    },
    "Mentoring & Sponsorship": {
        "sub_categories": [
            "Formal Mentoring",
            "Informal 'Pantry' Mentoring",
            "Career Guidance",
            "Sponsorship & Advocacy",
            "Skill Development Coaching",
        ],
        "keywords": [
            "mentor", "coach", "career advice", "I'd recommend",
            "in my experience", "when I was", "let me tell you",
            "for your career", "you should apply", "I'll vouch",
            "I'll put your name forward", "I'll recommend you",
        ],
        "base_equity_score": 9.5,  # Highest score — most undervalued
        "hourly_rate_usd": 200.0,  # Executive coaching rate proxy
        "description": "Career-shaping guidance with zero formal recognition",
    },
}


# ─────────────────────────────────────────────
# RULE-BASED FALLBACK CLASSIFIER
# (Used when DeBERTa model is not loaded — e.g. demo mode)
# ─────────────────────────────────────────────

def rule_based_classify(text: str) -> tuple[str, str, float]:
    """
    Fast keyword-based classifier.
    Returns: (category, sub_category, confidence)
    """
    text_lower = text.lower()
    best_category = "Relational Labor"
    best_sub = "Team Cohesion Building"
    best_score = 0.0

    for category, data in IMPACT_TAXONOMY.items():
        matches = sum(1 for kw in data["keywords"] if kw.lower() in text_lower)
        if matches > best_score:
            best_score = matches
            best_category = category
            # Pick the most relevant sub-category
            for sub in data["sub_categories"]:
                sub_keywords = sub.lower().split()
                if any(sk in text_lower for sk in sub_keywords):
                    best_sub = sub
                    break
            else:
                best_sub = data["sub_categories"][0]

    confidence = min(0.55 + (best_score * 0.08), 0.92)
    return best_category, best_sub, confidence


def extract_key_phrases(text: str, category: str) -> list[str]:
    """Extract the specific phrases that triggered classification."""
    text_lower = text.lower()
    found = []
    if category in IMPACT_TAXONOMY:
        for kw in IMPACT_TAXONOMY[category]["keywords"]:
            if kw.lower() in text_lower:
                # Find the surrounding context (±3 words)
                idx = text_lower.find(kw.lower())
                start = max(0, idx - 20)
                end = min(len(text), idx + len(kw) + 20)
                phrase = "..." + text[start:end].strip() + "..."
                found.append(phrase)
    return found[:5]  # Return top 5 triggers


def estimate_monetary_value(
    category: str,
    equity_score: float,
    estimated_minutes: int = 15
) -> float:
    """
    Calculate the monetary proxy for this invisible labor.

    Formula:
        value = (hourly_rate / 60) * minutes * (equity_score / 10) * impact_multiplier
    """
    if category not in IMPACT_TAXONOMY:
        return 0.0

    hourly_rate = IMPACT_TAXONOMY[category]["hourly_rate_usd"]
    per_minute = hourly_rate / 60
    base_value = per_minute * estimated_minutes
    score_weight = equity_score / 10
    impact_multiplier = 1.0 + (equity_score - 5.0) * 0.1  # Amplify high-impact work

    return round(base_value * score_weight * impact_multiplier, 2)


# ─────────────────────────────────────────────
# MAIN CLASSIFIER (Orchestrator)
# ─────────────────────────────────────────────

class ImpactCatcher:
    """
    Main entry point for the NLP pipeline.

    In production: loads microsoft/deberta-v3-base fine-tuned checkpoint.
    In demo mode:  uses fast rule-based classifier (same output schema).
    """

    def __init__(self, model_path: Optional[str] = None, demo_mode: bool = True):
        self.demo_mode = demo_mode
        self.model = None
        self.tokenizer = None

        if not demo_mode and model_path:
            self._load_deberta(model_path)

    def _load_deberta(self, model_path: str):
        """Load fine-tuned DeBERTa-v3 model."""
        try:
            # These imports are only needed when running the real model
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            print(f"Loading DeBERTa-v3 from: {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=len(IMPACT_TAXONOMY),
                problem_type="multi_label_classification"
            )
            self.model.eval()
            print("✓ DeBERTa-v3 model loaded successfully")
        except Exception as e:
            print(f"⚠ Model load failed, falling back to rule-based: {e}")
            self.demo_mode = True

    def _deberta_predict(self, text: str) -> tuple[str, str, float]:
        """Run DeBERTa inference."""
        import torch
        import torch.nn.functional as F

        categories = list(IMPACT_TAXONOMY.keys())

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.sigmoid(logits)[0]   # Multi-label sigmoid

        top_idx = int(probs.argmax())
        confidence = float(probs[top_idx])
        category = categories[top_idx]
        sub_category = IMPACT_TAXONOMY[category]["sub_categories"][0]

        return category, sub_category, confidence

    def analyze(self, text: str, estimated_minutes: int = 15) -> ImpactAnalysis:
        """
        Analyze a piece of text and return full ImpactAnalysis.

        Args:
            text: The message / note to analyze
            estimated_minutes: How long the interaction took

        Returns:
            ImpactAnalysis with equity score, category, and SBT eligibility
        """
        if not text or len(text.strip()) < 5:
            raise ValueError("Text too short to analyze meaningfully.")

        # ── Step 1: Classification ──
        if self.demo_mode:
            category, sub_category, confidence = rule_based_classify(text)
        else:
            category, sub_category, confidence = self._deberta_predict(text)

        # ── Step 2: Equity Score ──
        base_score = IMPACT_TAXONOMY[category]["base_equity_score"]
        # Adjust: longer text = more substantive interaction
        length_bonus = min(len(text.split()) / 100, 0.8)
        # Adjust: higher confidence = more clear-cut labor
        confidence_bonus = (confidence - 0.5) * 1.0
        equity_score = round(
            min(base_score + length_bonus + confidence_bonus, 10.0), 2
        )

        # ── Step 3: Key Phrases ──
        key_phrases = extract_key_phrases(text, category)

        # ── Step 4: Monetary Estimate ──
        monetary_value = estimate_monetary_value(
            category, equity_score, estimated_minutes
        )

        # ── Step 5: SBT Eligibility ──
        # Mint a token if: high equity score AND high confidence
        sbt_eligible = equity_score >= 7.0 and confidence >= 0.65

        # ── Step 6: Build flags ──
        flags = []
        if equity_score >= 9.0:
            flags.append("HIGH_IMPACT")
        if category == "Mentoring & Sponsorship":
            flags.append("PANTRY_MENTORING_CANDIDATE")
        if "crisis" in text.lower() or "difficult" in text.lower():
            flags.append("CRISIS_SUPPORT")
        if sbt_eligible:
            flags.append("SBT_READY")

        return ImpactAnalysis(
            raw_text=text,
            primary_category=category,
            sub_category=sub_category,
            equity_score=equity_score,
            confidence=confidence,
            monetary_estimate_usd=monetary_value,
            sbt_eligible=sbt_eligible,
            key_phrases=key_phrases,
            flags=flags,
        )

    def analyze_batch(self, texts: list[str]) -> list[ImpactAnalysis]:
        """Analyze multiple messages — e.g. a Slack thread."""
        return [self.analyze(t) for t in texts if t.strip()]

    def aggregate_vault_score(self, analyses: list[ImpactAnalysis]) -> dict:
        """
        Aggregate multiple analyses into a Vault Summary.
        Used to build the Promotion Advocacy Report in Phase 5.
        """
        if not analyses:
            return {}

        total_equity = sum(a.equity_score for a in analyses)
        total_monetary = sum(a.monetary_estimate_usd for a in analyses)
        sbt_count = sum(1 for a in analyses if a.sbt_eligible)

        category_breakdown = {}
        for a in analyses:
            cat = a.primary_category
            category_breakdown[cat] = category_breakdown.get(cat, 0) + 1

        return {
            "total_interactions": len(analyses),
            "total_equity_score": round(total_equity, 2),
            "average_equity_score": round(total_equity / len(analyses), 2),
            "total_monetary_value_usd": round(total_monetary, 2),
            "sbt_eligible_count": sbt_count,
            "category_breakdown": category_breakdown,
            "top_impact_area": max(category_breakdown, key=category_breakdown.get),
        }


# ─────────────────────────────────────────────
# DEMO: Run this file directly to see it work
# ─────────────────────────────────────────────

if __name__ == "__main__":
    catcher = ImpactCatcher(demo_mode=True)

    test_messages = [
        "Hey, I noticed you seemed stressed in the meeting today. I just wanted to check in — are you doing okay? I've been there and I know how overwhelming it can get. My door is always open.",
        "Here's a complete guide I put together on how to navigate the performance review process. I wish someone had shown me this when I started. Let me walk you through it step by step.",
        "I wanted to make sure you got credit for the Q3 analysis — I mentioned your work to the VP and said I think you should lead the next phase. You deserve that visibility.",
        "Reminder to everyone: please submit your timesheets by Friday. I've already followed up with 3 people individually and organized the approval queue so it flows smoothly.",
        "I'd love to introduce you to Priya — she's doing incredible work in the same space and I think you two would really benefit from knowing each other.",
    ]

    print("\n" + "═"*60)
    print("  HERizon Sovereign — Impact Catcher Demo")
    print("═"*60 + "\n")

    analyses = catcher.analyze_batch(test_messages)

    for i, analysis in enumerate(analyses, 1):
        print(f"Message {i}:")
        print(f"  Text preview: \"{analysis.raw_text[:60]}...\"")
        print(f"  Category:     {analysis.primary_category}")
        print(f"  Sub-type:     {analysis.sub_category}")
        print(f"  Equity Score: {analysis.equity_score}/10")
        print(f"  Confidence:   {analysis.confidence:.0%}")
        print(f"  $ Value:      ${analysis.monetary_estimate_usd}")
        print(f"  SBT Eligible: {'✓ YES — Mint Token' if analysis.sbt_eligible else '✗ Not yet'}")
        print(f"  Flags:        {', '.join(analysis.flags) if analysis.flags else 'none'}")
        print()

    print("─"*60)
    summary = catcher.aggregate_vault_score(analyses)
    print(f"  VAULT SUMMARY:")
    print(f"  Total Interactions:   {summary['total_interactions']}")
    print(f"  Total Equity Score:   {summary['total_equity_score']}")
    print(f"  Total $ Value:        ${summary['total_monetary_value_usd']}")
    print(f"  Tokens to Mint:       {summary['sbt_eligible_count']} SBTs")
    print(f"  Top Impact Area:      {summary['top_impact_area']}")
    print("═"*60 + "\n")