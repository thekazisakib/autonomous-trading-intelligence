from __future__ import annotations

from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """Template for LLM prompts."""
    system_prompt: str
    user_prompt_template: str
    variables: Dict[str, str]


# Prompt template for market analysis
MARKET_ANALYSIS_TEMPLATE = PromptTemplate(
    system_prompt="""
You are a financial analyst AI specializing in market analysis for algorithmic trading.
Your role is to provide THESIS SCORES, not trading signals.
- Provide thesis scores between -1.0 and +1.0
- +1.0 = Strong bullish thesis
- -1.0 = Strong bearish thesis
- 0.0 = Neutral/No clear thesis
- NEVER make specific price predictions
- NEVER recommend specific entry/exit prices
- NEVER recommend order sizes
- Focus on fundamentals, technicals, and market sentiment
- Output JSON only with: "thesis_score", "reasoning", "confidence"
""",
    user_prompt_template="""
Analyze the following market data for {symbol}:

Current Data:
- LTP: {ltp}
- RSI (14): {rsi}
- Volume Surge: {volume_surge}%
- Thesis Score (from cold path): {thesis_score}
- Market Sentiment: {market_sentiment}
- Portfolio Position: {portfolio_pct}% of capital

Please provide:
1. thesis_score: A score between -1.0 and +1.0
2. reasoning: A 2-3 sentence explanation
3. confidence: A confidence score between 0.0 and 1.0

Return JSON only with keys: thesis_score, reasoning, confidence
""")


def get_market_analysis_prompt(symbol: str, ltp: float, rsi: float,
                               volume_surge: float, thesis_score: float,
                               market_sentiment: float, portfolio_pct: float) -> str:
    """Get the market analysis prompt with substituted variables."""
    return MARKET_ANALYSIS_TEMPLATE.user_prompt_template.format(
        symbol=symbol,
        ltp=ltp,
        rsi=rsi,
        volume_surge=volume_surge,
        thesis_score=thesis_score,
        market_sentiment=market_sentiment,
        portfolio_pct=portfolio_pct
    )