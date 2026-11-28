from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from enum import Enum
import json
import time


class AnalysisDecision(Enum):
    """LLM analysis decisions."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    NEUTRAL = "NEUTRAL"


@dataclass
class AnalysisContext:
    """Context for LLM analysis."""
    symbol: str
    timestamp: float  # Unix timestamp
    ltp: float
    rsi: float
    volume_surge: float
    thesis_score: float  # From cold path (LangGraph/MCP)
    market_sentiment: float  # -1 to +1 from news analysis
    portfolio_pct: float  # Current position %


@dataclass
class AnalysisResult:
    """Result from LLM analysis."""
    decision: AnalysisDecision
    thesis_adjustment: float  # Adjustment to add to existing thesis (-1 to +1)
    reasoning: str
    confidence: float  # 0.0 to 1.0
    timestamp: float


class AgenticAnalyst:
    """
    Agentic AI analyst that provides market analysis and thesis adjustments.
    
    Key design principles:
    1. NEVER controls entry/exit timing directly
    2. NEVER bypasses the risk engine
    3. ONLY provides thesis adjustments (-1 to +1)
    4. Analyzes on a timer, not per-tick
    5. Output is conservative and defensive
    """
    
    def __init__(self, enabled: bool = True, llm_enabled: bool = False) -> None:
        self.enabled = enabled
        self.llm_enabled = llm_enabled
        self.analysis_count = 0
        self.last_analysis_time = 0
        self.analysis_interval = 300  # 5 minutes in seconds
        
        # If LLM is not available, use rule-based analysis
        self.llm_available = False
    
    def analyze(self, context: AnalysisContext) -> AnalysisResult:
        """Analyze market conditions and provide thesis adjustment."""
        self.analysis_count += 1
        current_time = current_time = time.time()
        
        # Throttle analysis - don't analyze on every tick
        if self.enabled and (current_time - self.last_analysis_time < self.analysis_interval):
            # Return existing thesis if analysis too frequent
            return AnalysisResult(
                decision=AnalysisDecision.HOLD,
                thesis_adjustment=0.0,
                reasoning="Analysis throttled - too frequent",
                confidence=1.0,
                timestamp=current_time
            )
        
        self.last_analysis_time = current_time
        
        # If LLM is available, use it
        if self.llm_available:
            return self._llm_analyze(context)
        
        # Rule-based analysis (default)
        return self._rule_based_analyze(context)
    
    def _rule_based_analyze(self, context: AnalysisContext) -> AnalysisResult:
        """Rule-based analysis when LLM is not available."""
        thesis_adjustment = 0.0
        reasoning_parts = []
        
        # Factor 1: RSI contribution
        if context.rsi < 30:
            thesis_adjustment += 0.15
            reasoning_parts.append(f"RSI oversold ({context.rsi:.1f})")
        elif context.rsi > 70:
            thesis_adjustment -= 0.15
            reasoning_parts.append(f"RSI overbought ({context.rsi:.1f})")
        
        # Factor 2: Volume surge contribution
        if context.volume_surge > 50:
            thesis_adjustment += 0.10
            reasoning_parts.append(f"Volume surge ({context.volume_surge:.1f}%)")
        
        # Factor 3: Thesis score from cold path
        if context.thesis_score > 0.5:
            thesis_adjustment += context.thesis_score * 0.3
            reasoning_parts.append(f"Strong thesis ({context.thesis_score:.2f})")
        elif context.thesis_score < -0.5:
            thesis_adjustment -= abs(context.thesis_score) * 0.3
            reasoning_parts.append(f"Weak thesis ({context.thesis_score:.2f})")
        
        # Factor 4: Market sentiment contribution
        if context.market_sentiment > 0.5:
            thesis_adjustment += 0.10
            reasoning_parts.append(f"Positive sentiment ({context.market_sentiment:.2f})")
        elif context.market_sentiment < -0.5:
            thesis_adjustment -= 0.10
            reasoning_parts.append(f"Negative sentiment ({context.market_sentiment:.2f})")
        
        # Factor 5: Portfolio contribution
        if context.portfolio_pct > 0 and context.portfolio_pct > 10:
            thesis_adjustment -= 0.05
            reasoning_parts.append(f"High concentration ({context.portfolio_pct:.1f}%)")
        
        # Clamp thesis adjustment to [-1, 1]
        thesis_adjustment = max(-1.0, min(1.0, thesis_adjustment))
        
        # Determine decision
        if thesis_adjustment > 0.2:
            decision = AnalysisDecision.BUY
        elif thesis_adjustment < -0.2:
            decision = AnalysisDecision.SELL
        else:
            decision = AnalysisDecision.HOLD
        
        return AnalysisResult(
            decision=decision,
            thesis_adjustment=thesis_adjustment,
            reasoning="; ".join(reasoning_parts) if reasoning_parts else "No strong signals",
            confidence=min(abs(thesis_adjustment) * 2 + 0.3, 1.0),
            timestamp=time.time()
        )
    
    def _llm_analyze(self, context: AnalysisContext) -> AnalysisResult:
        """
        LLM-based analysis (placeholder - requires LLM API).
        
        In a real implementation, this would call an LLM API (OpenAI, Anthropic, etc.)
        with a carefully engineered prompt to analyze market conditions.
        """
        # Placeholder - in production, this would call an LLM
        # For now, fall back to rule-based
        return self._rule_based_analyze(context)
    
    def should_analyze(self, current_time: float) -> bool:
        """Check if enough time has passed since last analysis."""
        return (current_time - self.last_analysis_time) >= self.analysis_interval