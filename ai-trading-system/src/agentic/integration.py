from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any

from agentic.analyst import AnalysisContext, AnalysisResult, AnalysisDecision
from agentic.prompt_templates import get_market_analysis_prompt


class AgenticIntegration:
    """
    Integrates agentic AI analysis into the trading system.
    
    Key design principles:
    1. Agentic AI provides thesis ADJUSTMENTS, not trade signals
    2. Adjustments are added to the existing thesis score from the cold path
    3. Analysis is throttled (not on every tick)
    4. Output is conservative and defensive
    5. Never bypasses the risk engine
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        self.enabled = config.get("enabled", True)
        self.llm_enabled = config.get("llm_enabled", False)
        self.analysis_interval = config.get("analysis_interval_seconds", 300)  # 5 minutes
        self.thesis_weight = config.get("thesis_weight", 0.1)
        
        # Create the analyst
        from agentic.analyst import AgenticAnalyst
        self.analyst = AgenticAnalyst(
            enabled=self.enabled,
            llm_enabled=self.llm_enabled
        )
    
    def analyze_market(self, context: Dict[str, Any]) -> Optional[float]:
        """
        Analyze market conditions and return thesis adjustment.
        
        Args:
            context: Dictionary with market data:
                - symbol: str
                - ltp: float
                - rsi: float
                - volume_surge: float
                - thesis_score: float (from cold path)
                - market_sentiment: float (-1 to +1)
                - portfolio_pct: float
                
        Returns:
            thesis_adjustment: float (-1 to +1 to add to existing thesis)
            None if analysis should be skipped
        """
        if not self.enabled:
            return None
        
        # Build analysis context
        context = AnalysisContext(
            symbol=context.get("symbol", "UNKNOWN"),
            timestamp=self._get_timestamp(),
            ltp=context.get("ltp", 0.0),
            rsi=context.get("rsi", 50.0),
            volume_surge=context.get("volume_surge", 0.0),
            thesis_score=context.get("thesis_score", 0.0),
            market_sentiment=context.get("market_sentiment", 0.0),
            portfolio_pct=context.get("portfolio_pct", 0.0),
        )
        
        # Run analysis
        result = self.analyst.analyze(context)
        
        # Apply thesis weight - the agentic AI's adjustment is weighted
        # The weight controls how much influence the agentic AI has
        adjusted_adjustment = result.thesis_adjustment * self.thesis_weight
        
        return adjusted_adjustment
    
    def _get_timestamp(self) -> float:
        """Get current Unix timestamp."""
        import time
        return time.time()
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get statistics about analysis activity."""
        return {
            "enabled": self.enabled,
            "llm_enabled": self.llm_enabled,
            "analysis_count": self.analyst.analysis_count,
            "analysis_interval": self.analysis_interval,
            "thesis_weight": self.thesis_weight,
        }