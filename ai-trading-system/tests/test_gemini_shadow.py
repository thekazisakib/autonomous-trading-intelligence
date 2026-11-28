import asyncio
import json
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from agentic.gemini_shadow import (
    GeminiAnalysisError,
    GeminiShadowAnalyst,
    ShadowAnalysis,
)
from config.loader import load_config
from data.store import FeatureSnapshot, MarketTick
from orchestrator import TradingOrchestrator


class _Response:
    def __init__(self, value: dict) -> None:
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *_args) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.value).encode("utf-8")


def _gemini_response(score: float = 0.3) -> dict:
    return {
        "candidates": [{
            "content": {
                "parts": [{
                    "text": json.dumps({
                        "thesis_score": score,
                        "confidence": 0.6,
                        "reasoning": "Limited indicators; further evidence is needed.",
                    })
                }]
            }
        }]
    }


def test_gemini_shadow_uses_key_header_and_only_indicator_payload() -> None:
    analyst = GeminiShadowAnalyst(
        "rotated-key",
        "gemini-test-model",
        min_interval_seconds=300,
    )
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return _Response(_gemini_response())

    with patch("agentic.gemini_shadow.urlopen", side_effect=fake_urlopen):
        result = analyst.analyze("RELIANCE", 100.5, 42.0, 15.0, timestamp=1000)

    request = captured["request"]
    body = json.loads(request.data)
    prompt = body["contents"][0]["parts"][0]["text"]
    assert request.get_header("X-goog-api-key") == "rotated-key"
    assert captured["timeout"] == 15
    assert json.loads(prompt) == {
        "symbol": "RELIANCE",
        "last_price": 100.5,
        "rsi_14": 42.0,
        "volume_surge_pct": 15.0,
    }
    assert "position" not in prompt.lower()
    assert isinstance(result, ShadowAnalysis)
    assert result.thesis_score == 0.3
    assert analyst.analyze("RELIANCE", 100.5, 42.0, 15.0, timestamp=1100) is None


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"candidates": [{"content": {"parts": [{"text": "{}"}]}}]},
        {
            "candidates": [{
                "content": {
                    "parts": [{
                        "text": json.dumps({
                            "thesis_score": 4,
                            "confidence": 0.6,
                            "reasoning": "invalid score",
                        })
                    }]
                }
            }]
        },
    ],
)
def test_gemini_shadow_rejects_malformed_or_out_of_range_responses(response) -> None:
    analyst = GeminiShadowAnalyst("rotated-key", "gemini-test-model")
    with pytest.raises(GeminiAnalysisError):
        analyst._parse_response("RELIANCE", response, 1000)


def test_gemini_shadow_does_not_throttle_after_failed_request() -> None:
    analyst = GeminiShadowAnalyst("rotated-key", "gemini-test-model")
    with patch.object(
        analyst,
        "_request",
        side_effect=[
            GeminiAnalysisError("temporary failure"),
            _gemini_response(),
        ],
    ):
        with pytest.raises(GeminiAnalysisError):
            analyst.analyze("RELIANCE", 100, None, None, timestamp=1000)
        result = analyst.analyze("RELIANCE", 100, None, None, timestamp=1001)

    assert result is not None
    assert result.symbol == "RELIANCE"


def test_orchestrator_persists_gemini_shadow_result_without_using_it_to_trade(
    tmp_path,
) -> None:
    config = load_config()
    trading = config.trading.model_copy(update={
        "market_data_enabled": False,
        "auto_execute": False,
    })
    observability = config.observability.model_copy(
        update={"db_path": str(tmp_path / "gemini-shadow.sqlite")}
    )
    telegram_control = config.telegram_control.model_copy(update={"enabled": False})
    app_config = config.model_copy(update={
        "trading": trading,
        "observability": observability,
        "telegram_control": telegram_control,
    })
    app = TradingOrchestrator(app_config)

    class FakeAnalyst:
        def analyze(self, *args):
            return ShadowAnalysis(
                symbol="RELIANCE",
                timestamp=1000,
                thesis_score=0.25,
                confidence=0.5,
                reasoning="Shadow-only test.",
                model="fake-model",
            )

    app.gemini_analyst = FakeAnalyst()
    tick = MarketTick(
        symbol="RELIANCE",
        timestamp=datetime.now(timezone.utc),
        ltp=100,
    )
    feature = FeatureSnapshot(
        symbol="RELIANCE",
        timestamp=tick.timestamp,
        rsi_14=40,
        volume_surge_pct=10,
    )

    try:
        asyncio.run(app._run_gemini_shadow_analysis(tick, feature))
        stored = app.store.get_latest_shadow_gemini_analysis("RELIANCE")
        assert stored is not None
        assert stored["thesis_score"] == 0.25
        assert app.store.get_position("RELIANCE") is None
        assert "Shadow-only test." in app._format_gemini_analysis("RELIANCE")
    finally:
        app.store.close()
