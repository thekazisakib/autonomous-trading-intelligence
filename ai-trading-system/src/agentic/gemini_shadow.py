from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)


class GeminiAnalysisError(RuntimeError):
    pass


@dataclass(frozen=True)
class ShadowAnalysis:
    symbol: str
    timestamp: float
    thesis_score: float
    confidence: float
    reasoning: str
    model: str


class GeminiShadowAnalyst:
    """Rate-limited Gemini technical summary; output is not an execution signal."""

    def __init__(
        self,
        api_key: str,
        model: str,
        min_interval_seconds: int = 900,
        timeout_seconds: int = 15,
    ) -> None:
        if not api_key.strip():
            raise ValueError("A rotated Gemini API key is required")
        if not model.strip():
            raise ValueError("Gemini model name is required")
        if min_interval_seconds < 300 or timeout_seconds <= 0:
            raise ValueError("Invalid Gemini throttle or timeout")
        self._api_key = api_key.strip()
        self.model = model.strip()
        self.min_interval_seconds = min_interval_seconds
        self.timeout_seconds = timeout_seconds
        self._last_request_by_symbol: dict[str, float] = {}

    def analyze(
        self,
        symbol: str,
        ltp: float,
        rsi: float | None,
        volume_surge_pct: float | None,
        timestamp: float | None = None,
    ) -> ShadowAnalysis | None:
        now = time.time() if timestamp is None else timestamp
        if now - self._last_request_by_symbol.get(symbol, 0) < self.min_interval_seconds:
            return None
        if not math.isfinite(ltp) or ltp <= 0:
            raise ValueError("Price must be finite and positive")
        for value in (rsi, volume_surge_pct):
            if value is not None and not math.isfinite(value):
                raise ValueError("Market indicators must be finite")

        payload = {
            "systemInstruction": {
                "parts": [{
                    "text": (
                        "You are a shadow-mode technical research assistant. "
                        "Return JSON only with thesis_score from -1 to 1, "
                        "confidence from 0 to 1, and concise reasoning. "
                        "This is not a buy/sell signal. Do not recommend orders, "
                        "prices, position sizes, or promise returns. Use only the "
                        "provided indicators; explicitly note that they are limited."
                    )
                }]
            },
            "contents": [{
                "parts": [{
                    "text": json.dumps({
                        "symbol": symbol,
                        "last_price": ltp,
                        "rsi_14": rsi,
                        "volume_surge_pct": volume_surge_pct,
                    })
                }]
            }],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "OBJECT",
                    "properties": {
                        "thesis_score": {"type": "NUMBER"},
                        "confidence": {"type": "NUMBER"},
                        "reasoning": {"type": "STRING"},
                    },
                    "required": ["thesis_score", "confidence", "reasoning"],
                },
                "temperature": 0.1,
                "maxOutputTokens": 256,
            },
        }

        # A failed request must not consume the per-symbol throttle window.
        result = self._request(payload)
        analysis = self._parse_response(symbol, result, now)
        self._last_request_by_symbol[symbol] = now
        return analysis

    def _request(self, payload: dict) -> dict:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent"
        )
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self._api_key,
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read())
        except HTTPError as exc:
            raise GeminiAnalysisError(
                f"Gemini returned HTTP {exc.code}"
            ) from None
        except (URLError, TimeoutError, OSError) as exc:
            raise GeminiAnalysisError(
                f"Gemini request failed: {type(exc).__name__}"
            ) from None
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise GeminiAnalysisError(
                f"Gemini returned invalid JSON: {type(exc).__name__}"
            ) from None
        if not isinstance(body, dict):
            raise GeminiAnalysisError("Gemini returned an invalid response")
        return body

    def _parse_response(
        self,
        symbol: str,
        response: dict,
        timestamp: float,
    ) -> ShadowAnalysis:
        try:
            text = response["candidates"][0]["content"]["parts"][0]["text"]
            values = json.loads(text)
            score = float(values["thesis_score"])
            confidence = float(values["confidence"])
            reasoning = str(values["reasoning"]).strip()
        except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise GeminiAnalysisError(
                f"Gemini response did not match the expected analysis schema: "
                f"{type(exc).__name__}"
            ) from None

        if (
            not math.isfinite(score)
            or not -1 <= score <= 1
            or not math.isfinite(confidence)
            or not 0 <= confidence <= 1
            or not reasoning
            or len(reasoning) > 1000
        ):
            raise GeminiAnalysisError("Gemini response values failed validation")
        return ShadowAnalysis(
            symbol=symbol,
            timestamp=timestamp,
            thesis_score=score,
            confidence=confidence,
            reasoning=reasoning,
            model=self.model,
        )
