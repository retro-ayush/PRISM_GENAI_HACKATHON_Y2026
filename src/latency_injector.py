"""Configurable, non-blocking latency injection for the mock APIs.

Simulates real backend latency so interruption, cancellation and stale-result
handling can be exercised. Delays use ``asyncio.sleep`` so a slow tool never
blocks the event loop (and therefore never blocks listening to the user).

Select a profile with the ``MOCK_LATENCY_PROFILE`` env var:
- a profile name from ``LATENCY_PROFILES`` applies that profile to every API
- ``realistic`` uses the per-API defaults in ``API_LATENCY_DEFAULTS``
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class LatencyProfile:
    name: str
    min_ms: int = 0
    max_ms: int = 0
    fixed_ms: int | None = None  # overrides min/max when set
    progressive_factor: float = 1.0  # multiplier applied per successive call

    def get_delay_ms(self, call_index: int = 0) -> int:
        if self.fixed_ms is not None:
            base = self.fixed_ms
        else:
            base = random.randint(self.min_ms, self.max_ms)
        if self.progressive_factor != 1.0 and call_index > 0:
            base = int(base * (self.progressive_factor**call_index))
        return base


LATENCY_PROFILES = {
    "instant": LatencyProfile("instant", fixed_ms=0),
    "fast": LatencyProfile("fast", min_ms=50, max_ms=200),
    "normal": LatencyProfile("normal", min_ms=200, max_ms=800),
    "slow": LatencyProfile("slow", min_ms=1000, max_ms=3000),
    "degraded": LatencyProfile(
        "degraded", min_ms=500, max_ms=2000, progressive_factor=1.5
    ),
    "timeout_risk": LatencyProfile("timeout_risk", min_ms=3000, max_ms=8000),
}

# Used by the "realistic" profile: mutations are slower than lookups.
API_LATENCY_DEFAULTS = {
    "search_flights": "normal",
    "book_flight": "slow",
    "update_identity_doc": "fast",
    "get_card_benefits": "fast",
    "get_exchange_rate": "fast",
    "modify_autopay": "slow",
    "search_apartments": "normal",
    "calculate_commute": "normal",
    "update_search_filter": "fast",
    "track_order": "fast",
    "search_products": "normal",
    "add_to_cart": "fast",
}


class LatencyInjector:
    def __init__(self, profile: str = "instant", enabled: bool = True) -> None:
        self.realistic = profile == "realistic"
        self.default_profile = LATENCY_PROFILES.get(
            profile, LATENCY_PROFILES["instant"]
        )
        self.enabled = enabled
        self.call_counts: dict[str, int] = {}
        self.call_log: list[dict] = []

    def _get_profile(self, api_name: str) -> LatencyProfile:
        if self.realistic and api_name in API_LATENCY_DEFAULTS:
            return LATENCY_PROFILES[API_LATENCY_DEFAULTS[api_name]]
        return self.default_profile

    async def inject(self, api_name: str) -> int:
        """Sleep for the profile's delay without blocking the loop. Returns ms."""
        if not self.enabled:
            return 0
        call_index = self.call_counts.get(api_name, 0)
        self.call_counts[api_name] = call_index + 1
        profile = self._get_profile(api_name)
        delay_ms = profile.get_delay_ms(call_index)
        self.call_log.append(
            {
                "api": api_name,
                "call_index": call_index,
                "profile": profile.name,
                "delay_ms": delay_ms,
                "timestamp": time.time(),
            }
        )
        if delay_ms > 0:
            await asyncio.sleep(delay_ms / 1000.0)
        return delay_ms
