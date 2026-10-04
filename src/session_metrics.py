"""Interruption metrics for a voice session, derived from LiveKit state events.

- interruption: the user starts speaking while the agent is speaking and the
  agent stops; ``cancel_ms`` is the time from detected user speech to the
  agent leaving the speaking state.
- missed interruption: the user spoke over the agent for at least
  ``missed_after_s`` and the agent kept talking the whole time.
- false interruption: LiveKit decided an interruption was not real speech
  (``agent_false_interruption``), e.g. a cough or a short backchannel.

Pure state machine fed with timestamps, so it is unit-tested without LiveKit.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class InterruptionTracker:
    def __init__(
        self,
        log: Callable[..., Any] | None = None,
        *,
        missed_after_s: float = 1.0,
    ) -> None:
        self._log = log or (lambda *_a, **_k: None)
        self.missed_after_s = missed_after_s
        self.agent_state = "initializing"
        self.user_speaking = False
        self._overlap_started: float | None = None
        self.interruptions = 0
        self.missed = 0
        self.false_interruptions = 0
        self.cancel_ms: list[float] = []

    def on_user_state(self, new_state: str, ts: float) -> None:
        if new_state == "speaking":
            self.user_speaking = True
            if self.agent_state == "speaking":
                self._overlap_started = ts
        elif self.user_speaking:
            self.user_speaking = False
            started = self._overlap_started
            self._overlap_started = None
            # The user finished talking over the agent and it never stopped.
            if (
                started is not None
                and self.agent_state == "speaking"
                and ts - started >= self.missed_after_s
            ):
                self.missed += 1
                self._log(
                    "INTERRUPTION_MISSED", overlap_ms=round((ts - started) * 1000)
                )

    def on_agent_state(self, new_state: str, ts: float) -> None:
        was_speaking = self.agent_state == "speaking"
        self.agent_state = new_state
        if was_speaking and new_state != "speaking" and self._overlap_started:
            ms = (ts - self._overlap_started) * 1000
            self._overlap_started = None
            self.interruptions += 1
            self.cancel_ms.append(ms)
            self._log("INTERRUPTION", cancel_ms=round(ms, 1))
        elif new_state == "speaking" and self.user_speaking:
            # Agent started talking while the user was already talking.
            self._overlap_started = ts

    def on_false_interruption(self, resumed: bool) -> None:
        self.false_interruptions += 1
        self._log("FALSE_INTERRUPTION", resumed=resumed)

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "interruptions": self.interruptions,
            "missed_interruptions": self.missed,
            "false_interruptions": self.false_interruptions,
        }
        if self.cancel_ms:
            ordered = sorted(self.cancel_ms)
            out["avg_cancel_ms"] = round(sum(ordered) / len(ordered), 1)
            out["p90_cancel_ms"] = round(ordered[int(0.9 * (len(ordered) - 1))], 1)
        return out
