"""Coordination layer: versioned session state, tool lifecycle, stale-result
protection, idempotent mutations and a JSONL event trace.

It is deterministic and provider-agnostic: the voice stack (LiveKit) calls
``on_user_turn`` when a user turn is committed and ``run_tool`` when the LLM
calls a tool. Models never mutate state directly; they only choose tools and
arguments, which are validated here first.

Version rule: every committed user turn creates a new state version. A tool
call is bound to the version it started under, so if the user speaks again
while it is in flight, its result is stale:
- read-only results are rejected and never reach the LLM as fact;
- mutations are aborted before commit if the intent changed during the
  simulated backend delay, and are committed at most once per idempotency key.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import itertools
import json
import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from tool_registry import ToolRegistry, ToolSpec, ToolValidationError

logger = logging.getLogger("coordinator")


class CallStatus(str, Enum):
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    SUPERSEDED = "SUPERSEDED"  # a newer user turn arrived while in flight
    CANCELLED = "CANCELLED"  # the voice layer cancelled it (barge-in)
    ABORTED = "ABORTED"  # mutation stopped before commit due to stale intent
    FAILED = "FAILED"


@dataclass
class ToolCall:
    call_id: str
    tool_name: str
    state_version: int
    parameters: dict[str, Any]
    idempotency_key: str | None
    status: CallStatus = CallStatus.RUNNING
    started_at: float = field(default_factory=time.time)
    ended_at: float | None = None
    task: asyncio.Task | None = field(default=None, repr=False)


@dataclass
class SessionState:
    session_id: str
    version: int = 0
    intent: str | None = None
    slots: dict[str, Any] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "version": self.version,
            "intent": self.intent,
            "slots": dict(self.slots),
        }


class TraceLogger:
    """Append-only, machine-readable event trace (JSON Lines)."""

    def __init__(self, session_id: str, path: str | Path | None = None) -> None:
        self.session_id = session_id
        self.events: list[dict[str, Any]] = []
        self._seq = itertools.count(1)
        self._file = None
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._file = open(path, "a", encoding="utf-8", buffering=1)  # noqa: SIM115

    def log(self, event_type: str, state_version: int, **data: Any) -> dict[str, Any]:
        event = {
            "seq": next(self._seq),
            "ts": time.time(),
            "session_id": self.session_id,
            "type": event_type,
            "state_version": state_version,
            **data,
        }
        self.events.append(event)
        if self._file:
            self._file.write(json.dumps(event, default=str) + "\n")
        logger.debug("trace %s", event)
        return event

    def of_type(self, event_type: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["type"] == event_type]

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None


_CANCEL_RE = re.compile(
    r"\b(cancel|stop|never ?mind|forget (it|that)|don'?t|do not|hold on)\b", re.I
)
_CORRECTION_RE = re.compile(
    r"\b(actually|wait|instead|i meant|rather|make it|change (it|that)|"
    r"not \w+,? but|no,)\b",
    re.I,
)


def classify_utterance(text: str) -> str:
    """Fast, deterministic cue-word routing for a user turn (no model call)."""
    if _CANCEL_RE.search(text):
        return "cancel"
    if _CORRECTION_RE.search(text):
        return "correction"
    return "new_request"


def _idempotency_key(tool_name: str, args: dict[str, Any]) -> str:
    canonical = json.dumps(
        {"tool": tool_name, "args": args}, sort_keys=True, default=str
    ).lower()
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


class Coordinator:
    def __init__(
        self,
        registry: ToolRegistry,
        session_id: str = "local",
        trace_path: str | Path | None = None,
        tool_log_path: str | Path | None = None,
    ) -> None:
        self.registry = registry
        self.state = SessionState(session_id=session_id)
        self.trace = TraceLogger(session_id, trace_path)
        self.calls: dict[str, ToolCall] = {}
        self._call_ids = itertools.count(1)
        self._committed: dict[str, dict[str, Any]] = {}
        self._inflight_mutations: dict[str, asyncio.Task] = {}
        self._tool_log_path = tool_log_path
        self._last_user_turn_at: float | None = None

    # ── User turns / interruptions ─────────────────────────────────
    def in_flight(self) -> list[ToolCall]:
        return [c for c in self.calls.values() if c.status == CallStatus.RUNNING]

    def on_user_turn(self, text: str) -> list[ToolCall]:
        """A user turn was committed: new version, supersede in-flight work.

        Returns the calls that were superseded.
        """
        self._last_user_turn_at = time.time()
        self.state.version += 1
        decision = classify_utterance(text)
        self.trace.log("USER_MESSAGE", self.state.version, text=text, decision=decision)

        superseded = self.in_flight()
        if superseded:
            self.trace.log(
                "USER_INTERRUPTION",
                self.state.version,
                superseded_calls=[c.call_id for c in superseded],
            )
        for call in superseded:
            call.status = CallStatus.SUPERSEDED
            self.trace.log(
                "TOOL_CANCEL_REQUESTED",
                self.state.version,
                call_id=call.call_id,
                tool=call.tool_name,
                call_version=call.state_version,
            )
            if call.task and not call.task.done():
                call.task.cancel()

        self.trace.log(
            "STATE_SNAPSHOT", self.state.version, state=self.state.snapshot()
        )
        return superseded

    def on_agent_speaking(self) -> float | None:
        """Record response latency from the end of the last user turn."""
        if self._last_user_turn_at is None:
            return None
        latency_ms = (time.time() - self._last_user_turn_at) * 1000
        self._last_user_turn_at = None
        self.trace.log(
            "RESPONSE_LATENCY", self.state.version, latency_ms=round(latency_ms, 1)
        )
        return latency_ms

    # ── Tool execution ──────────────────────────────────────────────
    async def run_tool(
        self, tool_name: str, raw_args: dict[str, Any]
    ) -> dict[str, Any]:
        """Validate, execute and version-check one tool call.

        Always returns a dict for the LLM; never raises except to propagate
        cancellation from the voice layer.
        """
        version = self.state.version
        spec = self.registry.get(tool_name)
        if spec is None:
            self.trace.log(
                "TOOL_REJECTED", version, tool=tool_name, error="unknown tool"
            )
            return {"status": "error", "error": f"Unknown tool: {tool_name}"}
        try:
            args = spec.validate(raw_args or {})
        except ToolValidationError as e:
            self.trace.log("TOOL_REJECTED", version, tool=tool_name, error=str(e))
            return {"status": "error", "error": f"Invalid arguments: {e}"}

        key = _idempotency_key(tool_name, args) if spec.mutating else None
        if key is not None:
            if key in self._committed:
                self.trace.log(
                    "DUPLICATE_SUPPRESSED", version, tool=tool_name, idempotency_key=key
                )
                return {
                    "status": "already_completed",
                    "note": "This exact action was already completed earlier in "
                    "this session; it was not performed again.",
                    "result": self._committed[key],
                }
            pending = self._inflight_mutations.get(key)
            if pending is not None:
                self.trace.log(
                    "DUPLICATE_JOINED", version, tool=tool_name, idempotency_key=key
                )
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.shield(pending)
                if key in self._committed:
                    return {
                        "status": "already_completed",
                        "note": "An identical request was already in progress; it "
                        "was performed once.",
                        "result": self._committed[key],
                    }

        call = ToolCall(
            call_id=f"call_{next(self._call_ids):03d}",
            tool_name=tool_name,
            state_version=version,
            parameters=args,
            idempotency_key=key,
        )
        self.calls[call.call_id] = call
        self.state.intent = tool_name
        self.state.slots.update(args)
        self.trace.log(
            "TOOL_STARTED",
            version,
            call_id=call.call_id,
            tool=tool_name,
            kind=spec.kind,
            args=args,
            idempotency_key=key,
        )

        call.task = asyncio.create_task(self._execute(call, spec))
        if key is not None:
            self._inflight_mutations[key] = call.task
        try:
            result = await asyncio.shield(call.task)
        except asyncio.CancelledError:
            if call.status == CallStatus.SUPERSEDED:
                # Our own supersede cancelled the inner task; the caller is fine.
                self._finish(call, CallStatus.SUPERSEDED, "TOOL_CANCELLED")
                return self._stale_response(call)
            # The voice layer cancelled us (barge-in): stop the work too.
            call.task.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await call.task
            if call.status == CallStatus.RUNNING:
                self._finish(call, CallStatus.CANCELLED, "TOOL_CANCELLED")
            raise
        except Exception as e:
            self._finish(call, CallStatus.FAILED, "TOOL_ERROR", error=repr(e))
            return {
                "status": "error",
                "error": "The service failed to complete this request.",
            }
        finally:
            if key is not None and self._inflight_mutations.get(key) is call.task:
                del self._inflight_mutations[key]

        if result is None:  # mutation aborted before commit
            return self._stale_response(call)

        if call.state_version != self.state.version:
            if spec.mutating:
                # It really happened; report it truthfully but flag the change.
                self._finish(call, CallStatus.COMPLETED, "RESULT_ACCEPTED_LATE")
                return {
                    **result,
                    "note": "This action completed before the user's latest "
                    "message. Check it still matches what they want now.",
                }
            if call.status == CallStatus.RUNNING:
                call.status = CallStatus.SUPERSEDED
            self.trace.log(
                "STALE_RESULT_REJECTED",
                self.state.version,
                call_id=call.call_id,
                result_version=call.state_version,
                expected_version=self.state.version,
            )
            return self._stale_response(call)

        self._finish(call, CallStatus.COMPLETED, "RESULT_ACCEPTED")
        return result

    async def _execute(self, call: ToolCall, spec: ToolSpec) -> dict[str, Any] | None:
        if spec.latency is not None:
            await spec.latency(spec.name)
        if spec.mutating and call.state_version != self.state.version:
            call.status = CallStatus.ABORTED
            call.ended_at = time.time()
            self.trace.log(
                "MUTATION_ABORTED_STALE",
                self.state.version,
                call_id=call.call_id,
                tool=call.tool_name,
                call_version=call.state_version,
            )
            return None

        t_start = time.time()
        result = await spec.execute(call.parameters)
        t_end = time.time()
        if spec.mutating and call.idempotency_key:
            self._committed[call.idempotency_key] = result
            self.trace.log(
                "MUTATION_COMMITTED",
                self.state.version,
                call_id=call.call_id,
                tool=call.tool_name,
                idempotency_key=call.idempotency_key,
            )
        self.trace.log(
            "TOOL_RESULT",
            self.state.version,
            call_id=call.call_id,
            call_version=call.state_version,
            result=result,
        )
        self._write_tool_log(call, t_start, t_end)
        return result

    def _finish(self, call: ToolCall, status: CallStatus, event: str, **data: Any):
        call.status = status
        call.ended_at = time.time()
        self.trace.log(
            event,
            self.state.version,
            call_id=call.call_id,
            tool=call.tool_name,
            call_version=call.state_version,
            **data,
        )

    def _stale_response(self, call: ToolCall) -> dict[str, Any]:
        done = (
            "was not performed" if call.status == CallStatus.ABORTED else "is outdated"
        )
        return {
            "status": "stale",
            "note": f"The user changed their request while this {call.tool_name} "
            f"call was running, so its result {done}. Follow the user's latest "
            "message instead; call tools again if still needed.",
        }

    def _write_tool_log(self, call: ToolCall, t_start: float, t_end: float) -> None:
        """Per-call log in the format the FDB-v3 harness reads."""
        if not self._tool_log_path:
            return
        try:
            with open(self._tool_log_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "room": self.state.session_id,
                            "call": {
                                "function": call.tool_name,
                                "args": call.parameters,
                                "timestamp_start": t_start,
                                "timestamp_end": t_end,
                            },
                        }
                    )
                    + "\n"
                )
        except OSError as e:
            logger.warning("could not write tool log: %s", e)

    def close(self) -> None:
        self.trace.close()
