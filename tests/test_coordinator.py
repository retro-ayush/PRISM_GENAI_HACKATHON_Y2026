"""Deterministic tests for the coordination layer.

Tool latency is simulated with gates (asyncio.Event) instead of sleeps, so each
test controls exactly when a result arrives relative to user interruptions.
"""

import asyncio
import json

import pytest

from coordinator import CallStatus, Coordinator, classify_utterance
from latency_injector import LatencyInjector
from mock_apis import register_mock_apis
from tool_registry import ToolRegistry, ToolSpec

OBJ = {"type": "object", "properties": {}, "required": []}


class Gate:
    """A controllable latency: each call blocks until released."""

    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self, _name: str) -> None:
        self.started.set()
        await self.release.wait()


def search_spec(gate: Gate | None = None) -> ToolSpec:
    return ToolSpec(
        name="search_flights",
        description="search",
        kind="read_only",
        parameters={
            "type": "object",
            "properties": {"destination": {"type": "string"}},
            "required": ["destination"],
        },
        handler=lambda destination: {"route": f"Chennai -> {destination}"},
        latency=gate,
    )


def booking_spec(commits: list, gate: Gate | None = None) -> ToolSpec:
    def book(passenger_name: str) -> dict:
        commits.append(passenger_name)
        return {"booking_ref": f"B{len(commits)}"}

    return ToolSpec(
        name="book_flight",
        description="book",
        kind="state_modifying",
        parameters={
            "type": "object",
            "properties": {"passenger_name": {"type": "string"}},
            "required": ["passenger_name"],
        },
        handler=book,
        latency=gate,
    )


def make(*specs: ToolSpec) -> Coordinator:
    registry = ToolRegistry()
    for spec in specs:
        registry.register(spec)
    return Coordinator(registry, session_id="test")


async def test_normal_request() -> None:
    coord = make(search_spec())
    coord.on_user_turn("Find flights to Delhi")
    result = await coord.run_tool("search_flights", {"destination": "Delhi"})
    assert result == {"route": "Chennai -> Delhi"}
    assert coord.trace.of_type("RESULT_ACCEPTED")
    assert coord.state.slots["destination"] == "Delhi"


async def test_interruption_supersedes_and_rejects_old_result() -> None:
    """The core demo: Delhi search in flight, user says 'actually Mumbai'."""
    gate = Gate()
    coord = make(search_spec(gate))
    coord.on_user_turn("Find flights Chennai to Delhi")
    delhi = asyncio.create_task(
        coord.run_tool("search_flights", {"destination": "Delhi"})
    )
    await gate.started.wait()

    superseded = coord.on_user_turn("Wait! Actually Mumbai.")
    assert [c.tool_name for c in superseded] == ["search_flights"]

    gate.release.set()
    old = await delhi
    assert old["status"] == "stale"
    assert "Delhi" not in json.dumps(old)

    new = await coord.run_tool("search_flights", {"destination": "Mumbai"})
    assert new == {"route": "Chennai -> Mumbai"}
    assert coord.state.version == 2
    assert coord.state.slots["destination"] == "Mumbai"
    statuses = {c.parameters["destination"]: c.status for c in coord.calls.values()}
    assert statuses == {"Delhi": CallStatus.SUPERSEDED, "Mumbai": CallStatus.COMPLETED}


async def test_late_result_rejected_when_not_cancelled() -> None:
    """Even if cancellation can't stop the work, the version check rejects it."""
    coord = make(search_spec())
    coord.on_user_turn("Find flights to Delhi")

    async def late_result() -> None:
        # Simulates a result arriving after a new turn, with no in-flight
        # cancellation possible (e.g. a remote API that already answered).
        call_task = asyncio.create_task(
            coord.run_tool("search_flights", {"destination": "Delhi"})
        )
        await asyncio.sleep(0)  # tool starts, inner task scheduled
        coord.state.version += 1  # newer intent without cancelling the call
        return await call_task

    result = await late_result()
    assert result["status"] == "stale"
    rejected = coord.trace.of_type("STALE_RESULT_REJECTED")
    assert rejected[0]["result_version"] == 1
    assert rejected[0]["expected_version"] == 2


async def test_multiple_interruptions_only_latest_counts() -> None:
    gates = [Gate(), Gate(), Gate()]
    coord = make(search_spec())
    tasks = []
    for gate, city in zip(gates, ["Delhi", "Mumbai", "Bangalore"], strict=True):
        coord.registry.get("search_flights").latency = gate
        coord.on_user_turn(f"Actually {city}")
        tasks.append(
            asyncio.create_task(coord.run_tool("search_flights", {"destination": city}))
        )
        await gate.started.wait()
    for gate in gates:
        gate.release.set()
    results = await asyncio.gather(*tasks)

    assert [r.get("status") for r in results[:2]] == ["stale", "stale"]
    assert results[2] == {"route": "Chennai -> Bangalore"}
    assert coord.state.version == 3
    assert coord.state.slots["destination"] == "Bangalore"


async def test_duplicate_mutation_executes_once() -> None:
    commits: list = []
    coord = make(booking_spec(commits))
    coord.on_user_turn("Book it for Ayush")
    first = await coord.run_tool("book_flight", {"passenger_name": "Ayush"})
    second = await coord.run_tool("book_flight", {"passenger_name": "Ayush"})

    assert commits == ["Ayush"]
    assert first == {"booking_ref": "B1"}
    assert second["status"] == "already_completed"
    assert second["result"] == {"booking_ref": "B1"}
    assert coord.trace.of_type("DUPLICATE_SUPPRESSED")


async def test_concurrent_duplicate_mutation_executes_once() -> None:
    commits: list = []
    gate = Gate()
    coord = make(booking_spec(commits, gate))
    coord.on_user_turn("Book it for Ayush")
    a = asyncio.create_task(coord.run_tool("book_flight", {"passenger_name": "Ayush"}))
    await gate.started.wait()
    b = asyncio.create_task(coord.run_tool("book_flight", {"passenger_name": "Ayush"}))
    await asyncio.sleep(0)
    gate.release.set()
    results = await asyncio.gather(a, b)

    assert commits == ["Ayush"]
    assert results[0] == {"booking_ref": "B1"}
    assert results[1]["status"] == "already_completed"


async def test_mutation_not_committed_after_user_cancels() -> None:
    """'Book it' → booking in flight → 'Wait, don't book' → no booking."""
    commits: list = []
    gate = Gate()
    coord = make(booking_spec(commits, gate))
    coord.on_user_turn("Book the flight for Ayush")
    booking = asyncio.create_task(
        coord.run_tool("book_flight", {"passenger_name": "Ayush"})
    )
    await gate.started.wait()

    coord.on_user_turn("Wait, don't book it, just tell me the price")
    gate.release.set()
    result = await booking

    assert commits == []
    assert result["status"] == "stale"
    assert not coord.trace.of_type("MUTATION_COMMITTED")


async def test_mutation_aborted_if_intent_changes_before_commit() -> None:
    """Even without task cancellation, the pre-commit version check aborts."""
    commits: list = []
    coord = make(booking_spec(commits))

    async def bump_version(_name: str) -> None:
        coord.state.version += 1

    coord.registry.get("book_flight").latency = bump_version
    coord.on_user_turn("Book for Ayush")
    result = await coord.run_tool("book_flight", {"passenger_name": "Ayush"})

    assert commits == []
    assert result["status"] == "stale"
    assert coord.trace.of_type("MUTATION_ABORTED_STALE")


async def test_voice_layer_cancellation_stops_mutation() -> None:
    """A barge-in cancels LiveKit's tool task; the commit must not happen."""
    commits: list = []
    gate = Gate()
    coord = make(booking_spec(commits, gate))
    coord.on_user_turn("Book for Ayush")
    booking = asyncio.create_task(
        coord.run_tool("book_flight", {"passenger_name": "Ayush"})
    )
    await gate.started.wait()
    booking.cancel()
    with pytest.raises(asyncio.CancelledError):
        await booking
    gate.release.set()
    await asyncio.sleep(0)

    assert commits == []
    (call,) = coord.calls.values()
    assert call.status == CallStatus.CANCELLED


async def test_tool_failure_is_reported_not_raised() -> None:
    def broken(**_kwargs):
        raise TimeoutError("backend timeout")

    coord = make(
        ToolSpec(
            name="lookup",
            description="x",
            kind="read_only",
            parameters=OBJ,
            handler=broken,
        )
    )
    coord.on_user_turn("look it up")
    result = await coord.run_tool("lookup", {})
    assert result["status"] == "error"
    (call,) = coord.calls.values()
    assert call.status == CallStatus.FAILED
    assert coord.trace.of_type("TOOL_ERROR")


async def test_invalid_and_unknown_tools_rejected() -> None:
    commits: list = []
    coord = make(booking_spec(commits))
    coord.on_user_turn("book")
    missing = await coord.run_tool("book_flight", {})
    wrong_type = await coord.run_tool("book_flight", {"passenger_name": 42})
    unknown = await coord.run_tool("launch_rocket", {})
    assert missing["status"] == wrong_type["status"] == unknown["status"] == "error"
    assert commits == []
    assert len(coord.trace.of_type("TOOL_REJECTED")) == 3


async def test_dynamic_tool_loaded_from_json(tmp_path) -> None:
    schema = [
        {
            "name": "create_ticket",
            "description": "Open a support ticket",
            "kind": "state_modifying",
            "parameters": {
                "type": "object",
                "properties": {"summary": {"type": "string"}},
                "required": ["summary"],
            },
            "mock_response": {"status": "success", "ticket_id": "T001"},
        }
    ]
    path = tmp_path / "tools.json"
    path.write_text(json.dumps(schema))
    registry = ToolRegistry()
    assert registry.load_json(path) == ["create_ticket"]
    coord = Coordinator(registry)
    coord.on_user_turn("open a ticket")
    result = await coord.run_tool("create_ticket", {"summary": "printer jam"})
    assert result["ticket_id"] == "T001"
    again = await coord.run_tool("create_ticket", {"summary": "printer jam"})
    assert again["status"] == "already_completed"


async def test_trace_is_jsonl_and_tool_log_written(tmp_path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    tool_log = tmp_path / "tools.log"
    registry = register_mock_apis(ToolRegistry(), LatencyInjector("instant"))
    coord = Coordinator(
        registry, "room1", trace_path=trace_path, tool_log_path=tool_log
    )
    coord.on_user_turn("Track order 55")
    await coord.run_tool("track_order", {"order_id": "55"})
    coord.close()

    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    types = [e["type"] for e in events]
    assert types[0] == "USER_MESSAGE"
    assert "TOOL_STARTED" in types and "RESULT_ACCEPTED" in types
    assert [e["seq"] for e in events] == list(range(1, len(events) + 1))
    logged = json.loads(tool_log.read_text())
    assert logged["room"] == "room1"
    assert logged["call"]["function"] == "track_order"


async def test_mock_latency_does_not_block_event_loop() -> None:
    registry = register_mock_apis(ToolRegistry(), LatencyInjector("fast"))
    coord = Coordinator(registry)
    coord.on_user_turn("track")
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while True:
            ticks += 1
            await asyncio.sleep(0.01)

    t = asyncio.create_task(ticker())
    await coord.run_tool("track_order", {"order_id": "1"})
    t.cancel()
    assert ticks >= 3  # loop kept running during the 50-200 ms delay


def test_all_mock_tools_have_valid_schemas() -> None:
    registry = register_mock_apis(ToolRegistry())
    assert len(registry) == 12
    kinds = {s.name: s.kind for s in registry}
    assert kinds["book_flight"] == "state_modifying"
    assert kinds["search_flights"] == "read_only"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Wait! Actually Mumbai.", "correction"),
        ("Don't book it, just tell me the price", "cancel"),
        ("Find flights to Delhi", "new_request"),
    ],
)
def test_classify_utterance(text: str, expected: str) -> None:
    assert classify_utterance(text) == expected
