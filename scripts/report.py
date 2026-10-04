"""Session report from a coordinator trace (traces/trace_<room>.jsonl).

    uv run python scripts/report.py                   # newest trace
    uv run python scripts/report.py traces/trace_x.jsonl --timeline
    uv run python scripts/report.py --all             # one table per trace

Prints Markdown: response latency, interruptions, tool outcomes (accepted,
stale, aborted, duplicate), vision pipeline metrics and the announcements
VisionGuide made on its own. Everything comes from the JSONL events the
coordinator writes, so a run can be audited after the fact.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACES = ROOT / "traces"


def load(path: Path) -> list[dict]:
    events = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(p * (len(ordered) - 1)))]


def fmt_ms(v: float | None) -> str:
    return "-" if v is None else f"{v:.0f} ms"


def summarize(events: list[dict]) -> dict:
    by_type: dict[str, list[dict]] = defaultdict(list)
    for e in events:
        by_type[e["type"]].append(e)
    t0 = events[0]["ts"] if events else 0.0
    t1 = events[-1]["ts"] if events else 0.0

    latencies = [e["latency_ms"] for e in by_type["RESPONSE_LATENCY"]]
    cancel = [e["cancel_ms"] for e in by_type["INTERRUPTION"]]
    tools: dict[str, Counter] = defaultdict(Counter)
    for e in by_type["TOOL_STARTED"]:
        tools[e["tool"]]["started"] += 1
    for e in by_type["RESULT_ACCEPTED"]:
        tools[e["tool"]]["accepted"] += 1
    for e in by_type["STALE_RESULT_REJECTED"]:
        tools[e.get("tool", "?")]["stale"] += 1
    for e in by_type["MUTATION_ABORTED_STALE"]:
        tools[e["tool"]]["aborted"] += 1
    for e in by_type["MUTATION_COMMITTED"]:
        tools[e["tool"]]["committed"] += 1
    for e in by_type["DUPLICATE_SUPPRESSED"] + by_type["DUPLICATE_JOINED"]:
        tools[e.get("tool", "?")]["duplicate"] += 1
    for e in by_type["TOOL_ERROR"] + by_type["TOOL_REJECTED"]:
        tools[e.get("tool", "?")]["failed"] += 1

    metrics = by_type["SESSION_METRICS"][-1] if by_type["SESSION_METRICS"] else {}
    return {
        "session": events[0]["session_id"] if events else "?",
        "duration_s": t1 - t0,
        "turns": len(by_type["USER_MESSAGE"]),
        "interrupted_turns": len(by_type["USER_INTERRUPTION"]),
        "latency": latencies,
        "cancel": cancel,
        "missed": len(by_type["INTERRUPTION_MISSED"]),
        "false": len(by_type["FALSE_INTERRUPTION"]),
        "tools": tools,
        "vision": metrics.get("vision") or {},
        "overlay_frames": metrics.get("overlay_frames"),
        "announcements": [
            (e["ts"] - t0, e.get("kind", "obstacle"), e.get("urgent"), e["text"])
            for e in by_type["VISION_ANNOUNCEMENT"]
        ],
        "timeline": [
            (e["ts"] - t0, e)
            for e in events
            if e["type"]
            in (
                "USER_MESSAGE",
                "TOOL_STARTED",
                "RESULT_ACCEPTED",
                "STALE_RESULT_REJECTED",
                "MUTATION_ABORTED_STALE",
                "MUTATION_COMMITTED",
                "VISION_ANNOUNCEMENT",
                "INTERRUPTION",
                "RESPONSE_LATENCY",
            )
        ],
    }


def render(s: dict, *, timeline: bool = False) -> str:
    out = [f"## Session `{s['session']}`", ""]
    out.append(
        f"- Duration: {s['duration_s']:.0f} s, user turns: {s['turns']}, "
        f"turns that superseded in-flight work: {s['interrupted_turns']}"
    )
    lat = s["latency"]
    if lat:
        out.append(
            f"- Response latency (end of user turn → agent speaking), n={len(lat)}: "
            f"mean {statistics.mean(lat):.0f} ms, p50 {pct(lat, 0.5):.0f} ms, "
            f"p90 {pct(lat, 0.9):.0f} ms, max {max(lat):.0f} ms"
        )
    else:
        out.append("- Response latency: no spoken turns recorded")
    cancel = s["cancel"]
    out.append(
        f"- Barge-ins: {len(cancel)} handled"
        + (
            f" (agent stopped within mean {statistics.mean(cancel):.0f} ms, "
            f"p90 {pct(cancel, 0.9):.0f} ms)"
            if cancel
            else ""
        )
        + f", {s['missed']} missed, {s['false']} false (auto-resumed)"
    )
    out.append("")
    if s["tools"]:
        out += [
            "| Tool | Calls | Accepted | Stale rejected | Aborted pre-commit | Committed | Duplicate suppressed | Failed |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for name, c in sorted(s["tools"].items()):
            out.append(
                f"| {name} | {c['started']} | {c['accepted']} | {c['stale']} | "
                f"{c['aborted']} | {c['committed']} | {c['duplicate']} | {c['failed']} |"
            )
        out.append("")
    v = s["vision"]
    if v:
        out += ["### Vision", ""]
        out.append(
            f"- Device: {v.get('device')}, model load: {v.get('load_s')} s, "
            f"state at end: {v.get('state')}, restarts: {v.get('restarts', 0)}"
        )
        out.append(
            f"- Frames sent: {v.get('frames_sent', 0)}, scenes: {v.get('scenes', 0)}, "
            f"fps: {v.get('fps')}, stale-frame rate: {v.get('stale_frame_rate')}"
        )
        out.append(
            f"- Latency per frame: inference {fmt_ms(v.get('avg_inference_ms'))}, "
            f"queue {fmt_ms(v.get('avg_queue_ms'))}, camera→scene {fmt_ms(v.get('avg_end_to_end_ms'))}"
        )
        if s["overlay_frames"] is not None:
            out.append(f"- Overlay frames streamed back: {s['overlay_frames']}")
        out.append("")
    if s["announcements"]:
        out += [
            "### Unprompted announcements",
            "",
            "| t | kind | urgent | said |",
            "| ---: | --- | :-: | --- |",
        ]
        for t, kind, urgent, text in s["announcements"]:
            out.append(f"| {t:6.1f}s | {kind} | {'yes' if urgent else ''} | {text} |")
        out.append("")
    if timeline:
        out += ["### Timeline", ""]
        for t, e in s["timeline"]:
            kind = e["type"]
            if kind == "USER_MESSAGE":
                detail = f"user (v{e['state_version']}, {e.get('decision')}): {e.get('text')!r}"
            elif kind == "TOOL_STARTED":
                detail = f"→ {e['tool']}({json.dumps(e.get('args', {}))})"
            elif kind == "RESPONSE_LATENCY":
                detail = f"agent speaking after {e['latency_ms']:.0f} ms"
            elif kind == "VISION_ANNOUNCEMENT":
                detail = f"VisionGuide: {e['text']!r}"
            elif kind == "INTERRUPTION":
                detail = f"barge-in, stopped in {e['cancel_ms']:.0f} ms"
            else:
                detail = f"{kind.lower()} {e.get('tool', '')}".strip()
            out.append(f"- {t:7.1f}s  {detail}")
        out.append("")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("trace", nargs="?", help="trace file (default: newest)")
    parser.add_argument("--all", action="store_true", help="report every trace")
    parser.add_argument(
        "--timeline", action="store_true", help="include the event timeline"
    )
    parser.add_argument("--out", help="write Markdown here instead of stdout")
    args = parser.parse_args()

    if args.all:
        paths = sorted(TRACES.glob("trace_*.jsonl"), key=lambda p: p.stat().st_mtime)
    elif args.trace:
        paths = [Path(args.trace)]
    else:
        candidates = sorted(
            TRACES.glob("trace_*.jsonl"), key=lambda p: p.stat().st_mtime
        )
        if not candidates:
            print(f"no traces in {TRACES}", file=sys.stderr)
            return 1
        paths = [candidates[-1]]

    parts = []
    for path in paths:
        events = load(path)
        if not events:
            continue
        parts.append(render(summarize(events), timeline=args.timeline))
    text = "\n".join(parts)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
