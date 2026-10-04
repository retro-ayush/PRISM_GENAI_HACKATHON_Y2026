# Vision: Interruptible Agent — Full-Duplex Voice Assistant

**A full-duplex voice agent that can listen, respond, get interrupted, recover from stale intent, and safely execute actions.**

This project is a LiveKit-based Python voice agent built for **Full-Duplex-Bench v3**. The main focus is not just getting a voice response out quickly, but handling the messy parts of real conversations: interruptions, corrections, tool calls, and actions that may already be in progress.

It also includes **VisionGuide**, an end-to-end voice + vision extension that lets the agent query the user's current surroundings through a camera pipeline.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [How Full-Duplex / Interruption Handling Works](#how-full-duplex--interruption-handling-works)
- [Tool & Action Architecture](#tool--action-architecture)
- [Model / Provider Details](#model--provider-details)
- [FDB-v3 Benchmark](#fdb-v3-benchmark)
- [Benchmark Results](#benchmark-results)
- [Extension Use Case](#extension-use-case)
- [Installation](#installation)
- [Configuration / API Keys](#configuration--api-keys)
- [Running the Agent](#running-the-agent)
- [One-Command Reproduction](#one-command-reproduction)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Reproducibility](#reproducibility)
- [Limitations](#limitations)
- [Demo Video](#demo-video)
- [Team / Credits](#team--credits)
- [References](#references)

---

## Project Overview

Most voice agents work like this:

```text
User speaks
    ↓
STT
    ↓
LLM
    ↓
TTS
    ↓
Agent speaks
```

That works for simple conversations, but real users do not always wait quietly for an agent to finish.

Our agent is built around a different idea:

```text
User
 │
 │ speaks
 ▼
┌───────────────────────────────────────┐
│        Full-Duplex Voice Agent        │
│                                       │
│  Listen ──► Reason ──► Act ──► Speak │
│      ▲             │          │       │
│      └── interrupt ┘          │       │
│                               │       │
│                 Tools ◄───────┘       │
└───────────────────────────────────────┘
```

The agent can continue working while the conversation changes, rather than treating every turn as an isolated request.

---

## Problem Statement

Traditional voice pipelines have a few practical problems:

- The user may interrupt while the agent is speaking.
- The user's latest statement may invalidate an action that was already being prepared.
- Tool calls can take time while the user continues talking.
- A stale response can be spoken after the user has already changed their request.
- Repeating or retrying an action can accidentally execute a mutation twice.

The goal of this project is to make voice interaction behave more like a real conversation:

> **The latest user intent should matter, stale work should be discarded, and state-changing actions should not accidentally happen twice.**

---

## Key Features

| Feature | Status |
| --- | --- |
| Real-time LiveKit voice agent | ✅ |
| Full-duplex interaction | ✅ |
| User interruption handling | ✅ |
| Async tool execution | ✅ |
| Stale-intent recovery | ✅ |
| Exactly-once mutation handling | ✅ |
| Function/tool calling | ✅ |
| Tool latency simulation | ✅ |
| Latency and tool-call tracking | ✅ |
| Self-correction handling | ✅ |
| Multi-step tool interactions | ✅ |
| VisionGuide extension | ✅ |
| YOLO26 detection + YOLO26 depth (approximate metres) | ✅ |
| LiveKit camera track → isolated vision process | ✅ |
| Latest-frame queue (no backlog) | ✅ |
| Scene memory + spoken obstacle announcements | ✅ |
| Voice access to current scene | ✅ |
| Interruption + vision latency metrics | ✅ |
| Free-space analysis: which way to go, pass left/right | ✅ |
| Wall and drop-off detection from the depth map | ✅ |
| Fast-approaching object warning (bike, car, runner) | ✅ |
| Two-tier perception: YOLO26 reflexes + vision-language model for text, colours, descriptions | ✅ |
| Find an object by name ("where is my phone?") | ✅ |
| Guide-to-target mode ("guide me to the chair") | ✅ |
| Guidance modes: quiet / normal / verbose / pause | ✅ |
| Live "agent vision" overlay streamed back into the room | ✅ |
| Alert tone before urgent warnings | ✅ |
| Language switching mid-call (Hindi, Spanish, ...) | ✅ |
| Session report from the event trace | ✅ |
| Full-Duplex-Bench v3 evaluation | ✅ |

---

## Architecture

```text
                         ┌──────────────────┐
                         │       USER       │
                         │   Voice Input    │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │       VAD        │
                         │     Silero       │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │       STT        │
                         │  Deepgram Nova-3 │
                         └────────┬─────────┘
                                  │
                                  ▼
                    ┌──────────────────────────┐
                    │          LLM             │
                    │   GPT-4o-mini           │
                    │   LiveKit Inference      │
                    └───────────┬──────────────┘
                                │
                    ┌───────────┼───────────┐
                    │           │           │
                    ▼           ▼           ▼
               ┌────────┐ ┌──────────┐ ┌───────────┐
               │ Tools  │ │  Vision  │ │  Intent   │
               │ / APIs │ │  Guide   │ │  Recovery │
               └───┬────┘ └────┬─────┘ └─────┬─────┘
                   │            │             │
                   ▼            ▼             ▼
              Mock APIs   YOLO26 + depth  Latest Intent
                   │    (vision process)       │
                   └────────────┬─────────────┘
                                │
                                ▼
                         ┌──────────────┐
                         │     TTS      │
                         │ Cartesia     │
                         │  Sonic-3     │
                         └──────┬───────┘
                                │
                                ▼
                              USER
```

---

## How Full-Duplex / Interruption Handling Works

The important difference is that the agent does not treat the conversation as:

```text
WAIT → PROCESS → SPEAK → FINISH → LISTEN
```

Instead, processing can overlap with the conversation:

```text
User speaks
     │
     ├──────────────► STT
     │                  │
     │                  ▼
     │               LLM starts
     │                  │
     │                  ├──────► Tool call
     │                  │
User interrupts ────────┘
     │
     ▼
Latest intent becomes authoritative
     │
     ├── cancel / ignore stale work
     └── continue with updated request
```

### Interruption flow

1. User starts speaking.
2. VAD detects the new speech.
3. The agent can stop or invalidate the current response.
4. The new utterance is processed.
5. If the new request changes the previous intent, stale work is not allowed to become the final response.
6. The agent continues from the newest valid state.

This is especially important when a tool call is already running.

---

## Tool & Action Architecture

The agent uses tools for actions that should not be handled as plain LLM text.

```text
                    LLM
                     │
              chooses a tool
                     │
                     ▼
              ┌──────────────┐
              │ Action Layer │
              └──────┬───────┘
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       Travel     Finance    Housing
          │          │          │
          └──────────┼──────────┘
                     ▼
                E-commerce
```

The repository currently uses simulated APIs for travel, finance, housing, and e-commerce operations.

These mock APIs are useful for testing tool behaviour and latency without performing real transactions.

### Coordination Layer

Every tool call goes through a deterministic coordinator (`src/coordinator.py`). The LLM only chooses tools and arguments; the coordinator owns state:

| Responsibility | How |
| --- | --- |
| Versioned state | Each committed user turn increments `state.version`; slots are updated from accepted tool arguments |
| Task tracking | Every call gets a `call_id`, is bound to the version it started under, and moves through `RUNNING → COMPLETED / SUPERSEDED / CANCELLED / ABORTED / FAILED` |
| Interruption | A new user turn supersedes and cancels in-flight calls; a barge-in that cancels LiveKit's tool task also cancels the underlying work |
| Stale-result check | `result.version != current.version` → rejected, returned to the LLM as `status: "stale"` |
| Idempotency | State-modifying calls are keyed by tool + arguments and committed at most once per session |
| Schema validation | Arguments are checked against each tool's JSON schema before anything runs |
| Event trace | Every transition is written to `traces/trace_<room>.jsonl` |

Tools are schema-driven (`src/tool_registry.py`): each has a JSON schema and a `kind` (`read_only` or `state_modifying`). Extra tools can be loaded at runtime from a JSON file with `EXTRA_TOOLS_FILE`.

### Stale Intent

A tool result is not automatically trusted just because it finishes successfully.

The agent checks whether the result still belongs to the current user intent.

```text
Request A
   ↓
Tool A starts
   ↓
User says "Actually, do B"
   ↓
Intent changes
   ↓
Tool A result becomes stale
   ↓
Do not use stale result as the final action
```

Mutations get one more check: after the backend delay and **before** the commit, the coordinator re-checks the version. "Book it… wait, don't book it" therefore never produces a booking, even if the tool task wasn't cancelled.

### Exactly-Once Mutations

For state-changing actions, the goal is:

```text
One user action
      ↓
One committed mutation
```

rather than:

```text
retry → retry → duplicate action
```

This is important for any future real-world integration involving bookings, purchases, account changes, or other state-changing operations.

---

## Model / Provider Details

| Component | Provider / Model |
| --- | --- |
| Voice infrastructure | LiveKit Agents |
| VAD | Silero |
| STT | Deepgram Nova-3 |
| LLM | OpenAI GPT-4o-mini via LiveKit Inference |
| Coordination | Deterministic Python (`asyncio`), no model in the loop |
| TTS | Cartesia Sonic-3 |
| Noise cancellation | LiveKit Background Noise Cancellation |
| Turn detection | LiveKit turn detector (`inference.TurnDetector`) + preemptive generation |
| Vision detection | Ultralytics YOLO26n (`yolo26n.pt`) |
| Vision depth | Ultralytics YOLO26n-depth (`yolo26n-depth.pt`, metric depth map) |
| Camera processing | LiveKit `VideoStream` + OpenCV (JPEG IPC) |

The project does **not** depend on the older Gemma/Fish Audio configuration from the original starter setup.

---

## FDB-v3 Benchmark

The project was built and evaluated using **Full-Duplex-Bench v3**.

The benchmark is useful because normal chatbot testing does not fully capture what happens when users:

- interrupt the agent,
- correct themselves,
- change intent,
- trigger multiple tools,
- or continue speaking while an action is running.

The evaluation process was used to identify failures and iterate on:

- interruption handling,
- self-correction,
- tool chains,
- stale responses,
- and conversational latency.

---

## Benchmark Results

Current recorded benchmark run:

| Metric | Result |
| --- | ---: |
| Scenarios tested | **100** |
| Task completion | **88 / 100** |
| Recorded latency | **381 ms** |

These numbers are from the current benchmark run and should be treated as the result of this version of the agent, not as a universal performance guarantee.

The benchmark work also exposed the main remaining issues: interruption quality, fragmented STT, and failures in some complex multi-step interactions.

---

## Extension Use Case

### VisionGuide — Voice + Vision Assistant

VisionGuide lets the voice agent see through the user's **LiveKit camera
track** and act as a walking companion for blind and visually impaired users:
it warns about obstacles, says which way to go, finds things, reads signs,
and shows a sighted helper exactly what it sees.

```text
                 VOICE AGENT PROCESS (never imports torch)
 Camera ──► LiveKit VideoStream (RGB24)
               │  throttle (8 fps) · resize to 640px · JPEG   (worker thread)
               ├──────────────────────────────► latest frame ──► describe_view (VLM)
               ▼
        frame queue (maxsize=1: newest frame wins, old frames dropped)
               │
               ▼        VISION PROCESS (multiprocessing "spawn")
        JPEG decode ─► YOLO26 detection + YOLO26 depth ─► scene JSON + overlay JPEG
               │
               ▼
        result queue ─► latest_scene ─┬─► get_current_scene / find_object / guide_to
                                      ├─► SceneMemory ─► alert tone + session.say()
                                      └─► OverlayPublisher ─► video track back into the room
```

**Why a separate process.** Initialising PyTorch, CUDA/MPS and the YOLO models
inside the LiveKit process blocked its event loop for over 20 seconds. The
vision worker is spawned as its own interpreter, which gives it its own GPU
context, so loading, inference, slow frames and crashes never touch STT, the
LLM, TTS or interruptions. The worker starts only when a camera track arrives,
so voice-only sessions (for example FDB-v3 runs) never load the models. If the
worker crashes, the bridge restarts it with backoff.

**Two tiers of perception.** The *fast* tier (YOLO26 detection + depth, ~30 ms
a frame) drives every safety warning and direction. The *slow* tier
(`describe_view`) sends the latest camera frame to a vision-language model
(gpt-4o-mini through LiveKit Inference, 3–5 s) for what a detector cannot do:
a detailed description, reading a sign, label or screen, colours, what is on
the table. It runs as a separate request, so the conversation never carries
images and the main agent stays fast. Questions that need detail ("read the
sign", "what colour is…", "describe what's around me") are routed to it
deterministically.

**Scene format** (what the LLM sees, trimmed to the essentials):

```json
{
  "guidance": "Warning: a chair directly ahead, about 1 metre away. Go to your left.",
  "objects": [
    {"label": "chair", "position": "center", "clock": "12 o'clock", "distance_m": 1.2, "proximity": "close", "pass_on": "left"},
    {"label": "person", "position": "left", "clock": "11 o'clock", "distance_m": 3.4, "proximity": "medium"}
  ],
  "clearance_m": {"left": 3.1, "center": 1.2, "right": 2.0},
  "wall_ahead": false,
  "recommendation": "step_left",
  "other_objects": [{"label": "laptop", "position": "right", "distance_m": 1.5}]
}
```

Distance is the median of the depth map over the inner half of each box, which
ignores background pixels at the box edges. Proximity buckets are under 1 m,
under 2 m and under 4 m. Without depth, the bucket falls back to bounding-box
size. `clock` is the orientation-and-mobility clock face (12 is straight
ahead); `elevation` (high / level / low) is kept for "where is my phone".

**Which way to go.** The depth map is split into left, centre and right
columns; the free distance in each is a low percentile of the body-height band
(rows 25–70%, so the floor near the feet never counts as an obstacle). The
`recommendation` is `straight`, `step_left`, `step_right`, `stop` (blocked on
every side) or `stop_drop`. For an object in the path, `pass_on` picks the
side without another object nearby, then the side with more room, then the
side away from where the object stands.

**Walls and drop-offs.** Walls have no detector class, so they are found in
the depth map: a near surface that fills the centre column at roughly the same
distance from top to bottom. The floor gets steadily closer toward the bottom
of the frame, so it does not qualify. Downward stairs, kerbs and platform
edges show up as the receding floor suddenly jumping far away; an obstacle's
top edge also jumps, but the bands below it are flat, so it is not reported.
Drop-off detection is a cue, not a measurement; the agent says "stop and check
with your cane".

**Fast-approaching objects.** Scene memory keeps a short distance history per
object. Something closing at more than 2 m/s (far faster than the user walking
toward a static object at ~1.3 m/s) triggers "Careful, a bicycle is coming
toward you fast from your left."

**Announcements the agent makes on its own.** `SceneMemory` compares each
scene with what has already been said. An obstacle that appears in the path
and stays for 2 frames is announced once, with which way to pass it. It is
announced again only when it moves into a closer bucket. After a stop, the
path opening up again is mentioned once. Unchanged, receding or off-path
objects stay silent. Urgent warnings (very close obstacle, stop, drop-off,
fast approach) are preceded by a short two-tone alert and may cut the agent's
own sentence short; everything else waits until nobody is talking. Modes:
`quiet` (urgent only), `normal`, `verbose` (also objects passing close on
either side), `pause`/`resume`, all changeable by voice.

**Guide-to-target.** "Guide me to the chair" starts a mode in which the
chair's direction is repeated every 3 s or whenever it changes ("The chair is
on your left, about 3 metres away. Turn a little to your left." … "The chair
is just ahead, about 1.5 metres away. Keep going slowly." … "You've reached
the chair; it's right in front of you, within arm's reach."). If it leaves the
view the agent asks the user to turn slowly, once.

**Agent-vision overlay.** The worker draws what the agent sees (boxes with
distances and pass side, free space per column, direction arrow, drop-off
banner, the current guidance) and the agent publishes it back into the room
as a video track named `visionguide`. In the Agents Playground it appears as
the agent's camera. A sighted helper can follow along, and it makes the
reasoning auditable.

**Language.** "Speak Hindi from now on" switches the TTS and STT language
mid-call (Hindi, Spanish, French, German, Italian, Portuguese, Dutch, Japanese,
Russian) and the agent replies in that language; tool arguments stay English.

**Example**

```text
User:  "Which way should I go?"
Agent: "There's a person directly ahead, about 2 metres away. Step to your left to pass."
       (walking)  ♪♪ "Careful, a wall directly ahead, less than a metre away. Go to your right."
User:  "Where is my phone?"
Agent: "Your phone is on your right, about 1 metre away, low down."
User:  "Read the sign in front of me."
Agent: "The sign reads 'M1 Sol–Sevilla'. Below it: 'cero emisiones'."
User:  "Guide me to the chair."
Agent: "The chair is on your left, about 3 metres away. Turn a little to your left." …
```

**Measured locally (Apple Silicon, MPS, 640px frames):**

| Stage | Time |
| --- | ---: |
| YOLO26n detection | ~11 ms |
| YOLO26n depth | ~21 ms |
| Free-space / wall / drop-off analysis | < 1 ms |
| Overlay drawing + JPEG | 1–5 ms |
| Full LiveKit frame → scene (encode + IPC + inference) | ~98 ms at the 8 fps cap |
| Model load in worker (agent loop stall during load) | ~3 s (worst loop lag 1.2 ms) |
| `describe_view` (vision-language model, description / reading text) | 3.3 s / 4.8 s |

On CPU the same pipeline runs at about 112 ms per frame.

> VisionGuide is experimental and is not intended for safety-critical navigation.

---

## Installation

### Prerequisites

- Python
- LiveKit project
- LiveKit credentials
- Required provider API credentials
- Webcam if using VisionGuide

### Clone

```bash
git clone <your-repository-url>
cd interruptible-agent
```

### Install dependencies

The project targets Python 3.10–3.12 and uses `uv`:

```bash
uv sync                 # voice agent
uv sync --extra vision  # also install OpenCV + Ultralytics for VisionGuide
```

---

## Configuration / API Keys

Create:

```text
.env
```

using:

```text
.env.example
```

as the reference.

The environment should contain the credentials required by the LiveKit and model providers used by the agent.

Do **not** commit `.env` or API keys.

Typical provider configuration includes:

```text
LIVEKIT_URL=...
LIVEKIT_API_KEY=...
LIVEKIT_API_SECRET=...
```

STT, LLM and TTS run through LiveKit Inference, so no separate provider keys are needed.

Optional settings:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MOCK_LATENCY_PROFILE` | `instant` | Mock API delay: `instant`, `fast`, `normal`, `slow`, `degraded`, `timeout_risk`, or `realistic` (per-API delays) |
| `TRACE_DIR` | `traces/` | Where JSONL event traces are written |
| `AGENT_TOOL_LOG` | `/tmp/agent_tool_calls.log` | Per-call log read by the FDB-v3 harness |
| `EXTRA_TOOLS_FILE` | — | JSON list of extra tool schemas to register |
| `SCENE_MAX_AGE_S` | `5` | VisionGuide snapshots older than this are reported as stale |
| `VISION_MODE` | `livekit` | `livekit` (camera track + vision process), `fake` (torch-free stand-in worker) or `off` |
| `VISION_ANNOUNCE` | `1` | Speak new or approaching obstacles without being asked |
| `VISION_DEVICE` | `auto` | `auto` (CUDA, then MPS, then CPU), `cuda:0`, `mps` or `cpu` |
| `VISION_MAX_FPS` | `8` | Frames per second sent to the vision process |
| `VISION_DETECT_MODEL` / `VISION_DEPTH_MODEL` | `yolo26n.pt` / `yolo26n-depth.pt` | Weights, downloaded into `models/` on first use. An empty depth model disables depth |
| `INTERRUPT_MIN_S` | `0.3` | Minimum user speech that interrupts the agent. False interruptions resume automatically |
| `VISION_OVERLAY` | `1` | Publish the annotated camera view back into the room as a video track |
| `VISION_ALERT_TONE` | `1` | Two-tone alert before urgent warnings |
| `VLM_MODEL` | `openai/gpt-4o-mini` | Vision-language model behind `describe_view` (any LiveKit Inference model that accepts images) |
| `STT_LANGUAGE` | `en` | Deepgram language; `multi` understands English/Hindi code-switching (keyterm boosting is English-only) |
| `BENCHMARK_MODE` | `0` | `1` for Full-Duplex-Bench runs: no greeting, no VisionGuide or noise cancellation, `gpt-4o`, longer end-of-turn wait |
| `LLM_MODEL` | `openai/gpt-4o-mini` (`openai/gpt-4o` in benchmark mode) | LiveKit Inference LLM used for conversation and tool calls |
| `ENDPOINT_MIN_S` / `ENDPOINT_MAX_S` | LiveKit default (`1.5` / `6.0` in benchmark mode) | Silence before a turn ends when the user sounds finished / mid-sentence |

---

## Running the Agent

The main entry point is:

```text
src/agent.py
```

Run the local agent with:

```bash
uv run src/agent.py dev
```

The agent connects to LiveKit and starts the voice session.

### Running VisionGuide

Install the vision extra (`uv sync --extra vision`), start the agent, and join
the room with **camera and microphone** enabled, for example from the
[Agents Playground](https://agents-playground.livekit.io/). The agent picks up
the camera track automatically and spawns the vision process. Without the
extra installed, the agent runs voice-only and logs a warning.

**Offline / local webcam fallback.** Without a LiveKit camera, you can run the
same engine on a local webcam:

```bash
uv run python vision_guide/vision.py            # --no-window, --camera N, --device mps
```

It writes `vision_guide/scene.json`, which `get_current_scene` reads whenever no
LiveKit camera is connected.

---

## One-Command Reproduction

After installing dependencies and configuring `.env`:

```bash
uv run src/agent.py dev
```

This starts the main voice-agent workflow.

For the complete benchmark reproduction, clone [Full-Duplex-Bench](https://github.com/ServiceNow/Full-Duplex-Bench) into `Full-Duplex-Bench/` (it is git-ignored, not vendored) and point its v3 runner at the `fdb-agent` worker. The harness reads tool calls from `AGENT_TOOL_LOG`.

---

## Testing

Coordinator unit tests run offline, with deterministic gated tools instead of sleeps:

```bash
uv run pytest
```

`tests/test_coordinator.py` covers the core loop and the failure cases: normal request, interruption with the old result rejected, a late result with no cancellation, three chained corrections (only the last counts), duplicate and concurrent duplicate bookings, a booking cancelled mid-flight, a barge-in cancelling the tool task, tool failure, invalid/unknown tools, dynamically loaded tools, the JSONL trace, and non-blocking latency.

`tests/test_vision.py` and `tests/test_vision_features.py` cover:
- scene building (depth-based distance, fallback to box size, urgency ordering, rounded spoken distances, never claiming a path is safe, clock bearings, elevation);
- free space, walls, drop-offs (receding floor + jump is a drop; an obstacle's top edge is not; a smooth floor is not), pass side, stop;
- scene memory (debouncing, announcing once, "getting closer", cooldown, waiting until quiet, urgent-only while the agent talks, path-open notice, fast-approach vs walking pace, quiet/verbose/pause modes, guide-to-target until reached, lost target);
- the tools (find_object with aliases, guide_to/stop_guidance, set_guidance_mode, set_language, describe_view refusing without a live frame);
- the overlay renderer, the alert tone and the session report;
- interruption metrics.

They also spawn a real (torch-free) vision process to prove the end-to-end IPC and overlay streaming, that only the newest frame is processed when the worker is busy, that the event loop stays responsive while the models load, and that `torch` is never imported in the agent process. `tests/test_overlay_publisher.py` pushes frames through a real LiveKit video source into a faked room.

After a session, `uv run python scripts/report.py` prints a Markdown report from the newest trace: response latency (mean/p50/p90), barge-ins handled and how fast the agent stopped, tool outcomes (accepted, stale, aborted before commit, duplicates suppressed), vision FPS and latencies, and every unprompted announcement. Add `--timeline` for the full event list.

Every session writes interruption counts and cancel latency (`INTERRUPTION`, `INTERRUPTION_MISSED`, `FALSE_INTERRUPTION`). Sessions with a camera also write vision FPS, inference, queue and end-to-end latency, and the stale-frame rate. Both go to `SESSION_METRICS` in the JSONL trace.

Conversational behaviour is covered by `scenarios.yaml` (needs LiveKit Cloud credentials):

```bash
lk agent simulate text --scenarios scenarios.yaml
```

Testing should cover:

- Basic voice conversations
- Interruptions while speaking
- User corrections
- Stale tool results
- Multi-step tool calls
- Tool latency
- Repeated state-changing requests
- Vision queries
- Benchmark scenarios

### Manual interruption test

Try:

```text
User:
"Book me a..."

Agent starts responding

User:
"Actually, don't book it. Just tell me the price."
```

The important behaviour is that the second instruction becomes the active intent and the earlier action should not continue into an unintended final mutation.

---

## Project Structure

```text
interruptible-agent/
│
├── src/
│   ├── agent.py              # LiveKit voice agent (entry point)
│   ├── coordinator.py        # Versioned state, call lifecycle, stale checks, idempotency, trace
│   ├── tool_registry.py      # Schema-driven tool registry + validation
│   ├── mock_apis.py          # Simulated service operations + their schemas
│   ├── latency_injector.py   # Non-blocking tool latency simulation
│   ├── session_metrics.py    # Interruption latency / missed / false interruptions
│   ├── scene.py              # Detections + depth → structured scene (no torch)
│   ├── scene_memory.py       # Scene-change detection → spoken announcements
│   ├── vision_bridge.py      # Agent side: LiveKit camera → queues → latest_scene, overlay publisher
│   ├── vision_worker.py      # Spawned vision process: YOLO26 detection + depth + overlay
│   ├── vision_tools.py       # describe_view (VLM), find_object, guide_to, modes, language
│   └── alert_tone.py         # Generated two-tone alert WAV
│
├── scripts/
│   └── report.py             # Markdown session report from a trace
│
├── vision_guide/
│   ├── vision.py             # Standalone local-webcam mode → scene.json
│   └── scene.json            # Latest scene snapshot (generated, git-ignored)
│
├── models/                   # YOLO26 weights (downloaded on first run, git-ignored)
│
├── tests/                    # Coordinator, vision, tools and overlay tests
├── traces/                   # JSONL event traces (generated, git-ignored)
├── scenarios.yaml            # Conversation scenarios
│
├── pyproject.toml            # Python project configuration
├── Dockerfile                # Voice-agent container
├── .env.example              # Environment template
├── LICENSE
├── interruptible_agent.mp4   # Demo video
└── README.md
```

---

## Reproducibility

The repository keeps the main pieces needed to reproduce the project:

- Agent source code
- Tool implementations
- Vision pipeline
- Benchmark scenarios
- Configuration templates
- Docker configuration
- Evaluation resources
- Demo video

For a clean reproduction:

```text
1. Clone repository
2. Install dependencies
3. Configure .env
4. Start LiveKit agent
5. Run VisionGuide separately if needed
6. Run FDB-v3 evaluation
7. Compare results with the recorded benchmark run
```

The benchmark result reported above belongs to the tested version of the agent and may change with model/provider versions, network conditions, hardware, and configuration.

---

## Limitations

1. **Interruption is not perfect** — low-volume or poorly detected speech can still fail to interrupt the agent.
2. **STT fragmentation** — speech may sometimes be split into smaller chunks.
3. **Latency** — STT, LLM, tool execution, and TTS all contribute to end-to-end delay.
4. **Complex tool chains** — longer multi-step interactions can still fail.
5. **Mock APIs** — the current travel, finance, housing, and e-commerce tools do not perform real-world transactions.
6. **Vision accuracy** — YOLO performance depends on lighting, camera position, and the detected object. COCO classes do not include stairs, doors or kerbs.
7. **Approximate depth** — distances come from a monocular depth model and are rounded estimates, not measurements. Wall, drop-off and fast-approach detection are heuristics on that depth map and can miss or misfire (glass, mirrors, strong shadows, a steeply tilted camera).
8. **Single camera, label-level memory** — objects are not tracked individually. Two chairs ahead count as one "chair" for announcements.
9. **Safety** — VisionGuide should not be used as a safety-critical navigation system; it complements a cane or guide dog.
11. **Vision-language answers** — `describe_view` takes 3–5 s and can misread small or partly hidden text; it is grounded with the detector's distances but remains a model's description.
10. **Provider dependency** — the voice pipeline depends on external model/provider services.

---

## Demo Script

A five-minute walk-through that shows every part of the system:

1. **Voice + tools.** "Find flights from Chennai to Delhi on October 2nd, actually make it Mumbai." The search uses Mumbai as destination and keeps Chennai. Then: "Book it for Priya Sharma… no wait, don't book it." Nothing is booked; the trace shows the mutation aborted before commit.
2. **Barge-in.** Ask for the apartment search plus commute, interrupt mid-answer with "stop, just the price", and watch it stop within a few hundred milliseconds (the report prints the number).
3. **Camera on.** Point the phone forward in the Playground; the `visionguide` video appears with boxes, distances and the arrow. Ask "Which way should I go?"
4. **Walk toward a wall.** The alert tone sounds, then "Careful, a wall directly ahead… Go to your right." Turn away: "The way ahead looks open now."
5. **Find and guide.** "Where is my phone?" → "on your right, about 1 metre away, low down." "Guide me to the chair" → turn-by-turn until "You've reached the chair."
6. **Read.** Hold up a sign or a product: "Read what's in front of me."
7. **Quiet and Hindi.** "Be quiet, only warn me about dangers." Then "Speak Hindi from now on" and ask for the flight again in Hindi.
8. **Report.** `uv run python scripts/report.py --timeline` for the latency, interruption and vision numbers of the session you just ran.

---

## Demo Video

### Full Demo

[**▶ Watch `interruptible_agent.mp4`**](./interruptible_agent.mp4)

The demo covers the voice interaction, interruption behaviour, tool usage, and the extended VisionGuide use case.

---

## Team / Credits

**Project:** Interruptible Full-Duplex Voice Agent

**Track:** Samsung PRISM / Full-Duplex-Bench v3

Built using:

- LiveKit
- Deepgram
- OpenAI
- Cartesia
- Silero
- Ultralytics YOLO

Team members / contributors are listed in the repository commit history and project submission.

---

## References

- [LiveKit Agents](https://docs.livekit.io/agents/)
- [Full-Duplex-Bench](https://github.com/ServiceNow/Full-Duplex-Bench)
- [Deepgram](https://deepgram.com/)
- [OpenAI](https://openai.com/)
- [Cartesia](https://cartesia.ai/)
- [Ultralytics](https://docs.ultralytics.com/)

---

## License

See [`LICENSE`](LICENSE) for license details.
