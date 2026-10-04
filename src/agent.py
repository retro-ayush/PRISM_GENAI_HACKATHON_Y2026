import dataclasses
import importlib.util
import json
import logging
import os
import re
import tempfile
import time
from pathlib import Path

from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    BackgroundAudioPlayer,
    JobContext,
    JobProcess,
    inference,
    llm,
    room_io,
)
from livekit.plugins import noise_cancellation, silero

from alert_tone import ensure_alert_wav
from coordinator import Coordinator, classify_utterance
from latency_injector import LatencyInjector
from mock_apis import register_mock_apis
from scene_memory import SceneMemory
from session_metrics import InterruptionTracker
from tool_registry import ToolRegistry, ToolSpec
from vision_bridge import OverlayPublisher, VisionBridge
from vision_tools import VisionContext, register_vision_tools
from vision_worker import run_fake_worker, run_worker

load_dotenv()
logger = logging.getLogger("agent")

ROOT_DIR = Path(__file__).resolve().parent.parent
SCENE_FILE = ROOT_DIR / "vision_guide" / "scene.json"
ALERT_WAV = Path(
    os.getenv("ALERT_WAV", Path(tempfile.gettempdir()) / "visionguide_alert.wav")
)
# Scene snapshots older than this are treated as stale (camera stopped).
SCENE_MAX_AGE_S = float(os.getenv("SCENE_MAX_AGE_S", "5"))
TRACE_DIR = Path(os.getenv("TRACE_DIR", ROOT_DIR / "traces"))
# Per-call log read by the Full-Duplex-Bench v3 harness.
TOOL_LOG_PATH = os.getenv("AGENT_TOOL_LOG", "/tmp/agent_tool_calls.log")

# VisionGuide: "livekit" reads the user's LiveKit camera in a spawned vision
# process, "fake" uses a torch-free stand-in worker, "off" disables the camera
# pipeline. The standalone webcam script's scene.json is always the fallback.
VISION_MODE = os.getenv("VISION_MODE", "livekit")
# Speak important scene changes (new or approaching obstacle) unprompted.
VISION_ANNOUNCE = os.getenv("VISION_ANNOUNCE", "1") == "1"
VISION_ANNOUNCE_GAP_S = 3.0
# Stream the annotated camera view back into the room as a video track.
VISION_OVERLAY = os.getenv("VISION_OVERLAY", "1") == "1"
# Play a short tone before urgent warnings (faster to react to than words).
VISION_ALERT_TONE = os.getenv("VISION_ALERT_TONE", "1") == "1"
# Vision-language model used by describe_view (any LiveKit Inference model
# that accepts images).
VLM_MODEL = os.getenv("VLM_MODEL", "openai/gpt-4o-mini")

# Workers with a name only join rooms that explicitly dispatch that name (the
# FDB-v3 harness dispatches "fdb-agent"). Set AGENT_NAME= (empty) to join every
# new room automatically, e.g. when testing from the Agents Playground.
AGENT_NAME = os.getenv("AGENT_NAME", "fdb-agent")

# Full-Duplex-Bench runs: the harness streams pre-recorded, disfluent audio
# 2 s after creating the room, so skip everything that delays joining or
# talks over the user (greeting, VisionGuide, noise cancellation) and wait
# longer before ending a turn on "um... like..." pauses.
BENCHMARK_MODE = os.getenv("BENCHMARK_MODE", "0") == "1"
LLM_MODEL = os.getenv(
    "LLM_MODEL", "openai/gpt-4o" if BENCHMARK_MODE else "openai/gpt-4o-mini"
)
# Silence (s) before a turn ends when the turn detector thinks the user is
# done (min) or still mid-sentence (max). Unset outside the benchmark, so
# LiveKit's defaults apply.
ENDPOINTING: dict = {}
if min_s := os.getenv("ENDPOINT_MIN_S", "1.5" if BENCHMARK_MODE else ""):
    ENDPOINTING["min_delay"] = float(min_s)
if max_s := os.getenv("ENDPOINT_MAX_S", "6.0" if BENCHMARK_MODE else ""):
    ENDPOINTING["max_delay"] = float(max_s)

# Minimum user speech (s) that interrupts the agent. Lower catches short or
# quiet barge-ins; LiveKit resumes speech on false interruptions.
INTERRUPT_MIN_S = float(os.getenv("INTERRUPT_MIN_S", "0.3"))
# Deepgram language. "multi" understands code-switching between English,
# Hindi, Spanish and more; keyterm boosting (below) is English-only.
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "en")
# Vocabulary the STT would otherwise mishear: tool domains and demo places.
STT_KEYTERMS = [
    "VisionGuide",
    "autopay",
    "Whitefield",
    "Chennai",
    "Mumbai",
    "Delhi",
    "Bangalore",
    "Goa",
    "Pune",
    "Hyderabad",
    "exchange rate",
    "card benefits",
    "track my order",
    "add to cart",
    "guide me",
]

INSTRUCTIONS = (
    "You are a helpful, fast, conversational voice AI assistant. "
    "Keep spoken responses concise and natural, in plain prose with no lists, "
    "markdown or emojis. "
    # Tools
    "You have simulated APIs for Travel, Finance, Housing and E-Commerce in a "
    "safe test environment. Use them whenever a request needs their data, and "
    "never invent API results. When a request clearly needs a tool, call it "
    "without unnecessary clarification. For multi-step tasks, complete the "
    "steps in order without stopping in between, and feed values returned by "
    "one tool into the next (for example, use the address of an apartment "
    "from search_apartments as the origin for calculate_commute) instead of asking "
    "the user for them. Only call a booking, purchase or update tool when the "
    "user explicitly asks for that action in words like 'book it' or 'buy it'. "
    "After a search, report the result and stop; do not ask whether to book. "
    "'Thanks', 'okay', 'that works' or 'sounds good' are acknowledgements, not "
    "authorization: reply briefly and do not start a booking or ask for "
    "booking details. If it is unclear whether the user wants an action, ask "
    "'Should I book it?' and wait for a yes. Never invent required details "
    "such as a passenger name: ask for them only once the user has asked to book. "
    "If a correction is ambiguous about which detail changed, "
    "ask a short clarifying question instead of guessing. "
    # Interruptions / corrections
    "The user may interrupt or correct themselves at any time. The newest user "
    "instruction is authoritative: drop details from the earlier request that "
    "it replaces, and do not continue an old task the user has changed. "
    "A correction replaces the detail the user said just before it: in 'from "
    "Chennai to Delhi, actually make it Mumbai', Mumbai replaces Delhi, the "
    "destination, and Chennai stays the origin. "
    "If a tool returns status 'stale', its result is outdated and must not be "
    "mentioned as fact; act on the latest request instead. If a tool returns "
    "status 'already_completed', the action was not repeated; say it was "
    "already done. Never claim an action is complete before its tool succeeded. "
)

VISION_INSTRUCTIONS = (
    "You are also VisionGuide, an experimental camera assistant for blind and "
    "visually impaired users. Never guess what the camera sees; only state what "
    "a tool returned. Tools: get_current_scene for what is ahead or around, "
    "obstacles and which way to go (it is fetched for you automatically on "
    "such questions); find_object for 'where is my phone'; describe_view for "
    "detailed descriptions, reading signs, labels or screens, colours, and "
    "anything the object list cannot answer; guide_to and stop_guidance to "
    "walk the user to an object; set_guidance_mode when they ask you to be "
    "quiet, say more, pause or resume warnings; set_language when they ask "
    "for another language. "
    "Answering from a scene: mention what is directly ahead first and warn "
    "clearly about anything close or very close. For direction, follow "
    "'recommendation' (straight, step_left, step_right, stop or stop_drop) and "
    "an object's 'pass_on' side; 'clearance_m' is the rough free distance on "
    "the left, center and right; a 'wall' or 'obstacle' comes from the depth "
    "camera and is something unrecognised in the way; 'drop_off' means the "
    "ground may fall away, so tell the user to stop and check with their cane. "
    "distance_m is a rough estimate: say 'about' and round it. The camera only "
    "sees forward, so you cannot know what is behind the user. Never say a path "
    "is safe or clear because nothing was detected. Keep guidance to one or two "
    "sentences, the most urgent thing first. When guide_to returns 'say_now', "
    "say exactly that. Acknowledge every tool result in your reply, including "
    "stop_guidance and set_guidance_mode, even when another tool failed. When "
    "set_language succeeds, follow its instruction. "
    "If get_current_scene, find_object, describe_view or guide_to reports the "
    "camera is not running or no frame is available, tell the user that in one "
    "sentence and do not call another camera tool in that turn. stop_guidance "
    "and set_guidance_mode are settings, not camera tools; call them only when "
    "the user asks for them. "
    "Tool arguments such as city names stay in English whatever language the "
    "user speaks. VisionGuide also announces obstacles on its own between turns."
)
if not BENCHMARK_MODE:
    INSTRUCTIONS += VISION_INSTRUCTIONS


def read_scene_file() -> dict:
    """Scene written by the standalone webcam script (vision_guide/vision.py)."""
    try:
        scene = json.loads(SCENE_FILE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {"status": "error", "error": "The vision camera is not running."}
    age = time.time() - float(scene.get("timestamp", 0))
    if age > SCENE_MAX_AGE_S:
        return {
            "status": "error",
            "error": f"The latest camera snapshot is {age:.0f}s old; the "
            "vision camera appears to be stopped.",
        }
    return {"status": "success", **scene}


_DROP_KEYS = ("confidence", "source", "box", "elevation")


def compact_scene(scene: dict) -> dict:
    """What the LLM needs from a scene, and nothing else: fewer tokens per
    vision question means a faster spoken answer."""
    if scene.get("status") != "success":
        return scene
    objects = [
        {k: v for k, v in o.items() if k not in _DROP_KEYS}
        for o in scene.get("objects", [])[:6]
    ]
    others = [
        {"label": o["label"], "position": o["position"], "distance_m": o["distance_m"]}
        for o in scene.get("others", [])[:5]
    ]
    path = scene.get("path") or {}
    out = {
        "status": "success",
        "guidance": scene.get("guidance"),
        "objects": objects,
        "clearance_m": path.get("clearance_m"),
        "wall_ahead": path.get("wall_ahead"),
        "recommendation": path.get("recommendation"),
    }
    if path.get("drop_off"):
        out["drop_off"] = path["drop_off"]
    if others:
        out["other_objects"] = others
    return out


def build_registry(vc: VisionContext) -> ToolRegistry:
    injector = LatencyInjector(profile=os.getenv("MOCK_LATENCY_PROFILE", "instant"))
    registry = register_mock_apis(ToolRegistry(), injector)
    if BENCHMARK_MODE:
        # Camera tools would only be unexpected calls in the benchmark.
        return _load_extra_tools(registry)
    registry.register(
        ToolSpec(
            name="get_current_scene",
            description="Read the latest VisionGuide camera scene: detected "
            "objects (including walls) with their position, approximate "
            "distance in metres and which side to pass them on, free space on "
            "each side, drop-off cues, and the recommended walking direction. "
            "Call it for any question about surroundings, obstacles or which "
            "way to go.",
            kind="read_only",
            parameters={"type": "object", "properties": {}, "required": []},
            handler=lambda: compact_scene(vc.raw_scene()),
        )
    )
    register_vision_tools(registry, vc)
    return _load_extra_tools(registry)


def _load_extra_tools(registry: ToolRegistry) -> ToolRegistry:
    # Unseen tools can be supplied at runtime as JSON schemas.
    if extra := os.getenv("EXTRA_TOOLS_FILE"):
        logger.info("loaded extra tools: %s", registry.load_json(extra))
    return registry


def build_livekit_tools(coordinator: Coordinator) -> list[llm.RawFunctionTool]:
    """Expose every registry tool to the LLM, routed through the coordinator."""

    def make_tool(spec: ToolSpec) -> llm.RawFunctionTool:
        async def handler(raw_arguments: dict[str, object]) -> str:
            result = await coordinator.run_tool(spec.name, dict(raw_arguments))
            return json.dumps(result, default=str)

        return llm.function_tool(handler, raw_schema=spec.schema())

    return [make_tool(spec) for spec in coordinator.registry]


# Questions about the user's surroundings. For these the scene is fetched
# before the LLM runs, so the answer always uses a fresh scene and needs no
# tool round trip.
_VISION_QUESTION = re.compile(
    r"\b(ahead|in front|around me|behind me|surround\w*|obstacle\w*|wall\w*|"
    r"door\w*|camera|which way|left or right|where (should|do|can) i|"
    r"(go|walk|move|turn|step)\w* (left|right|straight|forward|ahead)|"
    r"safe to|blocked|clear|in my way|path|bump|crash|stairs?|steps?|drop)\b",
    re.I,
)


# Questions about visual detail the detector cannot answer (text, colours,
# a full description). These are answered by the vision-language model, so
# the first LLM call of such a turn is forced to call describe_view.
_DETAIL_QUESTION = re.compile(
    r"(\bread (the|this|that|it|what|any|out|aloud)\b|"
    r"\bwhat does (it|this|that|the \w+) say\b|"
    r"\bwhat('s| is) (written|on the (sign|label|screen|board|menu|box|bottle|packet|door))\b|"
    r"\b(is there|any) (text|writing|sign|label)\b|"
    r"\b(describe|detail\w*) (the|my|what|this|everything|around)\b|"
    r"\bwhat (do|can) you see\b|\bwhat does \w+ look like\b|"
    r"\b(which|what) colou?r\b|\bcolou?r of\b|"
    r"\b(sign|label|text|writing|price tag|menu|screen|display|board|poster|notice)\b"
    r".*\b(say|says|read|front|ahead|see)\b|"
    r"\b(read|see|says)\b.*\b(sign|label|text|writing|number|price|menu|screen|display|board|notice)\b)",
    re.I,
)
_FORCE_DESCRIBE = {"type": "function", "function": {"name": "describe_view"}}


class VoiceAgent(Agent):
    def __init__(self, coordinator: Coordinator) -> None:
        super().__init__(
            instructions=INSTRUCTIONS, tools=build_livekit_tools(coordinator)
        )
        self.coordinator = coordinator
        self._turn_text: str | None = None
        self._turn_at = 0.0

    def _begin_turn(self, text: str) -> list[str]:
        """Make the newest user turn the authoritative intent (once per turn).

        Spoken turns arrive via on_user_turn_completed, typed turns only via
        llm_node, and with preemptive generation llm_node can run first; the
        text/time check makes whichever comes second a no-op.
        """
        now = time.monotonic()
        if text == self._turn_text and now - self._turn_at < 10:
            return []
        self._turn_text, self._turn_at = text, now
        superseded = self.coordinator.on_user_turn(text)
        if not superseded:
            return []
        names = ", ".join(sorted({c.tool_name for c in superseded}))
        return [
            f"The user's new message superseded in-flight work ({names}); "
            "those results are discarded. Act on the new message."
        ]

    async def on_user_turn_completed(
        self, turn_ctx: llm.ChatContext, new_message: llm.ChatMessage
    ) -> None:
        for note in self._begin_turn(new_message.text_content or ""):
            turn_ctx.add_message(role="system", content=note)

    async def llm_node(self, chat_ctx, tools, model_settings):
        last = chat_ctx.items[-1] if chat_ctx.items else None
        if isinstance(last, llm.ChatMessage) and last.role == "user":
            # First LLM call of a turn (not a follow-up after tool results).
            text = last.text_content or ""
            notes = self._begin_turn(text)
            if classify_utterance(text) == "correction":
                notes.append(
                    "This message corrects itself. The new value replaces only "
                    "the detail named just before the correction; keep every "
                    "other detail (e.g. 'from Chennai to Delhi, actually Mumbai' "
                    "means origin Chennai, destination Mumbai)."
                )
            extra: list[llm.ChatItem] = [
                llm.ChatMessage(role="system", content=[note]) for note in notes
            ]
            if BENCHMARK_MODE:
                pass  # no camera: never force or prefetch vision tools
            elif _DETAIL_QUESTION.search(text):
                # The detector's object list can't read text or see colours:
                # make this call go to the vision-language model.
                model_settings = dataclasses.replace(
                    model_settings, tool_choice=_FORCE_DESCRIBE
                )
                notes.append(
                    "This asks about visual detail (text, colours, a description): "
                    "call describe_view with the user's question and answer only "
                    "from its result."
                )
            elif _VISION_QUESTION.search(text):
                # Fetch the scene up front, as if the model had called the
                # tool: the answer always reflects where the user is now, and
                # it saves an LLM round trip. It goes into this request only
                # (it is logged in the trace), so later turns can't reuse a
                # stale scene from the history.
                scene = await self.coordinator.run_tool("get_current_scene", {})
                call_id = f"scene_{int(time.time() * 1000)}"
                extra += [
                    llm.FunctionCall(
                        call_id=call_id, name="get_current_scene", arguments="{}"
                    ),
                    llm.FunctionCallOutput(
                        call_id=call_id,
                        name="get_current_scene",
                        output=json.dumps(scene, default=str),
                        is_error=False,
                    ),
                ]
            if extra:
                chat_ctx = chat_ctx.copy()
                chat_ctx.items.extend(extra)
        async for chunk in Agent.default.llm_node(
            self, chat_ctx, tools, model_settings
        ):
            yield chunk


def make_vision_bridge() -> VisionBridge | None:
    if VISION_MODE == "off" or BENCHMARK_MODE:
        return None
    if VISION_MODE == "livekit" and importlib.util.find_spec("ultralytics") is None:
        logger.warning("VisionGuide disabled: install it with `uv sync --extra vision`")
        return None
    return VisionBridge(
        target=run_fake_worker if VISION_MODE == "fake" else run_worker,
        config={
            "detect_model": os.getenv("VISION_DETECT_MODEL", "yolo26n.pt"),
            "depth_model": os.getenv("VISION_DEPTH_MODEL", "yolo26n-depth.pt"),
            "device": os.getenv("VISION_DEVICE", "auto"),
            "min_confidence": float(os.getenv("VISION_MIN_CONFIDENCE", "0.45")),
            "overlay": VISION_OVERLAY,
        },
        max_fps=float(os.getenv("VISION_MAX_FPS", "8")),
        scene_max_age_s=SCENE_MAX_AGE_S,
    )


server = AgentServer()


def prewarm(proc: JobProcess) -> None:
    proc.userdata["vad"] = silero.VAD.load()
    if VISION_ALERT_TONE:
        ensure_alert_wav(ALERT_WAV)


server.setup_fnc = prewarm


@server.rtc_session(agent_name=AGENT_NAME)
async def entrypoint(ctx: JobContext) -> None:
    room_name = ctx.room.name
    vision = make_vision_bridge()
    memory = SceneMemory() if vision else None

    def raw_scene() -> dict:
        scene = vision.current_scene() if vision else None
        return scene if scene is not None else read_scene_file()

    vc = VisionContext(
        raw_scene=raw_scene,
        memory=memory,
        grab_frame=vision.grab_frame if vision else None,
        vlm_model=VLM_MODEL,
        scene_max_age_s=SCENE_MAX_AGE_S,
    )
    coordinator = Coordinator(
        build_registry(vc),
        session_id=room_name,
        trace_path=TRACE_DIR / f"trace_{room_name}.jsonl",
        tool_log_path=TOOL_LOG_PATH,
    )
    interruptions = InterruptionTracker(
        lambda event, **data: coordinator.trace.log(
            event, coordinator.state.version, **data
        )
    )

    stt_kwargs: dict = {}
    if STT_LANGUAGE == "en":
        stt_kwargs["extra_kwargs"] = {"keyterm": STT_KEYTERMS}
    session = AgentSession(
        stt=inference.STT(model="deepgram/nova-3", language=STT_LANGUAGE, **stt_kwargs),
        llm=inference.LLM(model=LLM_MODEL),
        tts=inference.TTS(model="cartesia/sonic-3"),
        vad=ctx.proc.userdata["vad"],
        turn_handling={
            # Context-aware end-of-turn model: fewer cut-offs on mid-sentence
            # pauses and fragmented STT than silence-only endpointing.
            "turn_detection": inference.TurnDetector(),
            "interruption": {"min_duration": INTERRUPT_MIN_S},
            # Start the LLM on the partial transcript before the turn ends.
            "preemptive_generation": {"enabled": True},
            **({"endpointing": ENDPOINTING} if ENDPOINTING else {}),
        },
    )
    vc.session = session

    @session.on("agent_state_changed")
    def on_agent_state(ev: agents.voice.AgentStateChangedEvent) -> None:
        interruptions.on_agent_state(ev.new_state, ev.created_at)
        if ev.new_state == "speaking":
            latency_ms = coordinator.on_agent_speaking()
            if latency_ms is not None:
                logger.info("response latency %.0f ms (room %s)", latency_ms, room_name)

    @session.on("user_state_changed")
    def on_user_state(ev: agents.voice.UserStateChangedEvent) -> None:
        interruptions.on_user_state(ev.new_state, ev.created_at)

    @session.on("agent_false_interruption")
    def on_false_interruption(ev: agents.voice.AgentFalseInterruptionEvent) -> None:
        interruptions.on_false_interruption(ev.resumed)

    audio: BackgroundAudioPlayer | None = None
    overlay: OverlayPublisher | None = None
    if vision:
        if VISION_OVERLAY:
            overlay = OverlayPublisher(ctx.room)
            vision.on_overlay = overlay.push
        last_announced = 0.0

        def on_scene(scene: dict) -> None:
            # Never talk over the user. While the agent itself is busy, only
            # urgent warnings (very close obstacle, path blocked) get through
            # and cut its speech short; other changes wait until it's quiet.
            nonlocal last_announced
            user_talking = session.user_state == "speaking"
            agent_busy = session.agent_state != "listening"
            gap_ok = time.monotonic() - last_announced >= VISION_ANNOUNCE_GAP_S
            message = memory.update(
                scene,
                announce=VISION_ANNOUNCE and not user_talking,
                urgent_only=agent_busy or not gap_ok,
            )
            if not message:
                return
            last_announced = time.monotonic()
            coordinator.trace.log(
                "VISION_ANNOUNCEMENT",
                coordinator.state.version,
                text=message.text,
                kind=message.kind,
                urgent=message.urgent,
            )
            if message.urgent:
                if session.agent_state == "speaking":
                    session.interrupt()
                if audio is not None:
                    audio.play(str(ALERT_WAV))
            session.say(message.text, allow_interruptions=True)

        vision.on_scene = on_scene

        def maybe_attach(
            track: rtc.Track, pub: rtc.RemoteTrackPublication, _p=None
        ) -> None:
            if track.kind == rtc.TrackKind.KIND_VIDEO and (
                pub.source == rtc.TrackSource.SOURCE_CAMERA
            ):
                logger.info("camera track subscribed: %s", track.sid)
                memory.reset()
                vision.attach_track(track)

        @ctx.room.on("track_subscribed")
        def on_track_subscribed(track, pub, participant) -> None:
            maybe_attach(track, pub, participant)

        @ctx.room.on("track_unsubscribed")
        def on_track_unsubscribed(track, pub, _participant) -> None:
            if track.kind == rtc.TrackKind.KIND_VIDEO:
                vision.detach_track()

    async def on_shutdown() -> None:
        summary = {"interruptions": interruptions.summary()}
        if vision:
            summary["vision"] = vision.metrics()
            if overlay is not None:
                summary["overlay_frames"] = overlay.frames
                await overlay.aclose()
            await vision.aclose()
        coordinator.trace.log("SESSION_METRICS", coordinator.state.version, **summary)
        logger.info("session metrics: %s", summary)
        coordinator.close()

    ctx.add_shutdown_callback(on_shutdown)

    await session.start(
        agent=VoiceAgent(coordinator),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            # Noise cancellation is for live microphones; the benchmark's
            # recorded audio is clean and BVC can clip its quiet fillers.
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=None if BENCHMARK_MODE else noise_cancellation.BVC(),
            ),
        ),
    )
    if vision:
        if VISION_ALERT_TONE:
            try:
                audio = BackgroundAudioPlayer()
                await audio.start(room=ctx.room, agent_session=session)
            except Exception:
                logger.exception("alert tone player failed to start")
                audio = None
        # Cameras published before the agent joined.
        for participant in ctx.room.remote_participants.values():
            for pub in participant.track_publications.values():
                if pub.track is not None:
                    maybe_attach(pub.track, pub)
    if not BENCHMARK_MODE:
        # In the benchmark the user is already talking when the agent joins.
        await session.say("Hello! I'm ready to assist you.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if os.path.dirname(TOOL_LOG_PATH):
        os.makedirs(os.path.dirname(TOOL_LOG_PATH), exist_ok=True)
    agents.cli.run_app(server)
