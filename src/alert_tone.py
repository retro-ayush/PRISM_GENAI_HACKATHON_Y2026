"""Short alert tone played before urgent VisionGuide warnings.

A tone reaches the user faster than words ("Stop" takes ~400 ms to say, and
TTS adds its own delay), and it is distinct from normal speech, so a blind
user learns to react to the sound itself. The WAV is generated on first use:
two short rising beeps, 16-bit mono.
"""

from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

SAMPLE_RATE = 24000


def ensure_alert_wav(path: Path) -> str:
    if path.exists():
        return str(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples: list[int] = []
    for freq, length in ((880.0, 0.09), (0.0, 0.05), (1320.0, 0.11)):
        n = int(SAMPLE_RATE * length)
        fade = int(SAMPLE_RATE * 0.008)
        for i in range(n):
            if freq == 0.0:
                samples.append(0)
                continue
            env = min(1.0, i / fade, (n - i) / fade)
            value = 0.45 * env * math.sin(2 * math.pi * freq * i / SAMPLE_RATE)
            samples.append(int(value * 32767))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return str(path)
