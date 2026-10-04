"""OverlayPublisher: decodes the worker's annotated JPEG and pushes it into a
LiveKit video track. The room is faked; the rtc video source/track are real."""

import asyncio

import numpy as np

from vision_bridge import OverlayPublisher


class _Publication:
    sid = "TR_overlay"


class _LocalParticipant:
    def __init__(self) -> None:
        self.published = []
        self.unpublished = []

    async def publish_track(self, track, options):
        self.published.append((track, options))
        return _Publication()

    async def unpublish_track(self, sid):
        self.unpublished.append(sid)


class _Room:
    def __init__(self) -> None:
        self.local_participant = _LocalParticipant()


def _jpeg(width: int, height: int) -> bytes:
    import cv2

    frame = np.full((height, width, 3), 90, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", frame)
    assert ok
    return buf.tobytes()


async def test_publishes_once_and_streams_frames():
    from livekit import rtc

    room = _Room()
    publisher = OverlayPublisher(room)
    publisher.push(_jpeg(320, 240))
    await asyncio.sleep(0.3)
    assert len(room.local_participant.published) == 1
    track, options = room.local_participant.published[0]
    assert isinstance(track, rtc.LocalVideoTrack)
    assert options.source == rtc.TrackSource.SOURCE_CAMERA
    assert publisher.frames == 1

    publisher.push(_jpeg(640, 480))  # a different size is resized to the source
    await asyncio.sleep(0.3)
    assert publisher.frames == 2 and len(room.local_participant.published) == 1

    await publisher.aclose()
    assert room.local_participant.unpublished == ["TR_overlay"]
    publisher.push(_jpeg(320, 240))  # closed: ignored
    await asyncio.sleep(0.1)
    assert publisher.frames == 2


async def test_frames_are_dropped_while_busy_not_queued():
    publisher = OverlayPublisher(_Room())
    jpeg = _jpeg(320, 240)
    for _ in range(5):
        publisher.push(jpeg)  # all five arrive before the first decode finishes
    await asyncio.sleep(0.4)
    assert publisher.frames == 1
    await publisher.aclose()
