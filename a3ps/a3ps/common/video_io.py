"""Frame reader/writer with fps handling. Thin OpenCV wrappers.

Imports of OpenCV are local so this module can be imported (for typing /
signatures) without the heavy dependency installed.
"""

from __future__ import annotations

from typing import Iterator, Optional, Tuple


class VideoReader:
    """Iterate frames from a video, yielding (frame_idx, t_seconds, frame)."""

    def __init__(self, path: str):
        import cv2

        self._cv2 = cv2
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"Cannot open video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __iter__(self) -> Iterator[Tuple[int, float, "object"]]:
        idx = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield idx, idx / self.fps, frame
            idx += 1

    def close(self) -> None:
        self.cap.release()

    def __enter__(self) -> "VideoReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


class VideoWriter:
    """Write annotated frames to an mp4 at a fixed fps."""

    def __init__(self, path: str, fps: float, width: int, height: int):
        import cv2

        self._cv2 = cv2
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not self.writer.isOpened():
            raise IOError(f"Cannot open writer: {path}")

    def write(self, frame) -> None:
        self.writer.write(frame)

    def close(self) -> None:
        self.writer.release()

    def __enter__(self) -> "VideoWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
