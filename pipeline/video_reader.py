from pathlib import Path
from typing import Generator, Union
import threading
from queue import Queue, Full

import cv2

from .config import DEFAULT_VIDEO_FPS
class VideoReader:

    def __init__(
        self,
        source: Union[int, str, Path],
        width: int | None = None,
        height: int | None = None
    ):
        self.source = source
        self.width = width
        self.height = height
        self.capture = None

    def open(self):
        self.capture = cv2.VideoCapture(self.source)

        if not self.capture.isOpened():
            raise RuntimeError(
                f"Không thể mở source: {self.source}"
            )

        try:
            self.capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
        except Exception:
            pass

        if self.width is not None:
            self.capture.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                self.width
            )

        if self.height is not None:
            self.capture.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                self.height
            )

        return self

    def read(self):
        if self.capture is None:
            raise RuntimeError(
            )

        return self.capture.read()

    def frames(self) -> Generator:
        if self.capture is None:
            self.open()

        while True:
            ret, frame = self.capture.read()

            if not ret:
                break

            yield frame

    def get_fps(self) -> float:
        if self.capture is None:
            return 0.0

        fps = self.capture.get(
            cv2.CAP_PROP_FPS
        )

        if fps <= 0:
            return DEFAULT_VIDEO_FPS

        return fps

    def get_width(self) -> int:
        if self.capture is None:
            return 0

        return int(
            self.capture.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        )

    def get_height(self) -> int:
        if self.capture is None:
            return 0

        return int(
            self.capture.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        )

    def release(self):
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback
    ):
        self.release()

class ThreadedVideoReader(VideoReader):

    def __init__(
        self,
        source: Union[int, str, Path],
        queue_size: int = 2,
        drop_frames: bool = True,
        width: int | None = None,
        height: int | None = None,
    ):
        super().__init__(source, width, height)
        self.queue_size = queue_size
        self.drop_frames = drop_frames
        self.frame_queue = Queue(maxsize=queue_size)
        self.stopped = False
        self.thread = None

    def start(self):
        self.open()
        self.stopped = False
        self.thread = threading.Thread(
            target=self._update, name="VideoReaderThread", daemon=True
        )
        self.thread.start()
        return self

    def _update(self):
        while not self.stopped:
            if not self.capture.isOpened():
                break

            ret, frame = self.capture.read()
            if not ret:
                self.stopped = True
                break

            if self.drop_frames:
                try:
                    self.frame_queue.put_nowait(frame)
                except Full:
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        pass
                    try:
                        self.frame_queue.put_nowait(frame)
                    except Full:
                        pass
            else:
                self.frame_queue.put(frame)

        self.stopped = True

    def is_finished(self) -> bool:
        return self.stopped and self.frame_queue.empty()

    def release(self):
        self.stopped = True
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        super().release()