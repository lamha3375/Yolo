import os
import threading
import time
from queue import Empty, Queue

import cv2

from .pipeline import PersonPipeline
from .video_reader import ThreadedVideoReader
from .config import (
    WINDOW_NAME,
    SHOW_FPS,
    SHOW_PERSON_COUNT,
    DISPLAY_MAX_WIDTH,
    DISPLAY_MAX_HEIGHT,
)


def _fit_frame_for_display(
    frame,
    max_width=DISPLAY_MAX_WIDTH,
    max_height=DISPLAY_MAX_HEIGHT,
):
    h, w = frame.shape[:2]

    if h <= 0 or w <= 0:
        return frame

    scale = min(
        max_width / w,
        max_height / h,
        1.0,
    )

    if scale == 1.0:
        return frame

    new_width = max(1, int(round(w * scale)))
    new_height = max(1, int(round(h * scale)))

    return cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def _create_display_window():
    cv2.namedWindow(
        WINDOW_NAME,
        cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO,
    )

class ThreadedPersonPipeline(PersonPipeline):

    def run_threaded(
        self,
        source,
        show=True,
        output_path=None,
        queue_size=2,
        drop_frames=True,
    ):
 
        reader = ThreadedVideoReader(
            source=source,
            queue_size=queue_size,
            drop_frames=drop_frames,
        )

        result_queue = Queue(maxsize=queue_size)
        stop_event = threading.Event()

        writer = None
        frame_count = 0
        processing_count = 0

        previous_time = time.time()
        fps = 0.0

        worker_error = []

        def processing_worker():
            nonlocal processing_count

            try:
                while not stop_event.is_set():
                    try:
                        frame = reader.frame_queue.get(
                            timeout=0.1
                        )
                    except Empty:
                        if reader.is_finished():
                            break
                        continue

                    result = self.process_frame(frame)

                    try:
                        result_queue.put_nowait(result)
                    except:
                        try:
                            result_queue.get_nowait()
                        except Empty:
                            pass

                        try:
                            result_queue.put_nowait(result)
                        except:
                            pass

                    processing_count += 1

            except Exception as exc:
                worker_error.append(exc)
                stop_event.set()

        try:
            reader.start()

            if show:
                _create_display_window()

            video_fps = reader.get_fps()

            processing_thread = threading.Thread(
                target=processing_worker,
                name="ProcessingThread",
                daemon=True,
            )

            processing_thread.start()

            while not stop_event.is_set():
                try:
                    result = result_queue.get(
                        timeout=0.1
                    )
                except Empty:
                    if (
                        reader.is_finished()
                        and not processing_thread.is_alive()
                    ):
                        break
                    continue

                output_frame = result["frame"]

                input_height, input_width = (
                    output_frame.shape[:2]
                )

                current_time = time.time()
                elapsed = current_time - previous_time

                if elapsed > 0:
                    fps = 1.0 / elapsed

                previous_time = current_time

                if SHOW_FPS:
                    cv2.putText(
                        output_frame,
                        f"FPS: {fps:.1f}",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                if SHOW_PERSON_COUNT:
                    person_count = len(
                        result["detected_objects"]
                    )

                    cv2.putText(
                        output_frame,
                        f"Persons: {person_count}",
                        (20, 70),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                if output_path is not None and writer is None:
                    output_dir = os.path.dirname(
                        output_path
                    )

                    if output_dir:
                        os.makedirs(
                            output_dir,
                            exist_ok=True,
                        )

                    fourcc = cv2.VideoWriter_fourcc(
                        *"mp4v"
                    )

                    writer = cv2.VideoWriter(
                        output_path,
                        fourcc,
                        (
                            video_fps
                            if video_fps > 0
                            else 30.0
                        ),
                        (input_width, input_height),
                    )

                    if not writer.isOpened():
                        raise RuntimeError(
                            "Không thể tạo video output: "
                            f"{output_path}"
                        )

                if writer is not None:
                    writer.write(output_frame)
                if show:
                    preview_frame = _fit_frame_for_display(
                        output_frame
                    )

                    cv2.imshow(
                        WINDOW_NAME,
                        preview_frame,
                    )

                    key = cv2.waitKey(1) & 0xFF

                    if (
                        key == ord("q")
                        or cv2.getWindowProperty(
                            WINDOW_NAME,
                            cv2.WND_PROP_VISIBLE,
                        ) < 1
                    ):
                        stop_event.set()
                        break

                frame_count += 1

            processing_thread.join(timeout=2.0)

            if worker_error:
                raise worker_error[0]

            if reader.error is not None:
                raise reader.error

        finally:
            stop_event.set()

            reader.release()

            if writer is not None:
                writer.release()

            if show:
                cv2.destroyAllWindows()

            self.tracker.reset()
            self.smoother.reset()

        print(
            "Threaded pipeline finished. "
            f"Displayed/processed results: {frame_count}, "
            f"inference frames: {processing_count}"
        )

        if output_path is not None:
            print(f"Output video: {output_path}")
