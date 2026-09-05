from collections import defaultdict, deque, Counter
from typing import List, Dict, Any, Optional, Set
import numpy as np
import supervision as sv

from .config import (
    TRACK_LOST_BUFFER,
    TRACK_ACTIVATION_THRESHOLD,
    TRACK_MATCHING_THRESHOLD,
    TRACKER_FRAME_RATE,
    SMOOTHING_WINDOW_SIZE,
    SMOOTHING_MAX_MISSING_FRAMES
)


class PersonTracker:
    def __init__(
        self,
        track_activation_threshold: float = TRACK_ACTIVATION_THRESHOLD,
        lost_track_buffer: int = TRACK_LOST_BUFFER,
        minimum_matching_threshold: float = TRACK_MATCHING_THRESHOLD,
        frame_rate: int = TRACKER_FRAME_RATE
    ):
        try:
            self.byte_tracker = sv.ByteTrack(
                track_activation_threshold=track_activation_threshold,
                lost_track_buffer=lost_track_buffer,
                minimum_matching_threshold=minimum_matching_threshold,
                frame_rate=frame_rate
            )
        except Exception:
            self.byte_tracker = sv.ByteTrack(
                track_thresh=track_activation_threshold,
                track_buffer=lost_track_buffer,
                match_thresh=minimum_matching_threshold,
                frame_rate=frame_rate
            )

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not detections:
            empty_detections = sv.Detections.empty()
            tracked = self.byte_tracker.update_with_detections(empty_detections)
            return self._convert_result(tracked)

        xyxy = np.array([d["bbox"] for d in detections], dtype=np.float32)
        confidence = np.array([d["confidence"] for d in detections], dtype=np.float32)
        class_id = np.array([d["class_id"] for d in detections], dtype=int)

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id
        )

        tracked_detections = self.byte_tracker.update_with_detections(sv_detections)
        return self._convert_result(tracked_detections)

    def _convert_result(self, tracked_detections) -> List[Dict[str, Any]]:
        results = []
        if tracked_detections is None or len(tracked_detections) == 0:
            return results

        boxes = tracked_detections.xyxy
        confidences = tracked_detections.confidence
        class_ids = tracked_detections.class_id
        tracker_ids = tracked_detections.tracker_id

        for i in range(len(tracked_detections)):
            x1, y1, x2, y2 = map(int, boxes[i])
            track_id = int(tracker_ids[i]) if tracker_ids is not None else -1
            confidence = float(confidences[i]) if confidences is not None else 0.0
            class_id = int(class_ids[i]) if class_ids is not None else 0

            results.append({
                "track_id": track_id,
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence,
                "class_id": class_id
            })

        return results

    def reset(self):
        self.byte_tracker.reset()


class AttributeSmoother:

    def __init__(
        self,
        window_size: int = SMOOTHING_WINDOW_SIZE,
        max_missing_frames: int = SMOOTHING_MAX_MISSING_FRAMES
    ):
        self.window_size = window_size
        self.max_missing_frames = max_missing_frames

        self.history = defaultdict(
            lambda: defaultdict(
                lambda: deque(maxlen=self.window_size)
            )
        )
        self.missing_frames = defaultdict(int)

    def update(
        self,
        track_id: int,
        attributes: Dict[str, Any]
    ) -> Dict[str, Any]:
        if track_id < 0:
            return attributes

        self.missing_frames[track_id] = 0

        if not attributes:
            return {}

        smoothed = {}

        for attr_name, value in attributes.items():
            self.history[track_id][attr_name].append(value)
            smoothed[attr_name] = self._majority_vote(
                self.history[track_id][attr_name]
            )

        return smoothed

    def _majority_vote(self, values: deque) -> Any:
        if not values:
            return None
        counter = Counter(values)
        return counter.most_common(1)[0][0]

    def mark_missing(self, active_track_ids: Set[int]):
        all_track_ids = set(self.history.keys())

        for track_id in all_track_ids:
            if track_id in active_track_ids:
                self.missing_frames[track_id] = 0
            else:
                self.missing_frames[track_id] += 1

        self._remove_old_tracks()

    def _remove_old_tracks(self):
        remove_ids = [
            t_id for t_id, missing_count in self.missing_frames.items()
            if missing_count > self.max_missing_frames
        ]

        for track_id in remove_ids:
            self.history.pop(track_id, None)
            self.missing_frames.pop(track_id, None)

    def reset(self):
        self.history.clear()
        self.missing_frames.clear()


class TrackerSmootherManager:
    def __init__(
        self,
        window_size: int = SMOOTHING_WINDOW_SIZE,
        max_missing_frames: int = SMOOTHING_MAX_MISSING_FRAMES
    ):
        self.tracker = PersonTracker()
        self.smoother = AttributeSmoother(
            window_size=window_size,
            max_missing_frames=max_missing_frames
        )

    def update_tracks(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        tracked_objects = self.tracker.update(detections)
        active_ids = {obj["track_id"] for obj in tracked_objects if obj["track_id"] >= 0}
        self.smoother.mark_missing(active_ids)
        return tracked_objects

    def smooth_attributes(self, track_id: int, attributes: Dict[str, Any]) -> Dict[str, Any]:
        return self.smoother.update(track_id, attributes)

    def reset(self):
        self.tracker.reset()
        self.smoother.reset()