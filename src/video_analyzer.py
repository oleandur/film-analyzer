from pathlib import Path

import cv2
from scenedetect import detect, ContentDetector

class VideoAnalyzer:
    def __init__(self, video_path: str):
        self.video_path = Path(video_path)

    def analyze(self) -> dict:
        scenes = detect(str(self.video_path), ContentDetector(), show_progress=True)

        video_info = self._get_video_info()
        shots = []

        for index, (start, end) in enumerate(scenes, start=1):
            start_seconds = start.get_seconds()
            end_seconds = end.get_seconds()

            shots.append({
                "id": index,
                "start": start_seconds,
                "end": end_seconds,
                "duration": end_seconds - start_seconds
            })

        return {
            "video": {
                "filename": self.video_path.name,
                **video_info
            },
            "shots": shots
        }

    def _get_video_info(self) -> dict:
        capture = cv2.VideoCapture(str(self.video_path))

        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)

        capture.release()
        duration = frame_count / fps if fps else 0

        return {
            "duration": duration,
            "fps": fps,
            "frame_count": int(frame_count)
        }