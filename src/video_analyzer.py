from pathlib import Path

import cv2
from scenedetect import detect, ContentDetector

class VideoAnalyzer:
    def __init__(self, video_path: str):
        self.video_path = Path(video_path)

    def analyze(self, output_dir: str | Path = "data/output") -> dict:
        output_dir = Path(output_dir)

        scenes = detect(str(self.video_path), ContentDetector(), show_progress=True)

        video_info = self._get_video_info()

        capture = cv2.VideoCapture(str(self.video_path))

        if not capture.isOpened():
            raise RuntimeError(f'Could not open video: {self.video_path}')
        shots = []
        try:

            for index, (start, end) in enumerate(scenes, start=1):
                start_seconds = start.get_seconds()
                end_seconds = end.get_seconds()

                thumbnail = self._extract_thumbnail(
                    capture=capture,
                    shot_id=index,
                    start_seconds=start_seconds,
                    end_seconds=end_seconds,
                    output_dir=output_dir
                )

                shots.append({
                    "id": index,
                    "start": start_seconds,
                    "end": end_seconds,
                    "duration": end_seconds - start_seconds,
                    "thumbnail": thumbnail
                })
        finally:
            capture.release()

        statistics = self._calculate_shot_statistics(shots)
        editing_pace = self._calculate_editing_pace(
            shots,
            video_info["duration"]
        )

        return {
            "video": {
                "filename": self.video_path.name,
                **video_info
            },
            "statistics": statistics,
            "editing_pace": editing_pace,
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
    
    def _extract_thumbnail(
        self, 
        capture: cv2.VideoCapture, 
        shot_id: int,
        start_seconds: float,
        end_seconds: float,
        output_dir: Path    
    ) -> str:
        midpoint = (start_seconds + end_seconds) / 2

        capture.set(cv2.CAP_PROP_POS_MSEC, midpoint*1000)

        success, frame = capture.read()

        if not success:
            raise RuntimeError(f'Could not extract thumbnail for shot {shot_id}')

        shots_dir = output_dir / "shots"
        shots_dir.mkdir(parents=True, exist_ok=True)

        filename = f'shot_{shot_id:03d}.jpg'
        thumbnail_path = shots_dir / filename

        success = cv2.imwrite(str(thumbnail_path), frame)

        if not success:
            raise RuntimeError(f'Could not save thumbnail for shot {shot_id}')

        return str(Path("shots") / filename)

    def _calculate_shot_statistics(self, shots: list[dict]) -> dict:
        if not shots:
            return {
                "shot_count": 0,
                "average_shot_duration": 0,
                "shortest_shot_duration": 0,
                "longest_shot_duration": 0
            }

        durations = [shot["duration"] for shot in shots]

        return {
            "shot_count": len(shots),
            "average_shot_duration": sum(durations) / len(durations),
            "shortest_shot_duration": min(durations),
            "longest_shot_duration": max(durations)
        }
    def _calculate_editing_pace(
        self,
        shots: list[dict],
        video_duration: float
    ) -> dict:
        if video_duration <= 0:
            return {}

        segment_duration = video_duration / 3

        segments = {
            "beginning": [],
            "middle": [],
            "ending": []
        }

        for shot in shots:
            midpoint = (shot["start"] + shot["end"]) / 2

            if midpoint < segment_duration:
                segments["beginning"].append(shot)
            elif midpoint < segment_duration * 2:
                segments["middle"].append(shot)
            else:
                segments["ending"].append(shot)

        result = {}

        for name, segment_shots in segments.items():
            durations = [
                shot["duration"]
                for shot in segment_shots
            ]

            shot_count = len(segment_shots)

            result[name] = {
                "shot_count": shot_count,
                "average_shot_duration": (
                    sum(durations) / shot_count
                    if shot_count
                    else 0
                ),
                "shots_per_minute": (
                    shot_count / segment_duration * 60
                    if segment_duration
                    else 0
                )
            }

        return result

