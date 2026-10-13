import json
import sys
from pathlib import Path

from video_analyzer import VideoAnalyzer

video_path = sys.argv[1]

analyzer = VideoAnalyzer(video_path)
analysis = analyzer.analyze(output_dir="data/output")

output_directory = Path("data/output")
output_directory.mkdir(parents=True, exist_ok=True)

output_path = output_directory / "analysis.json"

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(analysis, file, indent=2)

print(f"Analysis saved to {output_path}")