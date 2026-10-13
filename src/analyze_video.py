from scenedetect import detect, ContentDetector 

VIDEO_PATH = "data/input/test.mp4"

scenes = detect(VIDEO_PATH, ContentDetector(), show_progress=True)

for i, (start, end) in enumerate(scenes, start=1):
    print(f'Shot {i}: {start.get_timecode()} -> {end.get_timecode()}')

    
