import os

import cv2


def extract_frames(video_path, output_folder, fps=5):
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Erro ao abrir o vídeo: {video_path}")
        return

    frame_rate = max(1, int(cap.get(cv2.CAP_PROP_FPS) / fps))
    count = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_rate == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_count += 1

        count += 1

    cap.release()

def process_all_videos(input_dir, output_dir, fps=5):
    os.makedirs(output_dir, exist_ok=True)

    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith(".mp4"):
            video_path = os.path.join(input_dir, filename)
            video_name = os.path.splitext(filename)[0]  # Remove a extensão .mp4
            output_folder = os.path.join(output_dir, video_name)
            extract_frames(video_path, output_folder, fps)

# Diretórios
input_directory = "/home/kelson/Projects/MBA/TCC/fall_detection/experiment1/videos/fail/"
output_directory = "/home/kelson/Projects/MBA/TCC/fall_detection/experiment1/videos/fail/frames/"

# Processar todos os vídeos
process_all_videos(input_directory, output_directory)
