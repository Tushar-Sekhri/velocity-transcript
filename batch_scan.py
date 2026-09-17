"""
Velocity Transcript - Step 4
Scans every video across our chosen target words and reports:
- number of clips per word
- min / max / average frame count per word
- average hand-detection rate per word

No live video window here (kept fast on purpose) - this is a batch
data-analysis pass, not a visual check.

Usage:
    python batch_scan.py
"""

import glob
import os
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands

# Map: word label -> folder path
# Adjust these paths if your folder names differ.
WORD_FOLDERS = {
    "Hello": "data/raw/Greetings_1of2/Greetings/48. Hello",
    "How are you": "data/raw/Greetings_1of2/Greetings/49. How are you",
    "Thank you": "data/raw/Greetings_2of2/Greetings/55. Thank you",
}

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)


def analyze_video(video_path):
    """Returns (total_frames, zero_hand_frames, one_hand_frames, two_hand_frames)."""
    cap = cv2.VideoCapture(video_path)
    total_frames = 0
    zero_hands = 0
    one_hand = 0
    two_hands = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        total_frames += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

        if num_hands == 0:
            zero_hands += 1
        elif num_hands == 1:
            one_hand += 1
        else:
            two_hands += 1

    cap.release()
    return total_frames, zero_hands, one_hand, two_hands


print("Scanning videos... this may take a few minutes.\n")

for word, folder in WORD_FOLDERS.items():
    video_paths = sorted(glob.glob(os.path.join(folder, "*.MOV")))

    if not video_paths:
        print(f"[{word}] No .MOV files found in: {folder}\n")
        continue

    frame_counts = []
    zero_hand_ratios = []

    for path in video_paths:
        total, zero, one, two = analyze_video(path)
        if total == 0:
            print(f"  Warning: could not read {path}, skipping.")
            continue
        frame_counts.append(total)
        zero_hand_ratios.append(zero / total)

    print(f"[{word}]")
    print(f"  Clips found: {len(video_paths)}")
    print(f"  Frame count -> min: {min(frame_counts)}, max: {max(frame_counts)}, "
          f"avg: {sum(frame_counts) / len(frame_counts):.1f}")
    print(f"  Avg zero-hand-detection rate: {100 * sum(zero_hand_ratios) / len(zero_hand_ratios):.1f}%\n")

hands.close()
print("Done.")