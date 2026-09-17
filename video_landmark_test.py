"""
Velocity Transcript - Step 3
Runs MediaPipe hand-landmark extraction on a single dataset video file
(instead of live webcam) and reports how reliably hands were detected.

Usage:
    python video_landmark_test.py "/path/to/video.MOV"
"""

import sys
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

if len(sys.argv) < 2:
    print("Usage: python video_landmark_test.py <path_to_video>")
    sys.exit(1)

video_path = sys.argv[1]

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video file: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Video: {video_path}")
print(f"Resolution: {width}x{height} | FPS: {fps:.1f} | Total frames: {total_frames}\n")

frame_count = 0
frames_with_zero_hands = 0
frames_with_one_hand = 0
frames_with_two_hands = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    frame_count += 1

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    num_hands = len(results.multi_hand_landmarks) if results.multi_hand_landmarks else 0

    if num_hands == 0:
        frames_with_zero_hands += 1
    elif num_hands == 1:
        frames_with_one_hand += 1
    else:
        frames_with_two_hands += 1

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

    cv2.imshow("Velocity Transcript - Dataset Video Test", frame)

    # Press 'q' to stop early if needed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()

print("---- Detection summary ----")
print(f"Total frames processed: {frame_count}")
print(f"Frames with 0 hands detected: {frames_with_zero_hands} ({100 * frames_with_zero_hands / frame_count:.1f}%)")
print(f"Frames with 1 hand detected:  {frames_with_one_hand} ({100 * frames_with_one_hand / frame_count:.1f}%)")
print(f"Frames with 2 hands detected: {frames_with_two_hands} ({100 * frames_with_two_hands / frame_count:.1f}%)")