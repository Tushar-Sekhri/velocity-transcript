"""
Velocity Transcript - Step 2
Extracts (x, y, z) coordinates for all 21 hand landmarks, per hand, per frame,
and prints them live. This is the raw numeric data our model will eventually consume.

Press 'q' to quit.
"""

import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check camera permissions.")

print("Webcam opened. Press 'q' in the video window to quit.\n")

frame_count = 0

while True:
    success, frame = cap.read()
    if not success:
        print("Failed to read frame from webcam.")
        break

    frame_count += 1

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        # Loop over each detected hand (could be 0, 1, or 2)
        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            # Extract all 21 landmarks as a flat list of (x, y, z) tuples
            coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

            # Print just the wrist (landmark 0) and index fingertip (landmark 8)
            # so the terminal output is readable instead of a wall of numbers.
            wrist = coords[0]
            index_tip = coords[8]

            # Which hand MediaPipe thinks this is (Left/Right)
            handedness = results.multi_handedness[hand_index].classification[0].label

            print(
                f"Frame {frame_count} | Hand {hand_index} ({handedness}) | "
                f"wrist=({wrist[0]:.3f}, {wrist[1]:.3f}, {wrist[2]:.3f}) | "
                f"index_tip=({index_tip[0]:.3f}, {index_tip[1]:.3f}, {index_tip[2]:.3f})"
            )

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style(),
            )

    cv2.imshow("Velocity Transcript - Landmark Extraction", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
hands.close()