"""
Velocity Transcript - Live Inference (trigger-based, matches training format)
Press SPACE to trigger a countdown, then perform ONE sign during the fixed
recording window - exactly matching how the model was trained (isolated
clips), rather than continuous streaming (which the model was never
trained to handle and performs unreliably on).

Controls:
    SPACE - start countdown -> record one sign -> predict
    q     - quit

Usage:
    python live_inference.py
"""

import json
import time
from collections import deque

import cv2
import numpy as np
import torch
import mediapipe as mp

from preprocess_single import normalize_hand, resample_sequence, add_velocity_features, FEATURES_PER_HAND, SEQUENCE_LENGTH
from train_baseline import SimpleLSTMClassifier

MODEL_PATH = "models/combined_lstm.pt"
LABEL_MAP_PATH = "models/label_map.json"

COUNTDOWN_SECONDS = 3
RECORD_DURATION_SECONDS = 4  # matches record_own_clips.py - covers our observed sign lengths comfortably

STATE_READY = "ready"
STATE_COUNTDOWN = "countdown"
STATE_RECORDING = "recording"


def predict(model, idx_to_label, buffer):
    sequence = np.array(buffer)
    fixed_sequence = resample_sequence(sequence, target_length=SEQUENCE_LENGTH)
    final_sequence = add_velocity_features(fixed_sequence)

    X_tensor = torch.tensor(final_sequence, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        outputs = model(X_tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        confidence = probs[pred_idx].item()

    return idx_to_label[pred_idx], confidence


def main():
    with open(LABEL_MAP_PATH) as f:
        config = json.load(f)
    idx_to_label = {int(k): v for k, v in config["idx_to_label"].items()}

    model = SimpleLSTMClassifier(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        num_classes=config["num_classes"],
    )
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    state = STATE_READY
    countdown_start_time = None
    recording_start_time = None
    buffer = []
    last_left = [0.0] * FEATURES_PER_HAND
    last_right = [0.0] * FEATURES_PER_HAND

    last_result_text = "Press SPACE to sign"
    sentence = []

    print("Controls: SPACE = countdown + record one sign | q = quit\n")

    while True:
        success, frame = cap.read()
        if not success:
            break

        display_frame = frame.copy()

        if state in (STATE_RECORDING,):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            left_vec = None
            right_vec = None
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    label = handedness.classification[0].label
                    coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                    normalized = normalize_hand(coords)
                    if label == "Left":
                        left_vec = normalized
                    else:
                        right_vec = normalized
                    mp_drawing.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if left_vec is None:
                left_vec = last_left
            else:
                last_left = left_vec
            if right_vec is None:
                right_vec = last_right
            else:
                last_right = right_vec

            buffer.append(left_vec + right_vec)

        if state == STATE_READY:
            cv2.putText(display_frame, "Ready - SPACE to sign", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        elif state == STATE_COUNTDOWN:
            elapsed = time.time() - countdown_start_time
            remaining = COUNTDOWN_SECONDS - elapsed
            if remaining <= 0:
                state = STATE_RECORDING
                recording_start_time = time.time()
                buffer = []
                last_left = [0.0] * FEATURES_PER_HAND
                last_right = [0.0] * FEATURES_PER_HAND
            else:
                count_display = int(remaining) + 1
                text = str(count_display)
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize(text, font, 6.0, 8)
                cx, cy = (frame_width - tw) // 2, (frame_height + th) // 2
                cv2.putText(display_frame, text, (cx, cy), font, 6.0, (0, 0, 0), 14)
                cv2.putText(display_frame, text, (cx, cy), font, 6.0, (0, 165, 255), 8)

        elif state == STATE_RECORDING:
            elapsed_recording = time.time() - recording_start_time
            time_left = max(0, RECORD_DURATION_SECONDS - elapsed_recording)
            cv2.circle(display_frame, (30, 30), 12, (0, 0, 255), -1)
            cv2.putText(display_frame, f"REC  {time_left:.1f}s left", (55, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if elapsed_recording >= RECORD_DURATION_SECONDS:
                word, confidence = predict(model, idx_to_label, buffer)
                last_result_text = f"{word} ({confidence:.2f})"
                print(f"Prediction: {last_result_text}")
                if confidence >= 0.4:
                    sentence.append(word)
                state = STATE_READY

        cv2.putText(display_frame, f"Last: {last_result_text}", (20, frame_height - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(display_frame, " ".join(sentence[-8:]), (20, frame_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Velocity Transcript - Live", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and state == STATE_READY:
            state = STATE_COUNTDOWN
            countdown_start_time = time.time()
        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    print("\nFinal sentence:", " ".join(sentence))


if __name__ == "__main__":
    main()