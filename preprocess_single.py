"""
Velocity Transcript - Step 5
Processes ONE video into a normalized, fixed-length landmark sequence
ready for model training. Prints the resulting shape and some sanity
stats so we can verify correctness before batch-processing everything.

Usage:
    python preprocess_single.py "data/raw/.../video.MOV"
"""

import sys
import numpy as np
import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands

SEQUENCE_LENGTH = 90  # fixed number of frames per sample, chosen from our batch_scan.py stats
NUM_LANDMARKS = 21
COORDS_PER_LANDMARK = 3  # x, y, z
FEATURES_PER_HAND = NUM_LANDMARKS * COORDS_PER_LANDMARK  # 63
FEATURES_PER_FRAME = FEATURES_PER_HAND * 2  # 126 (left hand + right hand)


def normalize_hand(landmarks):
    """
    landmarks: list of 21 (x, y, z) tuples for one hand.
    Returns a flat list of 63 normalized values:
    centered on the wrist, scaled by wrist-to-middle-finger-base distance.
    """
    coords = np.array(landmarks)  # shape (21, 3)
    wrist = coords[0]
    centered = coords - wrist  # translation invariance

    # landmark 9 = middle finger MCP (base knuckle) - used as a stable scale reference
    scale_ref = np.linalg.norm(centered[9])
    if scale_ref < 1e-6:  # avoid divide-by-zero on degenerate frames
        scale_ref = 1e-6

    normalized = centered / scale_ref
    return normalized.flatten().tolist()  # 63 values


def extract_raw_sequence(video_path):
    """
    Runs MediaPipe over every frame of the video.
    Returns a list of per-frame feature vectors (126 values each),
    BEFORE fixed-length resampling.
    """
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    # Carry-forward buffers - start as zeros until each hand is first seen
    last_left = [0.0] * FEATURES_PER_HAND
    last_right = [0.0] * FEATURES_PER_HAND

    sequence = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        left_vec = None
        right_vec = None

        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks, results.multi_handedness
            ):
                label = handedness.classification[0].label  # "Left" or "Right"
                coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                normalized = normalize_hand(coords)

                if label == "Left":
                    left_vec = normalized
                else:
                    right_vec = normalized

        # Carry forward if a hand wasn't detected this frame
        if left_vec is None:
            left_vec = last_left
        else:
            last_left = left_vec

        if right_vec is None:
            right_vec = last_right
        else:
            last_right = right_vec

        frame_features = left_vec + right_vec  # 63 + 63 = 126
        sequence.append(frame_features)

    cap.release()
    hands.close()
    return sequence


def resample_sequence(sequence, target_length=SEQUENCE_LENGTH):
    """
    Uniformly samples `target_length` frames from the sequence,
    preserving the full motion regardless of original clip length.
    """
    sequence = np.array(sequence)  # shape (original_length, 126)
    original_length = sequence.shape[0]

    if original_length == target_length:
        return sequence

    indices = np.linspace(0, original_length - 1, target_length)
    indices = np.round(indices).astype(int)
    return sequence[indices]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocess_single.py <path_to_video>")
        sys.exit(1)

    video_path = sys.argv[1]

    print(f"Processing: {video_path}")
    raw_sequence = extract_raw_sequence(video_path)
    print(f"Raw sequence length: {len(raw_sequence)} frames")

    fixed_sequence = resample_sequence(raw_sequence)
    print(f"Resampled sequence shape: {fixed_sequence.shape}")  # should be (90, 126)

    print(f"\nSample values (frame 0, first 6 features): {fixed_sequence[0][:6]}")
    print(f"Sample values (frame 45, first 6 features): {fixed_sequence[45][:6]}")
    print(f"\nMin value in sequence: {fixed_sequence.min():.3f}")
    print(f"Max value in sequence: {fixed_sequence.max():.3f}")