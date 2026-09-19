"""
Velocity Transcript - Step 5 (+ velocity features)
Processes ONE video into a normalized, fixed-length landmark sequence
ready for model training. Also provides add_velocity_features() to
augment the position sequence with frame-to-frame motion, as an
experiment to help the model generalize better across recording
sessions (see session_split.py findings).

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
FEATURES_PER_FRAME = FEATURES_PER_HAND * 2  # 126 (left hand + right hand) - position only
FEATURES_PER_FRAME_WITH_VELOCITY = FEATURES_PER_FRAME * 2  # 252 (position + velocity)


def normalize_hand(landmarks):
    """
    landmarks: list of 21 (x, y, z) tuples for one hand.
    Returns a flat list of 63 normalized values:
    centered on the wrist, scaled by wrist-to-middle-finger-base distance.
    """
    coords = np.array(landmarks)  # shape (21, 3)
    wrist = coords[0]
    centered = coords - wrist  # translation invariance

    scale_ref = np.linalg.norm(centered[9])
    if scale_ref < 1e-6:
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
                label = handedness.classification[0].label
                coords = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
                normalized = normalize_hand(coords)

                if label == "Left":
                    left_vec = normalized
                else:
                    right_vec = normalized

        if left_vec is None:
            left_vec = last_left
        else:
            last_left = left_vec

        if right_vec is None:
            right_vec = last_right
        else:
            last_right = right_vec

        frame_features = left_vec + right_vec
        sequence.append(frame_features)

    cap.release()
    hands.close()
    return sequence


def resample_sequence(sequence, target_length=SEQUENCE_LENGTH):
    """
    Uniformly samples `target_length` frames from the sequence,
    preserving the full motion regardless of original clip length.
    """
    sequence = np.array(sequence)
    original_length = sequence.shape[0]

    if original_length == target_length:
        return sequence

    indices = np.linspace(0, original_length - 1, target_length)
    indices = np.round(indices).astype(int)
    return sequence[indices]


def add_velocity_features(fixed_sequence):
    """
    Takes a (target_length, 126) position sequence and returns a
    (target_length, 252) sequence: original position concatenated
    with frame-to-frame velocity (position difference).

    The first frame's velocity is zero (no previous frame to diff against).
    """
    velocity = np.diff(fixed_sequence, axis=0, prepend=fixed_sequence[0:1])
    return np.concatenate([fixed_sequence, velocity], axis=1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preprocess_single.py <path_to_video>")
        sys.exit(1)

    video_path = sys.argv[1]

    print(f"Processing: {video_path}")
    raw_sequence = extract_raw_sequence(video_path)
    print(f"Raw sequence length: {len(raw_sequence)} frames")

    fixed_sequence = resample_sequence(raw_sequence)
    print(f"Resampled sequence shape (position only): {fixed_sequence.shape}")

    with_velocity = add_velocity_features(fixed_sequence)
    print(f"With velocity features: {with_velocity.shape}")  # should be (90, 252)

    print(f"\nMin/Max (position only): {fixed_sequence.min():.3f} / {fixed_sequence.max():.3f}")
    print(f"Min/Max (with velocity): {with_velocity.min():.3f} / {with_velocity.max():.3f}")