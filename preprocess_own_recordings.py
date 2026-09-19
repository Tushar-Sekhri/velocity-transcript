"""
Velocity Transcript - Step 10
Processes your own recorded clips (data/own_recordings/<Word>/take_XX.mp4)
using the exact same pipeline as the INCLUDE data, saving:
    data/processed/X_own.npy
    data/processed/y_own.npy

Usage:
    python preprocess_own_recordings.py
"""

import glob
import os
import numpy as np

from preprocess_single import (
    extract_raw_sequence,
    resample_sequence,
    add_velocity_features,
    SEQUENCE_LENGTH,
    FEATURES_PER_FRAME_WITH_VELOCITY,
)

OWN_RECORDINGS_DIR = "data/own_recordings"
OUTPUT_DIR = "data/processed"


def main():
    if not os.path.isdir(OWN_RECORDINGS_DIR):
        print(f"No folder found at {OWN_RECORDINGS_DIR} - nothing to process.")
        return

    word_folders = sorted(
        d for d in os.listdir(OWN_RECORDINGS_DIR)
        if os.path.isdir(os.path.join(OWN_RECORDINGS_DIR, d))
    )

    X = []
    y = []
    skipped = []

    for word in word_folders:
        folder = os.path.join(OWN_RECORDINGS_DIR, word)
        video_paths = sorted(glob.glob(os.path.join(folder, "*.mp4")))
        print(f"[{word}] Found {len(video_paths)} clips")

        for path in video_paths:
            try:
                raw_sequence = extract_raw_sequence(path)
                if len(raw_sequence) == 0:
                    skipped.append((path, "no frames read"))
                    continue

                fixed_sequence = resample_sequence(raw_sequence, target_length=SEQUENCE_LENGTH)
                final_sequence = add_velocity_features(fixed_sequence)

                if final_sequence.shape != (SEQUENCE_LENGTH, FEATURES_PER_FRAME_WITH_VELOCITY):
                    skipped.append((path, f"unexpected shape {final_sequence.shape}"))
                    continue

                X.append(final_sequence)
                y.append(word)

            except Exception as e:
                skipped.append((path, str(e)))

    X = np.array(X)
    y = np.array(y)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.save(os.path.join(OUTPUT_DIR, "X_own.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y_own.npy"), y)

    print("\n---- Summary ----")
    print(f"Total samples saved: {len(y)}")
    for word in word_folders:
        count = int(np.sum(y == word))
        print(f"  {word}: {count} samples")

    print(f"\nX_own shape: {X.shape}")
    print(f"y_own shape: {y.shape}")
    print(f"Saved to: {OUTPUT_DIR}/X_own.npy and y_own.npy")

    if skipped:
        print(f"\n{len(skipped)} clips were skipped:")
        for path, reason in skipped:
            print(f"  {path} -> {reason}")


if __name__ == "__main__":
    main()