"""
Velocity Transcript - Step 6 (updated for full 9-word vocabulary)
Processes every video across our target words into fixed-length,
normalized landmark sequences, and saves the whole dataset as:
    data/processed/X.npy  -> shape (num_samples, 90, 126)
    data/processed/y.npy  -> shape (num_samples,) of string labels

Usage:
    python batch_preprocess.py
"""

import glob
import os
import numpy as np

from preprocess_single import extract_raw_sequence, resample_sequence, SEQUENCE_LENGTH, FEATURES_PER_FRAME
from config import WORD_FOLDERS

OUTPUT_DIR = "data/processed"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    X = []
    y = []
    skipped = []

    for word, folder in WORD_FOLDERS.items():
        video_paths = sorted(glob.glob(os.path.join(folder, "*.MOV")))
        print(f"[{word}] Found {len(video_paths)} clips")

        for path in video_paths:
            try:
                raw_sequence = extract_raw_sequence(path)
                if len(raw_sequence) == 0:
                    skipped.append((path, "no frames read"))
                    continue

                fixed_sequence = resample_sequence(raw_sequence, target_length=SEQUENCE_LENGTH)

                if fixed_sequence.shape != (SEQUENCE_LENGTH, FEATURES_PER_FRAME):
                    skipped.append((path, f"unexpected shape {fixed_sequence.shape}"))
                    continue

                X.append(fixed_sequence)
                y.append(word)

            except Exception as e:
                skipped.append((path, str(e)))

    X = np.array(X)
    y = np.array(y)

    np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)

    print("\n---- Summary ----")
    print(f"Total samples saved: {len(y)}")
    for word in WORD_FOLDERS:
        count = int(np.sum(y == word))
        print(f"  {word}: {count} samples")

    print(f"\nX shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Saved to: {OUTPUT_DIR}/X.npy and {OUTPUT_DIR}/y.npy")

    if skipped:
        print(f"\n{len(skipped)} clips were skipped:")
        for path, reason in skipped:
            print(f"  {path} -> {reason}")


if __name__ == "__main__":
    main()