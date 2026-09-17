"""
Velocity Transcript - Step 7 (v2: size-aware cluster selection)
1. Parses file numbers from every clip across our target words.
2. Groups them into clusters based on numeric proximity - a proxy for
   separate recording batches/sessions (NOT verified signer identity -
   see inspect_gaps.py findings: gaps only reliably separate recording
   days, not individual signers within a continuous block).
3. Greedily selects whole clusters for the test set, by total SAMPLE
   COUNT (not cluster count), to land close to TEST_FRACTION of the
   overall dataset while keeping every cluster intact.
4. Saves train/test indices aligned with data/processed/X.npy and y.npy.

Usage:
    python session_split.py
"""

import glob
import os
import re
import numpy as np

from config import WORD_FOLDERS

GAP_THRESHOLD = 10
TEST_FRACTION = 0.25  # target fraction of TOTAL SAMPLES in test, not clusters


def parse_number(filename):
    match = re.search(r"MVI_(\d+)", filename)
    if not match:
        return None
    return int(match.group(1))


def cluster_numbers(numbers):
    sorted_numbers = sorted(numbers)
    cluster_id = 0
    number_to_cluster = {sorted_numbers[0]: cluster_id}
    for prev, curr in zip(sorted_numbers, sorted_numbers[1:]):
        if curr - prev > GAP_THRESHOLD:
            cluster_id += 1
        number_to_cluster[curr] = cluster_id
    return number_to_cluster


def main():
    all_entries = []
    all_numbers = set()

    for word, folder in WORD_FOLDERS.items():
        video_paths = sorted(glob.glob(os.path.join(folder, "*.MOV")))
        for path in video_paths:
            number = parse_number(os.path.basename(path))
            if number is None:
                print(f"Warning: couldn't parse number from {path}")
                continue
            all_entries.append((word, path, number))
            all_numbers.add(number)

    number_to_cluster = cluster_numbers(all_numbers)
    num_clusters = len(set(number_to_cluster.values()))
    print(f"Detected {num_clusters} clusters (recording batches/sessions) "
          f"across {len(all_numbers)} unique file numbers.\n")

    cluster_word_counts = {}
    cluster_sample_counts = {}
    for word, path, number in all_entries:
        cluster = number_to_cluster[number]
        cluster_word_counts.setdefault(cluster, {}).setdefault(word, 0)
        cluster_word_counts[cluster][word] += 1
        cluster_sample_counts[cluster] = cluster_sample_counts.get(cluster, 0) + 1

    total_samples = sum(cluster_sample_counts.values())

    for cluster_id in sorted(cluster_word_counts.keys()):
        counts = cluster_word_counts[cluster_id]
        print(f"  Cluster {cluster_id} ({cluster_sample_counts[cluster_id]} samples): {counts}")

    # ---- Size-aware greedy selection for test set ----
    # Sort clusters smallest-first, keep adding whole clusters to test
    # until we're as close as possible to TEST_FRACTION of total samples.
    target_test_size = total_samples * TEST_FRACTION
    clusters_by_size = sorted(cluster_sample_counts.items(), key=lambda x: x[1])

    test_clusters = set()
    running_total = 0
    for cluster_id, size in clusters_by_size:
        if running_total + size <= target_test_size * 1.15:  # allow slight overshoot
            test_clusters.add(cluster_id)
            running_total += size
        if running_total >= target_test_size * 0.85:
            break

    train_clusters = set(cluster_sample_counts.keys()) - test_clusters

    print(f"\nTarget test size: ~{target_test_size:.0f} samples ({TEST_FRACTION*100:.0f}%)")
    print(f"Train clusters: {sorted(train_clusters)}")
    print(f"Test clusters:  {sorted(test_clusters)}")

    train_indices = []
    test_indices = []
    sample_index = 0

    for word, folder in WORD_FOLDERS.items():
        video_paths = sorted(glob.glob(os.path.join(folder, "*.MOV")))
        for path in video_paths:
            number = parse_number(os.path.basename(path))
            if number is None:
                continue
            cluster = number_to_cluster[number]
            if cluster in test_clusters:
                test_indices.append(sample_index)
            else:
                train_indices.append(sample_index)
            sample_index += 1

    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)

    os.makedirs("data/processed", exist_ok=True)
    np.save("data/processed/train_indices.npy", train_indices)
    np.save("data/processed/test_indices.npy", test_indices)

    print(f"\nTrain samples: {len(train_indices)} ({100*len(train_indices)/total_samples:.1f}%)")
    print(f"Test samples:  {len(test_indices)} ({100*len(test_indices)/total_samples:.1f}%)")
    print("Saved to data/processed/train_indices.npy and test_indices.npy")


if __name__ == "__main__":
    main()