"""
Velocity Transcript - diagnostic
Prints the sorted unique file numbers and the gaps between consecutive
ones, so we can pick a GAP_THRESHOLD based on evidence instead of a guess.

Usage:
    python inspect_gaps.py
"""

import glob
import os
import re

from config import WORD_FOLDERS


def parse_number(filename):
    match = re.search(r"MVI_(\d+)", filename)
    if not match:
        return None
    return int(match.group(1))


all_numbers = set()

for word, folder in WORD_FOLDERS.items():
    video_paths = sorted(glob.glob(os.path.join(folder, "*.MOV")))
    for path in video_paths:
        number = parse_number(os.path.basename(path))
        if number is not None:
            all_numbers.add(number)

sorted_numbers = sorted(all_numbers)

print(f"Total unique numbers: {len(sorted_numbers)}\n")
print("Sorted numbers with gaps to next:")
for i in range(len(sorted_numbers) - 1):
    gap = sorted_numbers[i + 1] - sorted_numbers[i]
    marker = "  <-- big gap" if gap > 10 else ""
    print(f"  {sorted_numbers[i]:6d}  (gap to next: {gap:4d}){marker}")
print(f"  {sorted_numbers[-1]:6d}  (last)")

# Summary of gap sizes to help pick a threshold
gaps = [sorted_numbers[i+1] - sorted_numbers[i] for i in range(len(sorted_numbers) - 1)]
print(f"\nGap size distribution:")
print(f"  min gap: {min(gaps)}")
print(f"  max gap: {max(gaps)}")
sorted_gaps = sorted(gaps, reverse=True)
print(f"  10 largest gaps: {sorted_gaps[:10]}")
