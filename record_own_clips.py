"""
Velocity Transcript - Step 9 (+ 3-second countdown)
Records your own webcam clips for a given word, saving each take as a
separate video file under data/own_recordings/<Word>/.

Flow per take:
    Press SPACE -> 3-2-1 countdown on screen (gives you time to get
    both hands ready, no rush) -> recording starts automatically ->
    press SPACE again to stop and save.

Controls:
    SPACE - start countdown -> recording / stop the current recording
    q     - quit

Usage:
    python record_own_clips.py "Hello"
    python record_own_clips.py "Good Morning"
"""

import sys
import os
import time
import cv2

FPS = 25  # matches INCLUDE's recording FPS, keeps our pipeline consistent
OUTPUT_ROOT = "data/own_recordings"
COUNTDOWN_SECONDS = 3
RECORD_DURATION_SECONDS = 4  # auto-stops recording after this long - no need to touch the keyboard mid-sign

STATE_READY = "ready"
STATE_COUNTDOWN = "countdown"
STATE_RECORDING = "recording"


def main():
    if len(sys.argv) < 2:
        print('Usage: python record_own_clips.py "<Word>"')
        sys.exit(1)

    word = sys.argv[1]
    word_dir = os.path.join(OUTPUT_ROOT, word)
    os.makedirs(word_dir, exist_ok=True)

    existing_takes = [f for f in os.listdir(word_dir) if f.startswith("take_")]
    next_take_num = len(existing_takes) + 1

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Detect the webcam's actual FPS instead of assuming it matches INCLUDE's 25fps -
    # different cameras report different rates, and a wrong assumption only affects
    # the printed duration / playback speed, not the landmark data itself.
    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    if reported_fps and 1 < reported_fps < 120:
        actual_fps = reported_fps
    else:
        actual_fps = FPS  # fallback if the camera reports something unreliable
        print(f"Warning: webcam reported an unreliable FPS ({reported_fps}), "
              f"falling back to assumed {FPS}fps.")
    print(f"Using FPS: {actual_fps:.1f}")

    print(f"Recording word: '{word}'")
    print(f"Saving to: {word_dir}")
    print(f"Next take will be: take_{next_take_num:02d}.mp4\n")
    print("Controls: SPACE = start countdown/recording, or stop current recording | q = quit\n")

    state = STATE_READY
    writer = None
    frame_count = 0
    countdown_start_time = None
    recording_start_time = None

    while True:
        success, frame = cap.read()
        if not success:
            break

        display_frame = frame.copy()

        if state == STATE_READY:
            cv2.putText(display_frame, f"Ready - take_{next_take_num:02d} - SPACE to start",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        elif state == STATE_COUNTDOWN:
            elapsed = time.time() - countdown_start_time
            remaining = COUNTDOWN_SECONDS - elapsed

            if remaining <= 0:
                # Countdown finished - start actual recording now
                take_path = os.path.join(word_dir, f"take_{next_take_num:02d}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(take_path, fourcc, actual_fps, (frame_width, frame_height))
                state = STATE_RECORDING
                frame_count = 0
                recording_start_time = time.time()
                print(f"Recording started: take_{next_take_num:02d}.mp4 (auto-stops in {RECORD_DURATION_SECONDS}s)")
            else:
                count_display = int(remaining) + 1  # shows 3, 2, 1

                # Large, centered countdown number - impossible to miss
                text = str(count_display)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 6.0
                thickness = 8
                (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                center_x = (frame_width - text_width) // 2
                center_y = (frame_height + text_height) // 2

                # Dark outline first (for visibility against any background), then orange fill
                cv2.putText(display_frame, text, (center_x, center_y),
                            font, font_scale, (0, 0, 0), thickness + 6)
                cv2.putText(display_frame, text, (center_x, center_y),
                            font, font_scale, (0, 165, 255), thickness)

                cv2.putText(display_frame, "Get ready...", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

        elif state == STATE_RECORDING:
            writer.write(frame)
            frame_count += 1
            elapsed_recording = time.time() - recording_start_time
            time_left = max(0, RECORD_DURATION_SECONDS - elapsed_recording)

            cv2.circle(display_frame, (30, 30), 12, (0, 0, 255), -1)
            cv2.putText(display_frame, f"REC  {time_left:.1f}s left  (frames: {frame_count})",
                        (55, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if elapsed_recording >= RECORD_DURATION_SECONDS:
                writer.release()
                state = STATE_READY
                duration_sec = frame_count / actual_fps
                print(f"Saved take_{next_take_num:02d}.mp4 - {frame_count} frames (~{duration_sec:.1f}s) [auto-stopped]\n")
                next_take_num += 1

        cv2.imshow(f"Recording: {word}", display_frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):
            if state == STATE_READY:
                state = STATE_COUNTDOWN
                countdown_start_time = time.time()
            elif state == STATE_RECORDING:
                writer.release()
                state = STATE_READY
                duration_sec = frame_count / actual_fps
                print(f"Saved take_{next_take_num:02d}.mp4 - {frame_count} frames (~{duration_sec:.1f}s)\n")
                next_take_num += 1
            # If SPACE pressed during countdown, ignore it (avoids accidental double-trigger)

        elif key == ord("q"):
            if state == STATE_RECORDING:
                writer.release()
                print(f"Saved take_{next_take_num:02d}.mp4 (stopped by quit)")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDone. Total takes for '{word}': {next_take_num - 1}")


if __name__ == "__main__":
    main()