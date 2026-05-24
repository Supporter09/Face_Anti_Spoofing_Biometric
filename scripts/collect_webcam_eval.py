#!/usr/bin/env python3
"""Collect fixed-duration webcam frame sequences for local FAS evaluation."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect webcam frames for liveness evaluation.")
    parser.add_argument("--category", required=True, choices=["live", "print_photo", "phone_screen_replay"])
    parser.add_argument("--out", default="data_collected/webcam_eval")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--duration", type=int, default=5)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--notes", default="")
    return parser.parse_args()


def draw_overlay(frame, text: str) -> None:
    cv2.putText(
        frame,
        text,
        (32, 64),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (0, 255, 255),
        3,
        cv2.LINE_AA,
    )


def main() -> int:
    args = parse_args()
    if args.fps <= 0 or args.duration <= 0:
        print("--fps and --duration must be positive.", file=sys.stderr)
        return 2

    cap = cv2.VideoCapture(args.device)
    if not cap.isOpened():
        print(f"Could not open camera device {args.device}.", file=sys.stderr)
        return 1

    timestamp = datetime.now()
    timestamp_dir = timestamp.strftime("%Y%m%d_%H%M%S")
    session_dir = Path(args.out) / args.category / timestamp_dir
    frames_dir = session_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    try:
        for remaining in range(3, 0, -1):
            deadline = cv2.getTickCount() + int(cv2.getTickFrequency())
            while cv2.getTickCount() < deadline:
                ok, frame = cap.read()
                if not ok:
                    print("Failed to read frame during countdown.", file=sys.stderr)
                    return 1
                draw_overlay(frame, f"Recording in {remaining}")
                cv2.imshow("webcam_eval", frame)
                if cv2.waitKey(1) == 27:
                    return 130

        total_frames = args.duration * args.fps
        frame_period_ms = int(1000 / args.fps)
        while saved < total_frames:
            start_tick = cv2.getTickCount()
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame during recording.", file=sys.stderr)
                return 1
            cv2.imwrite(str(frames_dir / f"frame_{saved:04d}.jpg"), frame)
            saved += 1
            preview = frame.copy()
            draw_overlay(preview, f"REC {saved}/{total_frames}")
            cv2.imshow("webcam_eval", preview)
            if cv2.waitKey(1) == 27:
                break
            elapsed_ms = int((cv2.getTickCount() - start_tick) * 1000 / cv2.getTickFrequency())
            delay_ms = max(1, frame_period_ms - elapsed_ms)
            if cv2.waitKey(delay_ms) == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    meta = {
        "category": args.category,
        "timestamp": timestamp.isoformat(timespec="seconds"),
        "fps": args.fps,
        "duration_s": args.duration,
        "frame_count": saved,
        "device": args.device,
        "notes": args.notes,
    }
    with (session_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Saved {saved} frames to {session_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
