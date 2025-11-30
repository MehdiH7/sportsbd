from __future__ import annotations

import argparse
from pathlib import Path

from .inference import (
    run_video_inference,
    save_detections_to_json,
)


def cmd_infer(args: argparse.Namespace) -> None:
    video_path = args.video
    checkpoint_path = args.checkpoint
    threshold = args.threshold
    stride = args.stride
    t_frames = args.t_frames
    fps = args.fps
    out = args.out

    detections = run_video_inference(
        video_path=video_path,
        checkpoint_path=checkpoint_path,
        threshold=threshold,
        stride=stride,
        t_frames=t_frames,
        fps=fps,
    )

    save_detections_to_json(detections, out)
    print(f"[sportsbd] Saved {len(detections)} detections to {out}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sportsbd",
        description="sportsbd: 3D CNN-based shot boundary detection for sports videos.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # infer
    infer_p = subparsers.add_parser("infer", help="Run shot boundary detection on a video.")
    infer_p.add_argument("--video", type=str, required=True, help="Path to input video.")
    infer_p.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt/.pth).")
    infer_p.add_argument("--threshold", type=float, default=0.7, help="Any-boundary confidence threshold.")
    infer_p.add_argument("--stride", type=int, default=4, help="Temporal stride (in frames) for sliding window.")
    infer_p.add_argument("--t-frames", type=int, default=16, help="Number of frames per clip.")
    infer_p.add_argument("--fps", type=int, default=25, help="Frame rate for ffmpeg extraction.")
    infer_p.add_argument("--out", type=str, required=True, help="Output JSON file for detections.")
    infer_p.set_defaults(func=cmd_infer)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()


