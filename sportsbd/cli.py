from __future__ import annotations

import argparse
from pathlib import Path

from .device import get_available_device, get_device_name
from .download import download_model, DEFAULT_MODEL_PATH
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
    device = args.device

    # Auto-detect device if not specified
    if device is None:
        device = get_available_device()
    else:
        device = get_available_device(prefer=device)
    print(f"[sportsbd] Using device: {get_device_name(device)}")

    # Auto-download model if checkpoint not provided or doesn't exist
    if checkpoint_path is None:
        checkpoint_path = str(DEFAULT_MODEL_PATH)
    
    checkpoint_path_obj = Path(checkpoint_path)
    if not checkpoint_path_obj.is_file():
        print(f"[sportsbd] Model checkpoint not found at {checkpoint_path}")
        print("[sportsbd] Downloading pre-trained model...")
        checkpoint_path_obj = download_model(destination=checkpoint_path_obj)
        checkpoint_path = str(checkpoint_path_obj)
        print(f"[sportsbd] Model downloaded to {checkpoint_path}")

    detections = run_video_inference(
        video_path=video_path,
        checkpoint_path=checkpoint_path,
        threshold=threshold,
        stride=stride,
        t_frames=t_frames,
        fps=fps,
        device=device,
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
    infer_p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (.pt/.pth). If not provided, uses default model and auto-downloads if needed.",
    )
    infer_p.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Confidence threshold on the maximum boundary class probability.",
    )
    infer_p.add_argument("--stride", type=int, default=4, help="Temporal stride (in frames) for sliding window.")
    infer_p.add_argument("--t-frames", type=int, default=16, help="Number of frames per clip.")
    infer_p.add_argument("--fps", type=int, default=25, help="Frame rate for ffmpeg extraction.")
    infer_p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use: 'cuda', 'mps', 'cpu', or None for auto-detection (default: auto)",
    )
    infer_p.add_argument("--out", type=str, required=True, help="Output JSON file for detections.")
    infer_p.set_defaults(func=cmd_infer)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()


