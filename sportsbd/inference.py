from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

from .config import DEFAULT_CONFIG
from .model import load_model
from .transforms import get_frame_transform, normalize_tensor_frame

FrameLike = Union[Image.Image, torch.Tensor]


def _ensure_tensor_frame(frame: FrameLike) -> torch.Tensor:
    if isinstance(frame, Image.Image):
        transform = get_frame_transform()
        return transform(frame)
    elif isinstance(frame, torch.Tensor):
        if frame.ndim == 3:
            # Assume (C, H, W) in [0,1] or unnormalized; normalize
            return normalize_tensor_frame(frame.float())
        elif frame.ndim == 2:
            # Single-channel; expand to 3 channels
            frame = frame.unsqueeze(0).repeat(3, 1, 1)
            return normalize_tensor_frame(frame.float())
        else:
            raise ValueError(f"Unsupported tensor frame shape {tuple(frame.shape)}; expected (C,H,W) or (H,W).")
    else:
        raise TypeError(f"Unsupported frame type: {type(frame)}")


def predict_clip(
    frames: Sequence[FrameLike],
    model: torch.nn.Module,
    device: Union[str, torch.device] = "cuda",
) -> Dict[str, Any]:
    """
    Run inference on a single clip (sequence of frames).

    Returns a dict with:
      - 'class_probs': list of per-class probabilities
      - 'any_boundary_prob': probability of any boundary (sum of first three classes)
      - 'predicted_index': argmax class index
      - 'predicted_class': class name (if available)
    """
    if len(frames) == 0:
        raise ValueError("predict_clip requires at least one frame.")

    device = torch.device(device) if isinstance(device, str) else device

    tensor_frames = [_ensure_tensor_frame(f) for f in frames]
    clip = torch.stack(tensor_frames, dim=0)  # (T, C, H, W)
    clip = clip.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)

    clip = clip.to(device)
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(clip)  # (1, num_classes)
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()

    any_boundary_prob = float(np.sum(probs[:3]))

    class_index = int(np.argmax(probs))
    class_name = None
    if 0 <= class_index < len(DEFAULT_CONFIG.class_names):
        class_name = DEFAULT_CONFIG.class_names[class_index]

    return {
        "class_probs": probs.tolist(),
        "any_boundary_prob": any_boundary_prob,
        "predicted_index": class_index,
        "predicted_class": class_name,
    }


def _extract_frames_with_ffmpeg(
    video_path: Path,
    fps: int,
    output_dir: Path,
) -> List[Path]:
    """
    Extract frames from a video into output_dir using ffmpeg.
    Returns a sorted list of extracted frame paths.
    """
    output_pattern = str(output_dir / "frame_%06d.jpg")
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps}",
        output_pattern,
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg failed with exit code {e.returncode}") from e

    frame_paths = sorted(output_dir.glob("frame_*.jpg"))
    if not frame_paths:
        raise RuntimeError(f"No frames were extracted from video: {video_path}")
    return frame_paths


def run_video_inference(
    video_path: str | Path,
    checkpoint_path: str | Path,
    threshold: float = 0.7,
    stride: int = 4,
    t_frames: int = 16,
    fps: int = DEFAULT_CONFIG.fps,
    device: Union[str, torch.device] = "cuda",
    progress: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run sliding-window inference over a full video.

    Returns a list of detections:
      { 'frame_idx', 'timestamp_ms', 'confidence', 'class_probs' }
    where confidence is the 'any-boundary' probability.
    """
    video_path = Path(video_path)
    checkpoint_path = Path(checkpoint_path)

    model = load_model(checkpoint_path, device=device)
    device = torch.device(device) if isinstance(device, str) else device

    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        frame_paths = _extract_frames_with_ffmpeg(video_path, fps=fps, output_dir=tmpdir)

        detections: List[Dict[str, Any]] = []
        num_frames = len(frame_paths)

        iterator = range(0, max(0, num_frames - t_frames + 1), stride)
        if progress:
            iterator = tqdm(iterator, desc="sportsbd inference", unit="window")

        for start_idx in iterator:
            window_paths = frame_paths[start_idx : start_idx + t_frames]
            if len(window_paths) < t_frames:
                break

            frames: List[Image.Image] = [Image.open(p).convert("RGB") for p in window_paths]
            result = predict_clip(frames, model=model, device=device)

            confidence = float(result["any_boundary_prob"])
            if confidence < threshold:
                continue

            class_probs = result["class_probs"]
            # Use the center frame as the representative boundary index
            center_idx = start_idx + t_frames // 2
            timestamp_ms = int(math.floor(center_idx / fps * 1000.0))

            detections.append(
                {
                    "frame_idx": int(center_idx),
                    "timestamp_ms": timestamp_ms,
                    "confidence": confidence,
                    "class_probs": class_probs,
                }
            )

        return detections


def save_detections_to_json(detections: List[Dict[str, Any]], output_path: str | Path) -> None:
    """
    Save detections to a JSON file.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(detections, f, indent=2)


def load_detections_from_json(path: str | Path) -> List[Dict[str, Any]]:
    """
    Load detections from a JSON file.
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Detections JSON must contain a list.")
    return data


