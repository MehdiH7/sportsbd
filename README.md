## sportsbd

**sportsbd: a 4-class 3D CNN-based shot boundary detection toolkit for sports videos.**

This library provides a simple Python and command-line interface around a 3D CNN model (`r2plus1d_18`) for detecting shot boundaries in sports videos. The model predicts four classes:

- **hard** – hard cuts
- **fadein** – fade-in boundaries
- **logo** – logo / overlay-based transitions
- **NaN** – no boundary

### Installation

Install from PyPI:

```bash
pip install sportsbd
```

Or from source:

```bash
git clone https://github.com/mehdih7/sportsbd.git
cd sportsbd
git lfs install  # Install Git LFS (required for model weights)
git lfs pull     # Download model weights
pip install .
```

### Quickstart

#### Python API – run inference on a video

```python
from sportsbd import load_model, run_video_inference

# Load model
model = load_model("data/models/best.pt", device="cuda")

# Run inference on a video
detections = run_video_inference(
    video_path="video.mp4",
    checkpoint_path="data/models/best.pt",
    threshold=0.7,
    stride=4,
    t_frames=16,
    fps=25,
)

# Detections is a list of dicts with:
# - 'frame_idx': frame index
# - 'timestamp_ms': timestamp in milliseconds
# - 'confidence': any-boundary probability
# - 'class_probs': per-class probabilities
print(f"Found {len(detections)} shot boundaries")
```

#### CLI – run inference on a video

```bash
sportsbd infer \
  --video video.mp4 \
  --checkpoint data/models/best.pt \
  --threshold 0.7 \
  --stride 4 \
  --t-frames 16 \
  --fps 25 \
  --out detections.json
```

### Model checkpoints

Model weights should be placed in the `data/models/` directory. The repository includes a pre-trained model at `data/models/best.pt` (358MB).

**Note:** Model weights are tracked with Git LFS due to their size. After cloning, run:
```bash
git lfs install
git lfs pull
```

`sportsbd` expects a PyTorch checkpoint (`.pt`/`.pth`) containing at least:

- **`state_dict`** – model weights compatible with `torchvision.models.video.r2plus1d_18`
- **`config`** – a dictionary with at least:
  - **`MODEL_NAME`** – model architecture name (defaults to `"r2plus1d_18"`)
  - **`NUM_CLASSES`** – number of output classes (defaults to `4`)

You can use any path to your checkpoint file - the examples use `data/models/best.pt` as a convention.

### Development

To run tests:

```bash
pip install -e ".[dev]"
pytest
```


