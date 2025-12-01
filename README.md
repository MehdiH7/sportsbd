## sportsbd

**sportsbd: a 4-class 3D CNN-based shot boundary detection toolkit for sports videos.**

This library provides a simple Python and command-line interface around a 3D CNN model (`r2plus1d_18`) for detecting shot boundaries in sports videos. The model predicts four classes:

- **hard** – hard cuts
- **fadein** – fade-in boundaries
- **logo** – logo / overlay-based transitions
- **NaN** – no boundary

### Installation

**Prerequisites:** 
- **Python 3.11.14** (recommended, tested and working)
- **PyTorch >= 2.3.0** for MPS Conv3D support (PyTorch 2.9.1 recommended)
- **TorchVision >= 0.18.0** (TorchVision 0.24.1 recommended)
- **NumPy 1.26.4** (tested version)

**Recommended setup (tested and working with MPS Conv3D):**
```bash
# Python 3.11.14 (use pyenv, conda, or your preferred Python version manager)
python --version  # Should show 3.11.14

# Install PyTorch 2.9.1 and TorchVision 0.24.1
pip install torch==2.9.1 torchvision==0.24.1

# Or install latest compatible versions
pip install torch torchvision

# Then install sportsbd
pip install sportsbd
```

Then install `sportsbd`:

**From PyPI:**
```bash
pip install sportsbd
```

**Or from source:**

**Option 1: Install as a package (recommended)**
```bash
git clone https://github.com/mehdih7/sportsbd.git
cd sportsbd
git lfs install  # Install Git LFS (required for model weights)
git lfs pull     # Download model weights
pip install .
```

**Option 2: Install dependencies only**
```bash
git clone https://github.com/mehdih7/sportsbd.git
cd sportsbd
git lfs install
git lfs pull
pip install -r requirements.txt
# Then use: python -m sportsbd.cli or add sportsbd/ to PYTHONPATH
```

**For development:**
```bash
pip install -r requirements-dev.txt
pip install -e .  # Install in editable mode
```

### Quickstart

#### Python API – run inference on a video

```python
from sportsbd import load_model, run_video_inference, get_available_device

# Load model (auto-detects best device: cuda > mps > cpu)
model = load_model("data/models/best.pt")

# Or explicitly specify device
model = load_model("data/models/best.pt", device="mps")  # Apple Silicon
# model = load_model("data/models/best.pt", device="cuda")  # NVIDIA GPU
# model = load_model("data/models/best.pt", device="cpu")   # CPU

# Run inference on a video (auto-detects device if not specified)
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
# - 'confidence': maximum boundary class probability
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

**Note:** The `--device` option is optional. If not specified, `sportsbd` automatically detects the best available device (CUDA > MPS > CPU). 

**Important:** Since the `r2plus1d_18` model uses 3D convolutions, you should use a PyTorch build with MPS support if you want to run on Apple Silicon GPUs. The configuration below has been tested and confirmed to work with MPS Conv3D:
- Python 3.11.14
- PyTorch 2.9.1
- TorchVision 0.24.1
- NumPy 1.26.4

Supported devices:
- `cuda` – NVIDIA GPU (Linux/Windows) - supports Conv3D
- `mps` – Apple Silicon GPU (macOS) - supports Conv3D with PyTorch >= 2.3.0
- `cpu` – CPU fallback - always works

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


