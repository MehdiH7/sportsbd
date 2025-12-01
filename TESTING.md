# Testing Guide for sportsbd

This guide helps you test the package before deploying to PyPI.

## Prerequisites

```bash
# Activate your virtual environment
source venv/bin/activate

# Or if using conda
conda activate sportsbd
```

## 1. Run Unit Tests

```bash
pytest tests/ -v
```

Expected: All 3 tests should pass.

## 2. Test Package Build

Build the package to verify it can be packaged correctly:

```bash
python -m build --wheel --sdist
```

This creates:
- `dist/sportsbd-0.1.0-py3-none-any.whl` (wheel)
- `dist/sportsbd-0.1.0.tar.gz` (source distribution)

## 3. Test Installation from Built Package

Simulate PyPI installation:

```bash
# Create a clean test environment (optional)
python -m venv test_env
source test_env/bin/activate

# Install from wheel (simulates PyPI)
pip install dist/sportsbd-0.1.0-py3-none-any.whl

# Or install from source distribution
pip install dist/sportsbd-0.1.0.tar.gz
```

## 4. Test CLI Commands

```bash
# Test main command
sportsbd --help

# Test infer subcommand
sportsbd infer --help
```

## 5. Test Python API

```python
# Test imports
from sportsbd import load_model, predict_clip, run_video_inference

# Test model loading (if you have a checkpoint)
model = load_model("data/models/best.pt", device="cpu")
print("✓ Model loaded successfully")
```

## 6. Test with Actual Model (Optional)

If you have a test video:

```bash
sportsbd infer \
  --video test_video.mp4 \
  --checkpoint data/models/best.pt \
  --threshold 0.7 \
  --stride 4 \
  --t-frames 16 \
  --fps 25 \
  --out test_detections.json
```

## 7. Verify Package Metadata

```bash
# Check package info
pip show sportsbd

# Check installed files
pip show -f sportsbd
```

## 8. Test Requirements Files

```bash
# Test installing from requirements.txt
pip install -r requirements.txt

# Test installing dev requirements
pip install -r requirements-dev.txt
```

## 9. Check for Common Issues

```bash
# Verify no syntax errors
python -m py_compile sportsbd/**/*.py

# Check imports work
python -c "import sportsbd; print(sportsbd.__all__)"
```

## 10. Test in Clean Environment (Recommended)

To fully simulate a fresh PyPI installation:

```bash
# Create fresh virtual environment
python -m venv fresh_test
source fresh_test/bin/activate

# Install from requirements
pip install -r requirements.txt

# Install package in editable mode
pip install -e .

# Test everything
pytest tests/ -v
sportsbd --help
```

## Pre-PyPI Checklist

- [ ] All tests pass (`pytest tests/ -v`)
- [ ] Package builds successfully (`python -m build`)
- [ ] Can install from wheel (`pip install dist/*.whl`)
- [ ] CLI commands work (`sportsbd --help`, `sportsbd infer --help`)
- [ ] Python API imports work (`from sportsbd import ...`)
- [ ] Model loading works (if checkpoint available)
- [ ] README examples are correct
- [ ] Version number is correct in `pyproject.toml`
- [ ] All dependencies are listed correctly
- [ ] License file is included
- [ ] `.gitignore` excludes build artifacts

## Testing PyPI Upload (TestPyPI)

Before uploading to real PyPI, test on TestPyPI:

```bash
# Install twine
pip install twine

# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test install from TestPyPI
pip install --index-url https://test.pypi.org/simple/ sportsbd
```

