## Changelog

### 0.1.2

- **Automatic model download**: CLI now automatically downloads the pre-trained model if not found
- `--checkpoint` argument is now optional (defaults to `data/models/best.pt`)
- Improved user experience - users can run inference without manually downloading the model

### 0.1.1

- Added `download_model()` function for downloading pre-trained model from GitHub releases
- Fixed model download URL to use correct GitHub username case

### 0.1.0

- Initial release of `sportsbd`.
- Core Python API for:
  - Loading 3D CNN checkpoints (`r2plus1d_18`).
  - Running shot boundary detection on clips and full videos.
- Command-line interface with `infer` subcommand for video inference.


