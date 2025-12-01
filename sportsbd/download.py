from __future__ import annotations

from pathlib import Path
from typing import Optional

from tqdm import tqdm
from urllib.request import urlopen


DEFAULT_MODEL_URL = (
    "https://github.com/mehdih7/sportsbd/releases/download/v0.1.0/best.pt"
)
DEFAULT_MODEL_PATH = Path("data/models/best.pt")


def download_model(
    destination: str | Path | None = None,
    url: Optional[str] = None,
    overwrite: bool = False,
    progress: bool = True,
) -> Path:
    """
    Download the pre-trained sportsbd model checkpoint.

    This helper is intended for users who installed `sportsbd` from PyPI and do
    not have the `best.pt` weights locally.

    By default it downloads to `data/models/best.pt` in the current working
    directory, matching the paths used in the examples and CLI.

    Args:
        destination:
            Target file path. If None, uses ``data/models/best.pt`` relative
            to the current working directory.
        url:
            Optional custom URL to download from. If None, uses the default
            GitHub release URL defined in ``DEFAULT_MODEL_URL``.
        overwrite:
            If True, overwrite an existing file at the destination. If False
            (default), an existing file is left untouched and its path is
            returned immediately.
        progress:
            If True (default), display a tqdm progress bar while downloading.

    Returns:
        Path to the downloaded (or existing) checkpoint file.
    """
    dest = Path(destination) if destination is not None else DEFAULT_MODEL_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.is_file() and not overwrite:
        return dest

    download_url = url or DEFAULT_MODEL_URL

    with urlopen(download_url) as resp:  # type: ignore[call-arg]
        total_size = int(resp.headers.get("Content-Length", 0))
        chunk_size = 1024 * 1024  # 1 MB

        pbar: Optional[tqdm] = None
        if progress and total_size > 0:
            pbar = tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=f"Downloading model to {dest}",
            )

        with dest.open("wb") as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                if pbar is not None:
                    pbar.update(len(chunk))

        if pbar is not None:
            pbar.close()

    return dest


