import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, UnidentifiedImageError
from scipy.ndimage import binary_dilation


@dataclass
class GridImageResult:
    grid_bytes: bytes
    width_cells: int
    height_cells: int


def image_to_grid(
    file_bytes: bytes,
    max_cells: int = 200,
    threshold: int = 128,
    invert: bool = False,
    dilate: int = 0,
) -> GridImageResult:
    try:
        img = Image.open(io.BytesIO(file_bytes))
    except UnidentifiedImageError as e:
        raise ValueError(f"unrecognised image format: {e}")

    img = img.convert("L")

    w, h = img.size
    if w == 0 or h == 0:
        raise ValueError("image has zero dimension")

    scale = max_cells / max(w, h)
    if scale < 1.0:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    arr = np.asarray(img, dtype=np.uint8)
    wall_mask = arr < threshold if not invert else arr >= threshold

    if dilate > 0:
        wall_mask = binary_dilation(wall_mask, iterations=dilate)

    grid = wall_mask.astype(np.uint8)
    height_cells, width_cells = grid.shape
    return GridImageResult(
        grid_bytes=grid.tobytes(),
        width_cells=int(width_cells),
        height_cells=int(height_cells),
    )
