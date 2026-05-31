import numpy as np
from scipy.ndimage import binary_dilation


def inflate(grid: np.ndarray, iterations: int = 2) -> np.ndarray:
    return binary_dilation(grid.astype(bool), iterations=iterations).astype(np.uint8)
