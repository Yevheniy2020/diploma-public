import numpy as np


def encode_grid(arr: np.ndarray) -> bytes:
    return np.ascontiguousarray(arr.astype(np.uint8)).tobytes()


def decode_grid(data: bytes, w: int, h: int) -> np.ndarray:
    return np.frombuffer(data, dtype=np.uint8).reshape((h, w))
