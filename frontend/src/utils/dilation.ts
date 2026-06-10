export function dilateGrid(
  grid: Uint8Array,
  width: number,
  height: number,
  radius = 2,
): Uint8Array {
  const out = new Uint8Array(grid)
  const r2 = radius * radius
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (grid[y * width + x] !== 1) continue
      for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          if (dx * dx + dy * dy > r2) continue
          const nx = x + dx
          const ny = y + dy
          if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue
          out[ny * width + nx] = 1
        }
      }
    }
  }
  return out
}
