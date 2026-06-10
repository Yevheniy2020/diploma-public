export function decodeGrid(b64: string): Uint8Array {
  const binary = atob(b64)
  const arr = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) arr[i] = binary.charCodeAt(i)
  return arr
}

export function encodeGrid(arr: Uint8Array): string {
  let binary = ''
  const chunk = 0x8000
  for (let i = 0; i < arr.length; i += chunk) {
    binary += String.fromCharCode(...arr.subarray(i, i + chunk))
  }
  return btoa(binary)
}

export function cellToWorld(
  col: number,
  row: number,
  height: number,
  cellSize: number,
): { x: number; y: number } {
  return {
    x: (col + 0.5) * cellSize,
    y: (height - 1 - row + 0.5) * cellSize,
  }
}

export function worldToCell(
  x: number,
  y: number,
  height: number,
  cellSize: number,
): { col: number; row: number } {
  return {
    col: Math.floor(x / cellSize),
    row: height - 1 - Math.floor(y / cellSize),
  }
}
