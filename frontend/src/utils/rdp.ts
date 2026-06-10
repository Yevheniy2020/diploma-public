type Pt = [number, number]

function perpendicularDistance(p: Pt, a: Pt, b: Pt): number {
  const [px, py] = p
  const [ax, ay] = a
  const [bx, by] = b
  const dx = bx - ax
  const dy = by - ay
  const len2 = dx * dx + dy * dy
  if (len2 === 0) {
    const ex = px - ax
    const ey = py - ay
    return Math.hypot(ex, ey)
  }
  const cross = dx * (ay - py) - dy * (ax - px)
  return Math.abs(cross) / Math.sqrt(len2)
}

export function simplify(points: Pt[], epsilon: number): Pt[] {
  if (points.length < 3) return points.slice()
  const a = points[0]
  const b = points[points.length - 1]
  let maxDist = 0
  let index = 0
  for (let i = 1; i < points.length - 1; i++) {
    const d = perpendicularDistance(points[i], a, b)
    if (d > maxDist) {
      maxDist = d
      index = i
    }
  }
  if (maxDist > epsilon) {
    const left = simplify(points.slice(0, index + 1), epsilon)
    const right = simplify(points.slice(index), epsilon)
    return left.slice(0, -1).concat(right)
  }
  return [a, b]
}
