// Renders one space as a translucent floor polygon + outline + billboarded
// name label. World coords (x_east, y_north) map to three (x, _, z), so a
// 2D Shape defined with negated y lands flat on the X-Z floor after a
// -π/2 X rotation.
import { useMemo } from 'react'
import * as THREE from 'three'
import { Billboard, Line, Text } from '@react-three/drei'
import type { SpaceResponse } from '../types'

interface Props {
  space: SpaceResponse
}

// 6-step palette mixed with the home colour. Picked by space id so a space's
// hue stays stable across renders and matches the 2D scene's palette.
// [fill (low alpha), stroke, label] — kept hex; r3f / drei consume strings.
const PALETTE: { fill: string; stroke: string; label: string }[] = [
  { fill: '#3b82f6', stroke: '#1d4ed8', label: '#1e3a8a' }, // blue
  { fill: '#10b981', stroke: '#047857', label: '#064e3b' }, // emerald
  { fill: '#8b5cf6', stroke: '#6d28d9', label: '#4c1d95' }, // violet
  { fill: '#f43f5e', stroke: '#be123c', label: '#881337' }, // rose
  { fill: '#06b6d4', stroke: '#0e7490', label: '#155e75' }, // cyan
  { fill: '#a16207', stroke: '#854d0e', label: '#713f12' }, // mustard
]

const HOME_COLOR = { fill: '#f59e0b', stroke: '#b45309', label: '#78350f' }

export function roomColor(space: SpaceResponse): { fill: string; stroke: string; label: string } {
  if (space.is_home) return HOME_COLOR
  return PALETTE[space.id % PALETTE.length]
}

function SpaceMarker3D({ space }: Props) {
  // Build the Shape once per polygon. Negated y so a -π/2 X rotation lands
  // the geometry on the X-Z plane with the correct orientation.
  const shape = useMemo(() => {
    const s = new THREE.Shape()
    if (space.vertices.length === 0) return s
    const [v0x, v0y] = space.vertices[0]
    s.moveTo(v0x, -v0y)
    for (let i = 1; i < space.vertices.length; i++) {
      const [vx, vy] = space.vertices[i]
      s.lineTo(vx, -vy)
    }
    s.closePath()
    return s
  }, [space.vertices])

  // Polygon outline as a closed line in three-coord space (vx, y_offset, vy).
  const outline = useMemo(() => {
    const y = 0.012 + (space.is_home ? 0.002 : 0)
    const pts: [number, number, number][] = space.vertices.map(([vx, vy]) => [vx, y, vy])
    if (pts.length > 0) pts.push(pts[0])
    return pts
  }, [space.vertices, space.is_home])

  // Vertex-average centroid (matches the 2D scene + the planner) for the
  // billboarded name label.
  const centroid = useMemo<[number, number, number]>(() => {
    if (space.vertices.length === 0) return [0, 0, 0]
    let sx = 0
    let sy = 0
    for (const [vx, vy] of space.vertices) {
      sx += vx
      sy += vy
    }
    // Float above WALL_HEIGHT (0.6 m in WallInstances) so the billboarded
    // name doesn't get hidden inside a wall / furniture block that
    // happens to sit at the polygon's geometric centroid.
    return [sx / space.vertices.length, 1.1, sy / space.vertices.length]
  }, [space.vertices])

  if (space.vertices.length < 3) return null

  const palette = roomColor(space)
  // Home gets a denser fill + a slight y-bias so it sits visually on top.
  const fillOpacity = space.is_home ? 0.28 : 0.16
  const floorY = 0.008 + (space.is_home ? 0.002 : 0)
  const label = space.is_home ? `★ ${space.name}` : space.name

  return (
    <group>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, floorY, 0]}>
        <shapeGeometry args={[shape]} />
        <meshBasicMaterial
          color={palette.fill}
          transparent
          opacity={fillOpacity}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>
      <Line points={outline} color={palette.stroke} lineWidth={1.5} />
      <Billboard position={centroid}>
        <Text
          fontSize={0.28}
          color={palette.label}
          anchorX="center"
          anchorY="middle"
          outlineWidth={0.012}
          outlineColor="#fafaf7"
        >
          {label}
        </Text>
      </Billboard>
    </group>
  )
}

export default SpaceMarker3D
