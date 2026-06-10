import { useMemo } from 'react'
import * as THREE from 'three'
import { Billboard, Line, Text } from '@react-three/drei'
import type { SpaceResponse } from '../types'

interface Props {
  space: SpaceResponse
}

const PALETTE: { fill: string; stroke: string; label: string }[] = [
  { fill: '#3b82f6', stroke: '#1d4ed8', label: '#1e3a8a' },
  { fill: '#10b981', stroke: '#047857', label: '#064e3b' },
  { fill: '#8b5cf6', stroke: '#6d28d9', label: '#4c1d95' },
  { fill: '#f43f5e', stroke: '#be123c', label: '#881337' },
  { fill: '#06b6d4', stroke: '#0e7490', label: '#155e75' },
  { fill: '#a16207', stroke: '#854d0e', label: '#713f12' },
]

const HOME_COLOR = { fill: '#f59e0b', stroke: '#b45309', label: '#78350f' }

export function roomColor(space: SpaceResponse): { fill: string; stroke: string; label: string } {
  if (space.is_home) return HOME_COLOR
  return PALETTE[space.id % PALETTE.length]
}

function SpaceMarker3D({ space }: Props) {
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

  const outline = useMemo(() => {
    const y = 0.012 + (space.is_home ? 0.002 : 0)
    const pts: [number, number, number][] = space.vertices.map(([vx, vy]) => [vx, y, vy])
    if (pts.length > 0) pts.push(pts[0])
    return pts
  }, [space.vertices, space.is_home])

  const centroid = useMemo<[number, number, number]>(() => {
    if (space.vertices.length === 0) return [0, 0, 0]
    let sx = 0
    let sy = 0
    for (const [vx, vy] of space.vertices) {
      sx += vx
      sy += vy
    }
    return [sx / space.vertices.length, 1.1, sy / space.vertices.length]
  }, [space.vertices])

  if (space.vertices.length < 3) return null

  const palette = roomColor(space)
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
