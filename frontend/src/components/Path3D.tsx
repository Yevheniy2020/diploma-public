import { Line } from '@react-three/drei'
import { useAppStore } from '../state/useAppStore'

const PATH_Y = 0.012

export default function Path3D() {
  const robot = useAppStore((s) => s.robot)
  const path = useAppStore((s) => s.path)

  if (path.length === 0) return null

  const points: [number, number, number][] = [
    [robot.x, PATH_Y, robot.y],
    ...path.map((p) => [p.x, PATH_Y, p.y] as [number, number, number]),
  ]

  return <Line points={points} color="#0c0a09" lineWidth={2} transparent opacity={0.85} />
}
