// A small dark-red torus on the floor at the active goal position.
import { useAppStore } from '../state/useAppStore'

export default function GoalMarker3D() {
  const goal = useAppStore((s) => s.currentGoal)
  if (!goal) return null
  return (
    <mesh position={[goal.x, 0.02, goal.y]} rotation={[-Math.PI / 2, 0, 0]}>
      <torusGeometry args={[0.18, 0.025, 12, 32]} />
      <meshBasicMaterial color="#b91c1c" />
    </mesh>
  )
}
