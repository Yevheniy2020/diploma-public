import { useAppStore } from '../state/useAppStore'

const BODY_RADIUS = 0.18
const BODY_HEIGHT = 0.36
const ARROW_LENGTH = 0.22
const ARROW_RADIUS = 0.07

export default function SimpleRobot3D() {
  const robot = useAppStore((s) => s.robot)

  return (
    <group position={[robot.x, 0, robot.y]} rotation={[0, -robot.theta, 0]}>
      <mesh position={[0, BODY_HEIGHT / 2, 0]} castShadow>
        <cylinderGeometry args={[BODY_RADIUS, BODY_RADIUS, BODY_HEIGHT, 24]} />
        <meshStandardMaterial color="#b91c1c" roughness={0.6} metalness={0.05} />
      </mesh>
      <mesh
        position={[BODY_RADIUS + ARROW_LENGTH / 2, BODY_HEIGHT * 0.6, 0]}
        rotation={[0, 0, -Math.PI / 2]}
      >
        <coneGeometry args={[ARROW_RADIUS, ARROW_LENGTH, 18]} />
        <meshStandardMaterial color="#fafaf7" roughness={0.4} />
      </mesh>
    </group>
  )
}
