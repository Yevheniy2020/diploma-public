import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import GoalMarker3D from '../components/GoalMarker3D'
import Path3D from '../components/Path3D'
import SpaceMarker3D from '../components/SpaceMarker3D'
import SimpleRobot3D from '../components/SimpleRobot3D'
import WallInstances from '../components/WallInstances'
import { useAppStore } from '../state/useAppStore'

function Scene3D() {
  const currentMap = useAppStore((s) => s.currentMap)
  const spaces = useAppStore((s) => s.spaces)

  if (!currentMap) return null

  const widthM = currentMap.width_cells * currentMap.cell_size_m
  const heightM = currentMap.height_cells * currentMap.cell_size_m
  const diag = Math.hypot(widthM, heightM)
  const camPos: [number, number, number] = [widthM / 2 + diag * 0.4, diag * 0.7, heightM / 2 + diag * 0.6]
  const target: [number, number, number] = [widthM / 2, 0, heightM / 2]

  return (
    <Canvas
      camera={{ position: camPos, fov: 42, near: 0.1, far: diag * 5 }}
      className="w-full h-full"
      style={{ background: '#fafaf7' }}
    >
      <color attach="background" args={['#fafaf7']} />
      <fog attach="fog" args={['#fafaf7', diag * 0.7, diag * 1.6]} />
      <ambientLight intensity={0.85} />
      <directionalLight position={[widthM * 0.6, diag, heightM * 0.4]} intensity={0.55} />
      <directionalLight position={[-widthM * 0.4, diag * 0.5, -heightM * 0.3]} intensity={0.18} color="#ffe9c4" />

      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[widthM / 2, 0, heightM / 2]}>
        <planeGeometry args={[widthM + 0.4, heightM + 0.4]} />
        <meshStandardMaterial color="#ffffff" roughness={0.95} />
      </mesh>

      <gridHelper
        args={[Math.max(widthM, heightM) + 0.4, Math.round(Math.max(widthM, heightM) * 2), '#d6d3d1', '#e7e5e4']}
        position={[widthM / 2, 0.001, heightM / 2]}
      />

      <WallInstances />
      {spaces.map((r) => (
        <SpaceMarker3D key={r.id} space={r} />
      ))}
      <SimpleRobot3D />
      <Path3D />
      <GoalMarker3D />

      <OrbitControls target={target} makeDefault />
    </Canvas>
  )
}

export default Scene3D
