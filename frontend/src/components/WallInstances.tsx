import { useEffect, useMemo, useRef } from 'react'
import * as THREE from 'three'
import { useAppStore } from '../state/useAppStore'
import { decodeGrid } from '../utils/grid'

const WALL_HEIGHT = 0.6

export default function WallInstances() {
  const currentMap = useAppStore((s) => s.currentMap)
  const meshRef = useRef<THREE.InstancedMesh>(null)

  const positions = useMemo(() => {
    if (!currentMap?.grid_data_b64) return null
    const { width_cells: W, height_cells: H, cell_size_m: c } = currentMap
    const grid = decodeGrid(currentMap.grid_data_b64)
    const pts: [number, number, number][] = []
    for (let row = 0; row < H; row++) {
      for (let col = 0; col < W; col++) {
        if (grid[row * W + col] === 1) {
          pts.push([(col + 0.5) * c, WALL_HEIGHT / 2, (H - 1 - row + 0.5) * c])
        }
      }
    }
    return pts
  }, [currentMap])

  useEffect(() => {
    const mesh = meshRef.current
    if (!mesh || !positions) return
    const dummy = new THREE.Object3D()
    for (let i = 0; i < positions.length; i++) {
      dummy.position.set(positions[i][0], positions[i][1], positions[i][2])
      dummy.updateMatrix()
      mesh.setMatrixAt(i, dummy.matrix)
    }
    mesh.count = positions.length
    mesh.instanceMatrix.needsUpdate = true
  }, [positions])

  if (!currentMap || currentMap.scene_glb_path) return null
  if (!positions || positions.length === 0) return null

  const c = currentMap.cell_size_m

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, positions.length]}
      frustumCulled={false}
    >
      <boxGeometry args={[c, WALL_HEIGHT, c]} />
      <meshStandardMaterial color="#1c1917" roughness={0.85} metalness={0.05} />
    </instancedMesh>
  )
}
