import { useCallback, useEffect, useMemo, useRef } from 'react'
import { plan, updateMap } from '../api/client'
import { useT } from '../i18n'
import { useAppStore } from '../state/useAppStore'
import { dilateGrid } from '../utils/dilation'
import { decodeGrid, encodeGrid, worldToCell } from '../utils/grid'
import { toast } from '../state/useToaster'

const PADDING = 28

const ROOM_PALETTE: { fillRGB: string; stroke: string; label: string }[] = [
  { fillRGB: '59, 130, 246', stroke: '#1d4ed8', label: '#1e3a8a' },
  { fillRGB: '16, 185, 129', stroke: '#047857', label: '#064e3b' },
  { fillRGB: '139, 92, 246', stroke: '#6d28d9', label: '#4c1d95' },
  { fillRGB: '244, 63, 94', stroke: '#be123c', label: '#881337' },
  { fillRGB: '6, 182, 212', stroke: '#0e7490', label: '#155e75' },
  { fillRGB: '161, 98, 7', stroke: '#854d0e', label: '#713f12' },
]

function Scene2D() {
  const t = useT()
  const currentMap = useAppStore((s) => s.currentMap)
  const editMode = useAppStore((s) => s.editMode)

  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const paintingRef = useRef<0 | 1 | null>(null)
  const lastCellRef = useRef<number>(-1)
  const saveTimerRef = useRef<number | null>(null)

  const decoded = useMemo(() => {
    if (!currentMap?.grid_data_b64) return null
    return decodeGrid(currentMap.grid_data_b64)
  }, [currentMap?.grid_data_b64])

  const inflated = useMemo(() => {
    if (!currentMap || !decoded) return null
    return dilateGrid(decoded, currentMap.width_cells, currentMap.height_cells, 2)
  }, [currentMap, decoded])

  const projection = useCallback(() => {
    const cvs = canvasRef.current
    const cont = containerRef.current
    if (!cvs || !cont || !currentMap) return null
    const W = cont.clientWidth
    const H = cont.clientHeight
    const widthM = currentMap.width_cells * currentMap.cell_size_m
    const heightM = currentMap.height_cells * currentMap.cell_size_m
    const scale = Math.min((W - PADDING * 2) / widthM, (H - PADDING * 2) / heightM)
    const ox = (W - widthM * scale) / 2
    const oy = (H - heightM * scale) / 2
    return { W, H, widthM, heightM, scale, ox, oy }
  }, [currentMap])

  useEffect(() => {
    if (!currentMap) return
    const cvs = canvasRef.current
    const cont = containerRef.current
    if (!cvs || !cont) return
    const ctx = cvs.getContext('2d')
    if (!ctx) return

    let raf = 0
    const draw = () => {
      const proj = projection()
      if (!proj) {
        raf = requestAnimationFrame(draw)
        return
      }
      const { W, H, widthM, heightM, scale, ox, oy } = proj
      const dpr = window.devicePixelRatio || 1
      if (cvs.width !== W * dpr || cvs.height !== H * dpr) {
        cvs.width = W * dpr
        cvs.height = H * dpr
        cvs.style.width = `${W}px`
        cvs.style.height = `${H}px`
      }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, W, H)

      const c = currentMap.cell_size_m
      const cs = c * scale
      const xToPx = (mx: number) => ox + mx * scale
      const yToPx = (my: number) => oy + (heightM - my) * scale

      ctx.fillStyle = '#ffffff'
      ctx.fillRect(ox, oy, widthM * scale, heightM * scale)

      const s = useAppStore.getState()

      if (s.showGrid2D) {
        ctx.strokeStyle = '#e7e5e4'
        ctx.lineWidth = 0.5
        for (let i = 0; i <= currentMap.width_cells; i += 5) {
          const x = xToPx(i * c)
          ctx.beginPath()
          ctx.moveTo(x, oy)
          ctx.lineTo(x, oy + heightM * scale)
          ctx.stroke()
        }
        for (let j = 0; j <= currentMap.height_cells; j += 5) {
          const y = yToPx(j * c)
          ctx.beginPath()
          ctx.moveTo(ox, y)
          ctx.lineTo(ox + widthM * scale, y)
          ctx.stroke()
        }
      }

      const sortedSpaces = [...s.spaces].sort(
        (a, b) => Number(a.is_home) - Number(b.is_home),
      )
      for (const space of sortedSpaces) {
        if (space.vertices.length < 3) continue
        const palette = space.is_home
          ? { fillRGB: '245, 158, 11', stroke: '#b45309', label: '#78350f' }
          : ROOM_PALETTE[space.id % ROOM_PALETTE.length]
        ctx.beginPath()
        const [vx0, vy0] = space.vertices[0]
        ctx.moveTo(xToPx(vx0), yToPx(vy0))
        for (let i = 1; i < space.vertices.length; i++) {
          const [vx, vy] = space.vertices[i]
          ctx.lineTo(xToPx(vx), yToPx(vy))
        }
        ctx.closePath()
        ctx.fillStyle = `rgba(${palette.fillRGB}, ${space.is_home ? 0.18 : 0.12})`
        ctx.fill()
        ctx.strokeStyle = palette.stroke
        ctx.lineWidth = space.is_home ? 1.5 : 1.2
        ctx.stroke()
      }

      if (s.showInflation && inflated && decoded) {
        ctx.fillStyle = 'rgba(180, 83, 9, 0.10)'
        for (let row = 0; row < currentMap.height_cells; row++) {
          for (let col = 0; col < currentMap.width_cells; col++) {
            const idx = row * currentMap.width_cells + col
            if (inflated[idx] === 1 && decoded[idx] === 0) {
              const wx = col * c
              const wy = (currentMap.height_cells - 1 - row) * c
              ctx.fillRect(xToPx(wx), yToPx(wy + c), cs + 0.5, cs + 0.5)
            }
          }
        }
      }

      if (decoded) {
        ctx.fillStyle = '#1c1917'
        for (let row = 0; row < currentMap.height_cells; row++) {
          for (let col = 0; col < currentMap.width_cells; col++) {
            if (decoded[row * currentMap.width_cells + col] === 1) {
              const wx = col * c
              const wy = (currentMap.height_cells - 1 - row) * c
              ctx.fillRect(xToPx(wx), yToPx(wy + c), cs + 0.5, cs + 0.5)
            }
          }
        }
      }

      ctx.strokeStyle = '#0c0a09'
      ctx.lineWidth = 1.5
      ctx.strokeRect(ox - 0.5, oy - 0.5, widthM * scale + 1, heightM * scale + 1)

      for (const space of sortedSpaces) {
        if (space.vertices.length < 3) continue
        const palette = space.is_home
          ? { fillRGB: '245, 158, 11', stroke: '#b45309', label: '#78350f' }
          : ROOM_PALETTE[space.id % ROOM_PALETTE.length]
        let cx = 0
        let cy = 0
        for (const [vx, vy] of space.vertices) {
          cx += vx
          cy += vy
        }
        cx /= space.vertices.length
        cy /= space.vertices.length

        const label = space.is_home ? `★ ${space.name}` : space.name
        ctx.font = `${space.is_home ? '600' : '500'} 12.5px "IBM Plex Sans", sans-serif`
        ctx.textAlign = 'center'
        ctx.textBaseline = 'middle'
        ctx.lineWidth = 4
        ctx.strokeStyle = 'rgba(250, 250, 247, 0.95)'
        ctx.strokeText(label, xToPx(cx), yToPx(cy))
        ctx.fillStyle = palette.label
        ctx.fillText(label, xToPx(cx), yToPx(cy))
      }

      if (s.path.length > 0) {
        ctx.strokeStyle = '#a8a29e'
        ctx.lineWidth = 1.5
        ctx.setLineDash([4, 4])
        ctx.beginPath()
        ctx.moveTo(xToPx(s.robot.x), yToPx(s.robot.y))
        for (const p of s.path) ctx.lineTo(xToPx(p.x), yToPx(p.y))
        ctx.stroke()
        ctx.setLineDash([])

        ctx.strokeStyle = '#0c0a09'
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.moveTo(xToPx(s.robot.x), yToPx(s.robot.y))
        for (let i = s.pathIndex; i < s.path.length; i++) {
          const p = s.path[i]
          ctx.lineTo(xToPx(p.x), yToPx(p.y))
        }
        ctx.stroke()
      }

      if (s.currentGoal) {
        const gx = xToPx(s.currentGoal.x)
        const gy = yToPx(s.currentGoal.y)
        ctx.strokeStyle = '#b91c1c'
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.arc(gx, gy, 8, 0, Math.PI * 2)
        ctx.stroke()
        ctx.beginPath()
        ctx.moveTo(gx - 12, gy)
        ctx.lineTo(gx + 12, gy)
        ctx.moveTo(gx, gy - 12)
        ctx.lineTo(gx, gy + 12)
        ctx.stroke()
      }

      if (s.draftSpace && s.draftSpace.points.length > 0) {
        const pts = s.draftSpace.points
        ctx.strokeStyle = 'rgba(249, 115, 22, 0.85)'
        ctx.fillStyle = 'rgba(249, 115, 22, 0.10)'
        ctx.lineWidth = 2
        ctx.setLineDash([5, 5])
        ctx.beginPath()
        const [px0, py0] = pts[0]
        ctx.moveTo(xToPx(px0), yToPx(py0))
        for (let i = 1; i < pts.length; i++) {
          const [pxi, pyi] = pts[i]
          ctx.lineTo(xToPx(pxi), yToPx(pyi))
        }
        if (pts.length >= 3) {
          ctx.closePath()
          ctx.fill()
        }
        ctx.stroke()
        ctx.setLineDash([])
        ctx.fillStyle = '#ea580c'
        for (const [pxi, pyi] of pts) {
          ctx.beginPath()
          ctx.arc(xToPx(pxi), yToPx(pyi), 3, 0, Math.PI * 2)
          ctx.fill()
        }
        ctx.font = 'italic 600 11px "IBM Plex Sans", sans-serif'
        ctx.fillStyle = '#9a3412'
        ctx.textAlign = 'left'
        ctx.textBaseline = 'bottom'
        ctx.fillText(
          `▶ recording «${s.draftSpace.name}» · ${pts.length} pts`,
          xToPx(px0) + 6,
          yToPx(py0) - 6,
        )
      }

      const robotR = 0.15 * scale
      const rx = xToPx(s.robot.x)
      const ry = yToPx(s.robot.y)
      ctx.fillStyle = 'rgba(185, 28, 28, 0.12)'
      ctx.beginPath()
      ctx.arc(rx, ry, robotR, 0, Math.PI * 2)
      ctx.fill()
      ctx.strokeStyle = '#b91c1c'
      ctx.lineWidth = 1
      ctx.stroke()
      ctx.fillStyle = '#b91c1c'
      ctx.beginPath()
      ctx.arc(rx, ry, robotR * 0.55, 0, Math.PI * 2)
      ctx.fill()
      const cosT = Math.cos(s.robot.theta)
      const sinT = -Math.sin(s.robot.theta)
      ctx.fillStyle = '#fff'
      ctx.beginPath()
      ctx.moveTo(rx + cosT * robotR * 0.95, ry + sinT * robotR * 0.95)
      const a1 = s.robot.theta + 2.4
      const a2 = s.robot.theta - 2.4
      ctx.lineTo(rx + Math.cos(a1) * robotR * 0.4, ry - Math.sin(a1) * robotR * 0.4)
      ctx.lineTo(rx + Math.cos(a2) * robotR * 0.4, ry - Math.sin(a2) * robotR * 0.4)
      ctx.closePath()
      ctx.fill()

      ctx.strokeStyle = '#0c0a09'
      ctx.lineWidth = 1
      const axOx = ox
      const axOy = oy + heightM * scale
      ctx.beginPath()
      ctx.moveTo(axOx - 6, axOy)
      ctx.lineTo(axOx + 16, axOy)
      ctx.moveTo(axOx, axOy + 6)
      ctx.lineTo(axOx, axOy - 16)
      ctx.stroke()
      ctx.fillStyle = '#57534e'
      ctx.font = '500 9.5px "IBM Plex Mono", monospace'
      ctx.textAlign = 'left'
      ctx.textBaseline = 'top'
      ctx.fillText('+x', axOx + 18, axOy - 12)
      ctx.fillText('+y', axOx + 2, axOy - 28)

      const sbY = oy + heightM * scale + 16
      ctx.strokeStyle = '#0c0a09'
      ctx.lineWidth = 1.5
      ctx.beginPath()
      ctx.moveTo(ox, sbY)
      ctx.lineTo(ox + scale, sbY)
      ctx.stroke()
      ctx.beginPath()
      ctx.moveTo(ox, sbY - 4)
      ctx.lineTo(ox, sbY + 4)
      ctx.moveTo(ox + scale, sbY - 4)
      ctx.lineTo(ox + scale, sbY + 4)
      ctx.stroke()
      ctx.fillStyle = '#0c0a09'
      ctx.font = '500 10px "IBM Plex Mono", monospace'
      ctx.textAlign = 'left'
      ctx.textBaseline = 'top'
      ctx.fillText('1.00 m', ox + scale + 6, sbY - 6)

      raf = requestAnimationFrame(draw)
    }
    raf = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(raf)
  }, [currentMap, decoded, inflated, projection])

  const eventToWorld = useCallback(
    (e: React.MouseEvent<HTMLCanvasElement>): { wx: number; wy: number } | null => {
      if (!currentMap) return null
      const proj = projection()
      if (!proj) return null
      const { ox, oy, widthM, heightM, scale } = proj
      const rect = (e.currentTarget as HTMLCanvasElement).getBoundingClientRect()
      const px = e.clientX - rect.left
      const py = e.clientY - rect.top
      const wx = (px - ox) / scale
      const wy = heightM - (py - oy) / scale
      if (wx < 0 || wy < 0 || wx > widthM || wy > heightM) return null
      return { wx, wy }
    },
    [currentMap, projection],
  )

  const writeCell = useCallback(
    (wx: number, wy: number, value: 0 | 1) => {
      if (!currentMap || !decoded) return
      const { col, row } = worldToCell(
        wx,
        wy,
        currentMap.height_cells,
        currentMap.cell_size_m,
      )
      if (col < 0 || col >= currentMap.width_cells) return
      if (row < 0 || row >= currentMap.height_cells) return
      const idx = row * currentMap.width_cells + col
      if (idx === lastCellRef.current) return
      lastCellRef.current = idx
      if (decoded[idx] === value) return
      const next = new Uint8Array(decoded)
      next[idx] = value
      const b64 = encodeGrid(next)
      useAppStore.getState().updateMapGridLocal(b64)

      if (saveTimerRef.current !== null) {
        window.clearTimeout(saveTimerRef.current)
      }
      const mapId = currentMap.id
      saveTimerRef.current = window.setTimeout(() => {
        const latest = useAppStore.getState().currentMap?.grid_data_b64
        if (!latest) return
        updateMap(mapId, { grid_data_b64: latest }).catch((err) => {
          toast.error(t('edit.toast.saveFail', { msg: (err as Error).message }))
        })
      }, 700)
    },
    [currentMap, decoded, t],
  )

  useEffect(() => {
    return () => {
      if (saveTimerRef.current !== null) {
        window.clearTimeout(saveTimerRef.current)
        saveTimerRef.current = null
      }
    }
  }, [currentMap?.id])

  const onMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (editMode === 'off') return
    if (e.button !== 0) return
    const w = eventToWorld(e)
    if (!w) return
    const value: 0 | 1 = editMode === 'paint' ? 1 : 0
    paintingRef.current = value
    lastCellRef.current = -1
    writeCell(w.wx, w.wy, value)
  }

  const onMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (paintingRef.current === null) return
    const w = eventToWorld(e)
    if (!w) return
    writeCell(w.wx, w.wy, paintingRef.current)
  }

  const endStroke = () => {
    paintingRef.current = null
    lastCellRef.current = -1
  }

  const onClick = async (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (editMode !== 'off') return
    if (!currentMap) return
    const w = eventToWorld(e)
    if (!w) return
    const { wx, wy } = w

    const { col, row } = worldToCell(wx, wy, currentMap.height_cells, currentMap.cell_size_m)
    if (
      col < 0 ||
      col >= currentMap.width_cells ||
      row < 0 ||
      row >= currentMap.height_cells
    ) {
      return
    }
    if (decoded && decoded[row * currentMap.width_cells + col] === 1) return

    const draft = useAppStore.getState().draftSpace
    if (draft !== null) {
      useAppStore.getState().appendDraftPoint([wx, wy])
      toast.info(`corner added (${draft.points.length + 1} pts)`)
      return
    }

    const robot = useAppStore.getState().robot
    const appendLog = useAppStore.getState().appendLog
    const setPath = useAppStore.getState().setPath
    const t0 = performance.now()
    try {
      const res = await plan(currentMap.id, robot, { x: wx, y: wy })
      const ms = performance.now() - t0
      if (!res.waypoints || res.waypoints.length === 0) {
        toast.error('path not found')
        appendLog({
          intent: 'CLICK_NAV',
          params: { x: +wx.toFixed(2), y: +wy.toFixed(2) },
          ok: false,
          msg: 'no path',
          src: '(map click)',
          latencyMs: ms,
        })
        return
      }
      setPath(res.waypoints, { x: wx, y: wy, name: '(click)' })
      appendLog({
        intent: 'CLICK_NAV',
        params: { x: +wx.toFixed(2), y: +wy.toFixed(2) },
        ok: true,
        msg: `path · ${res.waypoints.length} waypoints`,
        src: '(map click)',
        latencyMs: ms,
      })
    } catch (err) {
      toast.error((err as Error).message)
    }
  }

  if (!currentMap) return null

  const cursor =
    editMode === 'paint'
      ? 'cursor-crosshair'
      : editMode === 'erase'
        ? 'cursor-cell'
        : 'cursor-pointer'

  return (
    <div ref={containerRef} className="absolute inset-0">
      <canvas
        ref={canvasRef}
        onClick={onClick}
        onMouseDown={onMouseDown}
        onMouseMove={onMouseMove}
        onMouseUp={endStroke}
        onMouseLeave={endStroke}
        className={cursor}
      />
    </div>
  )
}

export default Scene2D
