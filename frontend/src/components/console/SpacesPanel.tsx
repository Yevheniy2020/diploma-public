import { useState } from 'react'
import { deleteSpace, patchSpace, plan } from '../../api/client'
import { useT } from '../../i18n'
import { useAppStore } from '../../state/useAppStore'
import { toast } from '../../state/useToaster'
import { Section } from './primitives'

function polygonCentroid(vertices: [number, number][]): { x: number; y: number } {
  if (vertices.length === 0) return { x: 0, y: 0 }
  let sx = 0
  let sy = 0
  for (const [vx, vy] of vertices) {
    sx += vx
    sy += vy
  }
  return { x: sx / vertices.length, y: sy / vertices.length }
}

function SpacesPanel() {
  const t = useT()
  const spaces = useAppStore((s) => s.spaces)
  const currentMap = useAppStore((s) => s.currentMap)
  const robot = useAppStore((s) => s.robot)
  const removeSpace = useAppStore((s) => s.removeSpace)
  const patchSpaceLocal = useAppStore((s) => s.patchSpaceLocal)
  const setPath = useAppStore((s) => s.setPath)
  const appendLog = useAppStore((s) => s.appendLog)

  const [busyId, setBusyId] = useState<number | null>(null)
  const [renamingId, setRenamingId] = useState<number | null>(null)
  const [renameValue, setRenameValue] = useState('')

  const onGo = async (spaceId: number) => {
    if (!currentMap) return
    const space = spaces.find((r) => r.id === spaceId)
    if (!space) return
    setBusyId(spaceId)
    const t0 = performance.now()
    try {
      const goal = polygonCentroid(space.vertices)
      const r = await plan(currentMap.id, robot, goal)
      const ms = performance.now() - t0
      if (r.waypoints.length === 0) {
        toast.error(t('labels.toast.pathNotFound', { name: space.name }))
        appendLog({
          intent: 'NAVIGATE',
          params: { target: space.name },
          ok: false,
          msg: 'no path',
          src: `(quick) ${space.name}`,
          latencyMs: ms,
        })
        return
      }
      setPath(r.waypoints, { x: goal.x, y: goal.y, name: space.name })
      toast.success(t('labels.toast.going', { name: space.name }))
      appendLog({
        intent: 'NAVIGATE',
        params: { target: space.name },
        ok: true,
        msg: `path · ${r.waypoints.length} waypoints`,
        src: `(quick) ${space.name}`,
        latencyMs: ms,
      })
    } catch (e) {
      toast.error((e as Error).message)
    } finally {
      setBusyId(null)
    }
  }

  const onDelete = async (spaceId: number) => {
    const space = spaces.find((r) => r.id === spaceId)
    if (!space) return
    setBusyId(spaceId)
    try {
      await deleteSpace(spaceId)
      removeSpace(spaceId)
      toast.success(`deleted «${space.name}»`)
      appendLog({
        intent: 'DELETE_SPACE',
        params: { name: space.name },
        ok: true,
        msg: 'manual delete',
        src: '(UI)',
      })
    } catch (e) {
      toast.error((e as Error).message)
    } finally {
      setBusyId(null)
    }
  }

  const onSetHome = async (spaceId: number) => {
    setBusyId(spaceId)
    try {
      const updated = await patchSpace(spaceId, { is_home: true })
      patchSpaceLocal(spaceId, { is_home: true })
      toast.success(`«${updated.name}» is now home`)
    } catch (e) {
      toast.error((e as Error).message)
    } finally {
      setBusyId(null)
    }
  }

  const onSubmitRename = async (spaceId: number) => {
    const newName = renameValue.trim().toLowerCase()
    if (!newName) {
      setRenamingId(null)
      return
    }
    setBusyId(spaceId)
    try {
      const updated = await patchSpace(spaceId, { name: newName })
      patchSpaceLocal(spaceId, { name: updated.name })
      toast.success(`renamed → «${updated.name}»`)
    } catch (e) {
      toast.error((e as Error).message)
    } finally {
      setBusyId(null)
      setRenamingId(null)
    }
  }

  return (
    <Section title={`spaces · ${spaces.length}`}>
      <div className="px-3 pb-3 space-y-1 max-h-[320px] overflow-y-auto scrollbar-slim">
        {spaces.length === 0 && (
          <div className="text-[12.5px] text-stone-500 italic">
            no spaces — create one by voice: «почни зону X» / «малюй мітку X»
          </div>
        )}
        {spaces.map((r) => {
          const isHome = r.is_home
          const isRenaming = renamingId === r.id
          return (
            <div
              key={r.id}
              className="border hairline px-2.5 py-1.5 hover:bg-stone-50 group flex items-start gap-2"
            >
              <span
                className={`w-1 h-1 mt-2 rounded-full shrink-0 ${
                  isHome ? 'bg-amber-500' : 'bg-stone-900'
                }`}
              />
              <div className="flex-1 min-w-0">
                <div className="flex items-baseline justify-between">
                  {isRenaming ? (
                    <input
                      autoFocus
                      value={renameValue}
                      onChange={(e) => setRenameValue(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') onSubmitRename(r.id)
                        if (e.key === 'Escape') setRenamingId(null)
                      }}
                      onBlur={() => onSubmitRename(r.id)}
                      className="text-[13px] font-medium border-b border-stone-400 bg-transparent flex-1 mr-2 outline-none"
                    />
                  ) : (
                    <button
                      type="button"
                      onClick={() => onGo(r.id)}
                      disabled={busyId === r.id || !currentMap}
                      className="text-[13px] font-medium hover:underline truncate text-left disabled:opacity-50"
                    >
                      {r.name}
                      {isHome && (
                        <span className="ml-1.5 text-[10px] text-amber-600 font-normal">
                          ★ home
                        </span>
                      )}
                    </button>
                  )}
                  <span className="font-mono text-[10px] text-stone-400 num shrink-0 ml-2">
                    id·{String(r.id).slice(-3)} · {r.vertices.length}pts
                  </span>
                </div>
              </div>
              <div className="flex flex-col gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity">
                {!isHome && (
                  <button
                    type="button"
                    onClick={() => onSetHome(r.id)}
                    disabled={busyId === r.id}
                    title="set as home"
                    className="upper-mono text-stone-500 hover:text-amber-700 self-end disabled:opacity-50"
                  >
                    ★
                  </button>
                )}
                <button
                  type="button"
                  onClick={() => {
                    setRenamingId(r.id)
                    setRenameValue(r.name)
                  }}
                  disabled={busyId === r.id}
                  title="rename"
                  className="upper-mono text-stone-500 hover:text-stone-900 self-end disabled:opacity-50"
                >
                  EDIT
                </button>
                {!isHome && (
                  <button
                    type="button"
                    onClick={() => onDelete(r.id)}
                    disabled={busyId === r.id}
                    title="delete"
                    className="upper-mono text-red-700 hover:text-red-900 self-end disabled:opacity-50"
                  >
                    ×
                  </button>
                )}
              </div>
            </div>
          )
        })}
      </div>
    </Section>
  )
}

export default SpacesPanel
