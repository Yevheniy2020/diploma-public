// Maps a VoiceResponse from the backend to store mutations and toast feedback.
// Keep this pure (no React hooks) so VoiceButton can call it from a
// non-component context. Action-result field names mirror
// backend/routers/voice.py exactly — see the per-intent helpers there.
import { createSpace } from '../api/client'
import { useAppStore } from '../state/useAppStore'
import { toast } from '../state/useToaster'
import type { Intent, Point2D, VoiceResponse } from '../types'
import { simplify } from '../utils/rdp'

function isPoint2DArray(v: unknown): v is Point2D[] {
  if (!Array.isArray(v)) return false
  return v.every(
    (p) =>
      typeof p === 'object' &&
      p !== null &&
      typeof (p as Point2D).x === 'number' &&
      typeof (p as Point2D).y === 'number',
  )
}

function asString(v: unknown): string | undefined {
  return typeof v === 'string' ? v : undefined
}

function asNumber(v: unknown): number | undefined {
  return typeof v === 'number' ? v : undefined
}

function dispatchNavigate(resp: VoiceResponse, returnHome: boolean) {
  const ar = resp.action_result ?? {}
  const target = asString(ar.target)
  const waypoints = ar.waypoints
  const reason = asString(ar.reason)

  if (isPoint2DArray(waypoints) && waypoints.length > 0) {
    useAppStore.getState().setPath(waypoints)
    toast.success(returnHome ? 'returning home' : `going to ${target ?? 'target'}`)
    return
  }

  // Backend sets reason='no_path' when the planner returned []; missing
  // waypoints array entirely means _find_space returned None.
  if (reason === 'no_path') {
    toast.error(`path not found${target ? `: ${target}` : ''}`)
  } else {
    toast.error(`space not found${target ? `: ${target}` : ''}`)
  }
}

export function dispatchVoiceResponse(resp: VoiceResponse): void {
  const ar = resp.action_result ?? {}
  const store = useAppStore.getState()

  // Mid-confidence: backend wants the operator to confirm. Park the
  // prediction in store; CorrectionDialog renders against this state.
  if (resp.intent === 'UNCERTAIN') {
    if (typeof resp.command_log_id !== 'number') {
      toast.error('uncertain prediction without command_log_id')
      return
    }
    const params = resp.params ?? {}
    const predicted = (params._predicted_intent as Intent) ?? 'UNKNOWN'
    const predictedParams =
      (params._predicted_params as Record<string, unknown>) ?? {}
    const confidence = asNumber(params._confidence) ?? 0
    const transcription = asString(params.original_text) ?? ''
    store.setPendingCorrection({
      command_log_id: resp.command_log_id,
      transcription,
      predicted_intent: predicted,
      predicted_params: predictedParams,
      confidence,
    })
    return
  }

  switch (resp.intent) {
    case 'NAVIGATE':
      dispatchNavigate(resp, false)
      return

    case 'RETURN_HOME':
      dispatchNavigate(resp, true)
      return

    case 'DRIVE_RELATIVE': {
      const waypoints = ar.waypoints
      const direction = asString(ar.direction)
      const distance = asNumber(ar.distance_m)
      const cells = asNumber(ar.distance_cells)
      const reason = asString(ar.reason)

      if (isPoint2DArray(waypoints) && waypoints.length > 0) {
        useAppStore.getState().setPath(waypoints)
        const dirUk: Record<string, string> = {
          forward: 'вперед',
          backward: 'назад',
          left: 'ліворуч',
          right: 'праворуч',
        }
        const dirLabel = direction ? (dirUk[direction] ?? direction) : 'driving'
        const distLabel = distance != null ? ` ${distance.toFixed(2)} м` : ''
        const cellsLabel =
          cells != null && cells > 0
            ? ` · ${cells} ${cells === 1 ? 'клітинка' : cells < 5 ? 'клітинки' : 'клітинок'}`
            : ''
        toast.success(`${dirLabel}${distLabel}${cellsLabel}`)
        return
      }
      toast.error(reason === 'no_path' ? 'path not found' : 'drive failed')
      return
    }

    case 'DELETE_SPACE': {
      // Backend returns { deleted: <name> }. Resolve to local id by name lookup.
      const deleted = asString(ar.deleted)
      if (deleted) {
        const row = store.spaces.find((r) => r.name === deleted)
        if (row) store.removeSpace(row.id)
        toast.success(`deleted: ${deleted}`)
      } else {
        const name = (resp.params?.name as string | undefined) ?? 'space'
        toast.error(`delete failed: ${name}`)
      }
      return
    }

    case 'RENAME_SPACE': {
      // Backend returns { old_name, new_name, space_id }.
      const spaceId = asNumber(ar.space_id)
      const newName = asString(ar.new_name)
      if (spaceId !== undefined && newName) {
        store.patchSpaceLocal(spaceId, { name: newName })
        toast.success(`renamed to ${newName}`)
      } else {
        toast.error('rename failed')
      }
      return
    }

    case 'STOP':
      store.stopMovement()
      toast.info('stopped')
      return

    case 'START_SPACE': {
      // Backend returns the space name in action_result.space_name; the
      // params.name is also valid (regex preempt sets it). Either source
      // works, action_result wins because that's what the dispatcher
      // already serialised.
      const name =
        asString(ar.space_name) ??
        asString(resp.params?.name) ??
        ''
      if (!name) {
        toast.error('start_space: missing name')
        return
      }
      store.startSpaceDraft(name)
      toast.success(`recording space «${name}» — click corners or drive`)
      return
    }

    case 'FINISH_SPACE': {
      const draftBefore = store.draftSpace
      if (draftBefore === null) {
        toast.error('no space is being recorded')
        return
      }
      const raw = store.finishSpaceDraft()
      if (raw === null) return
      const simplified = simplify(raw, 0.10)
      if (simplified.length < 3) {
        toast.error(`need ≥3 points to close (have ${simplified.length})`)
        return
      }
      const currentMapId = store.currentMap?.id
      if (currentMapId === undefined) {
        toast.error('no map loaded')
        return
      }
      createSpace({
        map_id: currentMapId,
        name: draftBefore.name,
        vertices: simplified,
      })
        .then((space) => {
          useAppStore.getState().addSpace(space)
          useAppStore.getState().cancelSpaceDraft()
          toast.success(`saved space «${space.name}» (${simplified.length} pts)`)
        })
        .catch((err: unknown) => {
          const msg = err instanceof Error ? err.message : String(err)
          toast.error(`save failed: ${msg}`)
        })
      return
    }

    case 'CANCEL_SPACE': {
      if (store.draftSpace === null) {
        toast.info('no space to cancel')
        return
      }
      const name = store.draftSpace.name
      store.cancelSpaceDraft()
      toast.info(`canceled space «${name}»`)
      return
    }

    case 'ROTATE': {
      const target = asNumber(ar.target_theta)
      const deltaDeg = asNumber(ar.delta_deg)
      if (target === undefined) {
        toast.error('rotate failed: missing target_theta')
        return
      }
      useAppStore.getState().rotateTo(target)
      const sign = (deltaDeg ?? 0) >= 0 ? '+' : ''
      toast.success(`rotating ${sign}${deltaDeg?.toFixed(0) ?? '?'}°`)
      return
    }

    case 'UNKNOWN': {
      const original = asString(ar.original_text) ?? asString(resp.params?.original_text)
      const err = asString(ar.error) ?? asString(resp.params?.error)
      const detail = original ? `"${original}"` : err ?? 'no transcription'
      toast.error(`didn't understand: ${detail}`)
      return
    }

    default:
      toast.error(`unhandled intent: ${resp.intent}`)
  }
}
