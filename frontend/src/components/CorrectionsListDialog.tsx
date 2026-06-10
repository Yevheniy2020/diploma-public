import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  deleteCorrection,
  listCorrections,
  patchCorrection,
  type CorrectionRow,
} from '../api/client'
import { useT } from '../i18n'
import { useAppStore } from '../state/useAppStore'
import { toast } from '../state/useToaster'
import { ACTION_INTENTS, type Intent } from '../types'
import Modal from './Modal'
import SlotEditor, { type SlotState } from './correction/SlotEditor'

interface Props {
  open: boolean
  onClose: () => void
}

const INPUT_CLS =
  'bg-white border hairline-strong px-3 py-2 outline-none focus:border-stone-700 text-stone-900 text-[13px] font-mono w-full'

function clamp(n: number, lo: number, hi: number): number {
  if (Number.isNaN(n)) return lo
  return Math.max(lo, Math.min(hi, n))
}

function asString(v: unknown): string {
  return typeof v === 'string' ? v : v == null ? '' : String(v)
}

function asNumber(v: unknown, fallback: number): number {
  if (typeof v === 'number' && Number.isFinite(v)) return v
  if (typeof v === 'string') {
    const n = Number(v)
    return Number.isFinite(n) ? n : fallback
  }
  return fallback
}

function paramsToSlotState(
  params: Record<string, unknown>,
  firstLabel: string,
): SlotState {
  return {
    delta_deg: clamp(asNumber(params.delta_deg, 90), -360, 360),
    direction: (params.direction as SlotState['direction']) ?? 'forward',
    distance_m: clamp(asNumber(params.distance_m, 1.0), 0.05, 10),
    target_or_name:
      asString(params.target ?? params.name) || firstLabel,
    new_name: asString(params.new_name),
    radius: clamp(asNumber(params.radius, 1.0), 0.2, 5.0),
    old_name: asString(params.old_name) || firstLabel,
  }
}

function slotsToParams(intent: Intent, slots: SlotState): Record<string, unknown> {
  const params: Record<string, unknown> = {}
  switch (intent) {
    case 'ROTATE':
      params.delta_deg = clamp(slots.delta_deg, -360, 360)
      break
    case 'DRIVE_RELATIVE':
      params.direction = slots.direction
      params.distance_m = clamp(slots.distance_m, 0.05, 10)
      break
    case 'NAVIGATE':
      if (slots.target_or_name) params.target = slots.target_or_name
      break
    case 'DELETE_SPACE':
      if (slots.target_or_name) params.name = slots.target_or_name
      break
    case 'RENAME_SPACE':
      if (slots.old_name) params.old_name = slots.old_name
      if (slots.new_name) params.new_name = slots.new_name.trim().toLowerCase()
      break
    case 'START_SPACE':
      if (slots.target_or_name)
        params.name = slots.target_or_name.trim().toLowerCase()
      break
    default:
      break
  }
  return params
}

function formatSlots(intent: string, params: Record<string, unknown>): string {
  const pick = (k: string): string | null => {
    const v = params[k]
    if (v === undefined || v === null) return null
    if (typeof v === 'number') return String(Math.round(v * 100) / 100)
    if (typeof v === 'string') return v
    return null
  }
  switch (intent) {
    case 'ROTATE': {
      const d = pick('delta_deg')
      return d ? `${d}°` : '—'
    }
    case 'DRIVE_RELATIVE': {
      const dir = pick('direction')
      const dist = pick('distance_m')
      return [dir, dist ? `${dist} m` : null].filter(Boolean).join(' · ') || '—'
    }
    case 'NAVIGATE':
      return pick('target') ?? '—'
    case 'DELETE_SPACE':
      return pick('name') ?? '—'
    case 'RENAME_SPACE': {
      const oldN = pick('old_name')
      const newN = pick('new_name')
      return oldN && newN ? `${oldN} → ${newN}` : '—'
    }
    case 'START_SPACE':
      return pick('name') ?? '—'
    case 'FINISH_SPACE':
    case 'CANCEL_SPACE':
    case 'STOP':
    case 'RETURN_HOME':
      return '—'
    default:
      return '—'
  }
}

function formatTimestamp(iso: string): string {
  try {
    const d = new Date(iso)
    if (Number.isNaN(d.getTime())) return iso
    const today = new Date()
    const isToday =
      d.getFullYear() === today.getFullYear() &&
      d.getMonth() === today.getMonth() &&
      d.getDate() === today.getDate()
    const hh = d.getHours().toString().padStart(2, '0')
    const mm = d.getMinutes().toString().padStart(2, '0')
    if (isToday) return `${hh}:${mm}`
    return `${d.toISOString().slice(5, 10)} ${hh}:${mm}`
  } catch {
    return iso
  }
}

interface RowProps {
  row: CorrectionRow
  labels: string[]
  onSaved: () => void
  onDeleted: () => void
}

function CorrectionListRow({ row, labels, onSaved, onDeleted }: RowProps) {
  const t = useT()
  const [editing, setEditing] = useState(false)
  const [confirmDelete, setConfirmDelete] = useState(false)
  const [submitting, setSubmitting] = useState(false)
  const [intent, setIntent] = useState<Intent>(row.effective_intent as Intent)
  const [slots, setSlots] = useState<SlotState>(() =>
    paramsToSlotState(row.effective_params, labels[0] ?? ''),
  )

  const onStartEdit = () => {
    setIntent(row.effective_intent as Intent)
    setSlots(paramsToSlotState(row.effective_params, labels[0] ?? ''))
    setEditing(true)
    setConfirmDelete(false)
  }

  const onSave = async () => {
    if (submitting) return
    setSubmitting(true)
    try {
      await patchCorrection(row.id, {
        corrected_intent: intent,
        corrected_params: slotsToParams(intent, slots),
      })
      toast.success(t('list.toast.updated'))
      setEditing(false)
      onSaved()
    } catch (err) {
      toast.error((err as Error).message)
    } finally {
      setSubmitting(false)
    }
  }

  const onDelete = async () => {
    if (submitting) return
    setSubmitting(true)
    try {
      await deleteCorrection(row.id)
      toast.info(t('list.toast.deleted'))
      onDeleted()
    } catch (err) {
      toast.error((err as Error).message)
      setSubmitting(false)
    }
  }

  const showsDiff =
    row.was_correct &&
    row.corrected_params != null &&
    JSON.stringify(row.predicted_params) !== JSON.stringify(row.effective_params)
  const isOverride =
    row.corrected_intent != null && row.corrected_intent !== row.predicted_intent
  const isReject = !row.was_correct && row.corrected_intent == null

  return (
    <div className="border hairline-strong bg-white px-3 py-2 flex flex-col gap-1.5">
      <div className="flex items-baseline justify-between gap-2">
        <div className="font-display text-[14px] text-stone-900 truncate" title={row.transcription}>
          «{row.transcription || '—'}»
        </div>
        <span className="font-mono text-[10.5px] text-stone-500 num shrink-0">
          {formatTimestamp(row.created_at)}
        </span>
      </div>

      {editing ? (
        <div className="flex flex-col gap-2 mt-1">
          <label className="flex items-center gap-2 text-[12px]">
            <span className="font-mono text-stone-700 w-20 shrink-0">{t('list.intent')}</span>
            <select
              value={intent}
              onChange={(e) => setIntent(e.target.value as Intent)}
              className={INPUT_CLS}
            >
              {ACTION_INTENTS.map((i) => (
                <option key={i} value={i}>
                  {i.toLowerCase()}
                </option>
              ))}
              <option value="UNKNOWN">unknown</option>
            </select>
          </label>
          <SlotEditor
            intent={intent}
            slots={slots}
            setSlots={setSlots}
            labels={labels}
          />
          <div className="flex gap-2 justify-end mt-1">
            <button
              type="button"
              onClick={() => setEditing(false)}
              disabled={submitting}
              className="upper-mono px-3 py-1 border hairline-strong bg-white hover:bg-stone-50 text-stone-700 disabled:opacity-50 text-[11px]"
            >
              {t('list.cancel')}
            </button>
            <button
              type="button"
              onClick={onSave}
              disabled={submitting}
              className="upper-mono px-3 py-1 border-2 border-stone-900 bg-stone-900 text-white hover:bg-stone-800 disabled:opacity-50 text-[11px]"
            >
              {submitting ? t('list.saving') : t('list.save')}
            </button>
          </div>
        </div>
      ) : (
        <>
          <div className="flex items-center gap-2 text-[12px] font-mono">
            <span className="upper-mono text-stone-700 text-[10.5px] shrink-0">
              {row.effective_intent.toLowerCase()}
            </span>
            <span className="text-stone-700 truncate">
              {formatSlots(row.effective_intent, row.effective_params)}
            </span>
            {isReject ? (
              <span className="upper-mono text-red-700 text-[10px] shrink-0">{t('list.chip.rejected')}</span>
            ) : isOverride ? (
              <span className="upper-mono text-amber-700 text-[10px] shrink-0">{t('list.chip.override')}</span>
            ) : showsDiff ? (
              <span className="upper-mono text-emerald-700 text-[10px] shrink-0">{t('list.chip.slotEdit')}</span>
            ) : null}
          </div>
          {(isOverride || showsDiff) && (
            <div className="text-[11px] text-stone-400 font-mono">
              {t('list.modelPrefix')}{' '}
              <span className="text-stone-500">
                {row.predicted_intent.toLowerCase()} ·{' '}
                {formatSlots(row.predicted_intent, row.predicted_params)}
              </span>
            </div>
          )}
          <div className="flex gap-2 justify-end mt-0.5">
            {confirmDelete ? (
              <>
                <span className="text-[11px] text-stone-500 italic mr-auto self-center">
                  {t('list.confirmDelete')}
                </span>
                <button
                  type="button"
                  onClick={() => setConfirmDelete(false)}
                  disabled={submitting}
                  className="upper-mono px-2 py-0.5 border hairline-strong bg-white hover:bg-stone-50 text-stone-700 text-[10px] disabled:opacity-50"
                >
                  {t('list.no')}
                </button>
                <button
                  type="button"
                  onClick={onDelete}
                  disabled={submitting}
                  className="upper-mono px-2 py-0.5 border-2 border-red-700 bg-red-700 text-white hover:bg-red-800 text-[10px] disabled:opacity-50"
                >
                  {t('list.yesDelete')}
                </button>
              </>
            ) : (
              <>
                <button
                  type="button"
                  onClick={onStartEdit}
                  className="upper-mono px-2 py-0.5 border hairline-strong bg-white hover:bg-stone-50 text-stone-700 text-[10px]"
                >
                  {t('list.edit')}
                </button>
                <button
                  type="button"
                  onClick={() => setConfirmDelete(true)}
                  className="upper-mono px-2 py-0.5 border hairline-strong bg-white hover:bg-red-50 text-red-700 text-[10px]"
                >
                  {t('list.delete')}
                </button>
              </>
            )}
          </div>
        </>
      )}
    </div>
  )
}

function CorrectionsListDialog({ open, onClose }: Props) {
  const t = useT()
  const [rows, setRows] = useState<CorrectionRow[]>([])
  const [loading, setLoading] = useState(false)
  const roomsRaw = useAppStore((s) => s.spaces)
  const labels = useMemo(() => roomsRaw.map((r) => r.name), [roomsRaw])

  const refresh = useCallback(async () => {
    setLoading(true)
    try {
      const data = await listCorrections(true)
      setRows(data)
    } catch (err) {
      toast.error((err as Error).message)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (open) refresh()
  }, [open, refresh])

  return (
    <Modal open={open} onClose={onClose} title={t('list.title')} size="lg">
      <div
        className="flex flex-col gap-2 -mx-2 px-2 max-h-[70vh] overflow-y-auto scrollbar-slim"
      >
        {loading && rows.length === 0 ? (
          <div className="text-stone-400 italic text-[12px] text-center py-4">
            {t('list.loading')}
          </div>
        ) : rows.length === 0 ? (
          <div className="text-stone-400 italic text-[12px] text-center py-4">
            {t('list.empty')}
          </div>
        ) : (
          rows.map((r) => (
            <CorrectionListRow
              key={r.id}
              row={r}
              labels={labels}
              onSaved={refresh}
              onDeleted={refresh}
            />
          ))
        )}
      </div>
      <div className="flex justify-between items-center mt-3">
        <span className="font-mono text-[11px] text-stone-500">
          {rows.length} {rows.length === 1 ? t('list.entry') : t('list.entries')} ·
          {' '}{t('list.pendingSuffix')}
        </span>
        <button
          type="button"
          onClick={onClose}
          className="upper-mono px-3 py-1.5 border hairline-strong bg-white hover:bg-stone-50 text-stone-700"
        >
          {t('list.close')}
        </button>
      </div>
    </Modal>
  )
}

export default CorrectionsListDialog
