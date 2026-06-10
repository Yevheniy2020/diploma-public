import { useEffect, useMemo, useState } from 'react'
import { sendVoiceFeedback } from '../api/client'
import { useT } from '../i18n'
import { dispatchVoiceResponse } from '../sim/voiceDispatch'
import { useAppStore } from '../state/useAppStore'
import { toast } from '../state/useToaster'
import { ACTION_INTENTS, type Intent } from '../types'
import Modal from './Modal'
import SlotEditor, { type SlotState } from './correction/SlotEditor'

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

function CorrectionDialog() {
  const t = useT()
  const pending = useAppStore((s) => s.pendingCorrection)
  const robot = useAppStore((s) => s.robot)
  const spaces = useAppStore((s) => s.spaces)
  const setPending = useAppStore((s) => s.setPendingCorrection)
  const appendLog = useAppStore((s) => s.appendLog)

  const [intent, setIntent] = useState<Intent>('UNKNOWN')
  const [slots, setSlots] = useState<SlotState>({
    delta_deg: 90,
    direction: 'forward',
    distance_m: 1.0,
    target_or_name: '',
    new_name: '',
    radius: 1.0,
    old_name: '',
  })
  const [submitting, setSubmitting] = useState(false)

  useEffect(() => {
    if (!pending) return
    const p = pending.predicted_params
    const firstSpace = spaces[0]?.name ?? ''
    setIntent(pending.predicted_intent)
    setSlots({
      delta_deg: clamp(asNumber(p.delta_deg, 90), -360, 360),
      direction:
        (p.direction as SlotState['direction']) ?? 'forward',
      distance_m: clamp(asNumber(p.distance_m, 1.0), 0.05, 10),
      target_or_name:
        asString(p.target ?? p.name) ||
        firstSpace,
      new_name: asString(p.new_name),
      radius: clamp(asNumber(p.radius, 1.0), 0.2, 5.0),
      old_name: asString(p.old_name) || firstSpace,
    })
    setSubmitting(false)
  }, [pending, spaces])

  const close = () => {
    setPending(null)
    setSubmitting(false)
  }

  const composed = useMemo(() => {
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
    return { intent, params }
  }, [intent, slots])

  if (!pending) return null

  const confidencePct = Math.round(pending.confidence * 100)

  const onConfirm = async () => {
    if (submitting) return
    setSubmitting(true)
    try {
      const intentChanged = composed.intent !== pending.predicted_intent
      const fb = await sendVoiceFeedback(
        pending.command_log_id,
        true,
        robot,
        intentChanged ? composed.intent : undefined,
        composed.params,
      )
      if (fb.intent && fb.action_result) {
        dispatchVoiceResponse({
          intent: fb.intent,
          params: fb.params ?? {},
          action_result: fb.action_result,
          follow_ups: [],
        })
      }
      appendLog({
        intent: composed.intent,
        params: composed.params,
        ok: true,
        msg: `confirmed (conf ${confidencePct}%)${intentChanged ? ' · intent edited' : ''}`,
        src: '(feedback)',
      })
      close()
    } catch (err) {
      toast.error((err as Error).message)
      setSubmitting(false)
    }
  }

  const onReject = async () => {
    if (submitting) return
    setSubmitting(true)
    try {
      await sendVoiceFeedback(pending.command_log_id, false, robot)
      toast.info(t('correction.toast.saved'))
      appendLog({
        intent: pending.predicted_intent,
        params: pending.predicted_params,
        ok: false,
        msg: `rejected (conf ${confidencePct}%)`,
        src: '(feedback)',
      })
      close()
    } catch (err) {
      toast.error((err as Error).message)
      setSubmitting(false)
    }
  }

  return (
    <Modal open={true} onClose={close} title={t('correction.title')}>
      <div className="flex flex-col gap-3">
        <div className="flex flex-col gap-1 text-[12px]">
          <span className="upper-mono text-stone-500">{t('correction.transcription')}</span>
          <div className="bg-stone-50 border hairline-strong px-3 py-2 font-mono text-[13px] text-stone-900">
            «{pending.transcription || '—'}»
          </div>
        </div>

        <div className="grid grid-cols-[1fr_auto] gap-2 items-end">
          <label className="flex flex-col gap-1 text-[12px]">
            <span className="upper-mono text-stone-500">{t('correction.intent')}</span>
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
          <div className="flex flex-col items-end pb-1">
            <span className="upper-mono text-stone-500 text-[10px]">{t('correction.confidence')}</span>
            <span
              className={`font-mono text-[14px] num ${
                confidencePct >= 75
                  ? 'text-emerald-700'
                  : confidencePct >= 60
                    ? 'text-amber-700'
                    : 'text-red-700'
              }`}
            >
              {confidencePct}%
            </span>
          </div>
        </div>

        <SlotEditor
          intent={intent}
          slots={slots}
          setSlots={setSlots}
          labels={spaces.map((r) => r.name)}
        />

        <div className="flex gap-2 justify-end mt-1">
          <button
            type="button"
            onClick={onReject}
            disabled={submitting}
            className="upper-mono px-3 py-1.5 border hairline-strong bg-white hover:bg-stone-50 text-stone-700 disabled:opacity-50"
          >
            {t('correction.cancel')}
          </button>
          <button
            type="button"
            onClick={onConfirm}
            disabled={submitting}
            className="upper-mono px-3 py-1.5 border-2 border-stone-900 bg-stone-900 text-white hover:bg-stone-800 disabled:opacity-50"
          >
            {submitting ? t('correction.saving') : t('correction.confirm')}
          </button>
        </div>
      </div>
    </Modal>
  )
}


export default CorrectionDialog
