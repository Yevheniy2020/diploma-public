// Right-sidebar panel that surfaces the active-learning retrain cycle
// to the operator. Polls /api/voice/feedback/stats every 5 s; while a
// retrain is running, polls /retrain/status every 2 s and renders a
// ticking timer + phase string. Triggers retraining via POST. The model
// hot-reloads in the backend on completion, so no page refresh needed.
import { useCallback, useEffect, useRef, useState } from 'react'
import {
  getFeedbackStats,
  getRetrainStatus,
  triggerRetrain,
  type FeedbackStats,
  type RetrainStatus,
} from '../../api/client'
import { t as tRaw, useT, type I18nKey } from '../../i18n'
import { toast } from '../../state/useToaster'
import CorrectionsListDialog from '../CorrectionsListDialog'
import { KV, Section } from './primitives'

const STATS_POLL_MS = 5_000
const STATUS_POLL_MS = 2_000

interface PhaseDetail {
  label: string
  hint: string
  etaText?: string
}

const PHASE_KEYS: Record<string, { label: I18nKey; hint?: I18nKey; eta?: I18nKey }> = {
  preparing: {
    label: 'training.phase.preparing.label',
    hint: 'training.phase.preparing.hint',
    eta: 'training.phase.preparing.eta',
  },
  paraphrasing: {
    label: 'training.phase.paraphrasing.label',
    hint: 'training.phase.paraphrasing.hint',
    eta: 'training.phase.paraphrasing.eta',
  },
  training: {
    label: 'training.phase.training.label',
    hint: 'training.phase.training.hint',
    eta: 'training.phase.training.eta',
  },
  done: { label: 'training.phase.done.label', hint: 'training.phase.done.hint' },
  failed: { label: 'training.phase.failed.label' },
}

function phaseDetail(
  phase: string | null,
  tr: (k: I18nKey, p?: Record<string, string | number>) => string,
): PhaseDetail {
  const keys = phase ? PHASE_KEYS[phase] : undefined
  if (!keys) return { label: phase ?? '...', hint: '' }
  return {
    label: tr(keys.label),
    hint: keys.hint ? tr(keys.hint) : '',
    etaText: keys.eta ? tr(keys.eta) : undefined,
  }
}

function formatElapsed(s: number): string {
  const mm = Math.floor(s / 60)
    .toString()
    .padStart(2, '0')
  const ss = Math.floor(s % 60)
    .toString()
    .padStart(2, '0')
  return `${mm}:${ss}`
}

function formatLastRetrained(
  iso: string | null | undefined,
  tr: (k: I18nKey, p?: Record<string, string | number>) => string,
): string {
  if (!iso) return '—'
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
    if (isToday) return tr('training.lastRetrained.today', { hhmm: `${hh}:${mm}` })
    const day = d.toISOString().slice(0, 10)
    return `${day} ${hh}:${mm}`
  } catch {
    return iso
  }
}

function TrainingPanel() {
  const tr = useT()
  const [stats, setStats] = useState<FeedbackStats | null>(null)
  const [status, setStatus] = useState<RetrainStatus | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [listOpen, setListOpen] = useState(false)
  // Local 1-Hz ticker so the elapsed seconds increment smoothly between
  // status polls (which run every 2 s).
  const [tick, setTick] = useState(0)
  const startedRef = useRef<number | null>(null)

  const refreshStats = useCallback(async () => {
    try {
      setStats(await getFeedbackStats())
    } catch (err) {
      console.error('feedback stats failed', err)
    }
  }, [])

  const refreshStatus = useCallback(async () => {
    try {
      const s = await getRetrainStatus()
      setStatus(s)
      if (s.state === 'running' && s.started_at) {
        startedRef.current = new Date(s.started_at).getTime()
      } else {
        startedRef.current = null
      }
      return s
    } catch (err) {
      console.error('retrain status failed', err)
      return null
    }
  }, [])

  // Initial load.
  useEffect(() => {
    refreshStats()
    refreshStatus()
  }, [refreshStats, refreshStatus])

  // Background poll: stats every 5 s; status more often when running.
  useEffect(() => {
    const statsId = setInterval(refreshStats, STATS_POLL_MS)
    return () => clearInterval(statsId)
  }, [refreshStats])

  useEffect(() => {
    const isRunning = status?.state === 'running'
    if (!isRunning) return
    const id = setInterval(async () => {
      const fresh = await refreshStatus()
      if (fresh && fresh.state !== 'running') {
        // Transition out of running — refresh stats and let the user know.
        refreshStats()
        if (fresh.state === 'completed') {
          toast.success(tRaw('training.toast.completed'))
        } else if (fresh.state === 'failed') {
          toast.error(
            tRaw('training.toast.failed', {
              err: fresh.error ? fresh.error.split('\n')[0] : '',
            }),
          )
        }
      }
    }, STATUS_POLL_MS)
    return () => clearInterval(id)
  }, [status?.state, refreshStatus, refreshStats])

  // 1-Hz ticker so the elapsed counter visibly moves while running.
  useEffect(() => {
    if (status?.state !== 'running') return
    const id = setInterval(() => setTick((t) => t + 1), 1000)
    return () => clearInterval(id)
  }, [status?.state])

  // Phase-transition toasts: every time the running phase changes
  // (e.g. preparing → paraphrasing), drop a small info toast so the
  // operator gets visible feedback without having to watch the panel.
  const prevPhaseRef = useRef<string | null>(null)
  useEffect(() => {
    if (status?.state === 'running' && status.phase) {
      if (prevPhaseRef.current && status.phase !== prevPhaseRef.current) {
        const det = phaseDetail(status.phase, tRaw)
        if (det.label) toast.info(`→ ${det.label}`)
      }
      prevPhaseRef.current = status.phase
    } else if (status?.state !== 'running') {
      prevPhaseRef.current = null
    }
  }, [status?.state, status?.phase])

  const onClick = async () => {
    if (submitting) return
    setSubmitting(true)
    try {
      const s = await triggerRetrain()
      setStatus(s)
      const eta =
        (stats?.pending_count ?? 0) > 50 ? tRaw('training.eta.many') : tRaw('training.eta.few')
      toast.info(tRaw('training.toast.started', { eta }))
    } catch (err) {
      toast.error((err as Error).message)
    } finally {
      setSubmitting(false)
    }
  }

  const isRunning = status?.state === 'running'
  const isFailed = status?.state === 'failed'
  const pending = stats?.pending_count ?? 0

  // Live elapsed: prefer locally-ticked value when running, otherwise
  // show whatever the server reported (mostly 0).
  const liveElapsed =
    isRunning && startedRef.current
      ? (Date.now() - startedRef.current) / 1000 + tick * 0
      : status?.elapsed_seconds ?? 0
  void tick // keep the dep so elapsed re-renders every second

  return (
    <Section title={tr('training.title')}>
      <div className="px-3 pb-3 space-y-2.5">
        <KV
          k={tr('training.pending')}
          v={
            pending > 0 ? (
              <button
                type="button"
                onClick={() => setListOpen(true)}
                className="text-emerald-700 underline decoration-dotted underline-offset-2 hover:text-emerald-800"
                title={tr('training.title.review')}
              >
                {pending}
              </button>
            ) : (
              <button
                type="button"
                onClick={() => setListOpen(true)}
                className="text-stone-400 hover:text-stone-600"
                title={tr('training.title.history')}
              >
                0
              </button>
            )
          }
        />
        <KV k={tr('training.lastRetrained')} v={formatLastRetrained(stats?.last_retrained_at, tr)} />

        {isRunning ? (
          <RunningBlock phase={status?.phase ?? null} elapsed={liveElapsed} />
        ) : null}

        {isFailed && status?.error ? (
          <div className="border border-red-700 bg-red-50 px-3 py-2 font-mono text-[11px] text-red-700">
            <div className="upper-mono mb-1">{tr('training.crashed')}</div>
            <pre className="whitespace-pre-wrap break-words">{status.error}</pre>
          </div>
        ) : null}

        <button
          type="button"
          onClick={onClick}
          disabled={submitting || isRunning || pending === 0}
          className="upper-mono px-3 py-1.5 border-2 border-stone-900 bg-stone-900 text-white hover:bg-stone-800 disabled:opacity-40 w-full"
        >
          {isRunning ? tr('training.button.running') : tr('training.button.retrain')}
        </button>

        {pending === 0 && !isRunning ? (
          <div className="text-stone-400 italic text-[11px] text-center">
            {tr('training.noNew')}
          </div>
        ) : null}
      </div>
      <CorrectionsListDialog
        open={listOpen}
        onClose={() => {
          setListOpen(false)
          // Refresh in case user edited / deleted while modal was open.
          refreshStats()
        }}
      />
    </Section>
  )
}

function RunningBlock({
  phase,
  elapsed,
}: {
  phase: string | null
  elapsed: number
}) {
  const tr = useT()
  const det = phaseDetail(phase, tr)
  return (
    <div className="border hairline-strong bg-stone-50 px-3 py-2.5">
      <div className="flex items-center gap-2">
        <span
          className="font-mono text-[11px] blink text-stone-500 inline-block min-w-[20px]"
          aria-hidden
        >
          ●●●
        </span>
        <span className="upper-mono text-stone-900 text-[11px]">{det.label}</span>
        <span className="num text-stone-900 text-[13px] ml-auto">
          {formatElapsed(elapsed)}
        </span>
      </div>
      {det.hint ? (
        <div className="text-stone-600 mt-1.5 text-[11px] font-mono leading-snug">
          {det.hint}
          {det.etaText ? (
            <span className="text-stone-400"> · {det.etaText}</span>
          ) : null}
        </div>
      ) : null}
      <div className="text-stone-400 italic mt-1.5 text-[10.5px]">
        {tr('training.bgHint')}
      </div>
    </div>
  )
}

export default TrainingPanel
