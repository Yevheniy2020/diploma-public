// Small reusable bits for the operator console — kept in one file because
// each is ~10 lines and they all share the same style language.
import { useEffect, useState } from 'react'
import type { ReactNode } from 'react'

export function Section({
  title,
  right,
  children,
}: {
  title: ReactNode
  right?: ReactNode
  children?: ReactNode
}) {
  return (
    <div className="border-b hairline">
      <div className="px-3 pt-2.5 pb-2 flex items-center justify-between">
        <span className="upper-mono text-stone-700">{title}</span>
        {right}
      </div>
      {children}
    </div>
  )
}

export function PoseRow({ k, v, unit }: { k: string; v: string; unit?: string }) {
  return (
    <div className="flex items-baseline justify-between font-mono text-[12.5px]">
      <span className="text-stone-500 num">{k}</span>
      <span className="text-stone-900 num tabular-nums">
        {v}
        {unit ? <span className="text-stone-400 ml-1 text-[10.5px]">{unit}</span> : null}
      </span>
    </div>
  )
}

export function KV({ k, v }: { k: ReactNode; v: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between text-[12px]">
      <span className="text-stone-500 font-mono">{k}</span>
      <span className="text-stone-900 font-mono num">{v}</span>
    </div>
  )
}

export type RobotStatus = 'idle' | 'moving' | 'arrived' | 'stopped'

const STATUS_STYLE: Record<RobotStatus, { c: string; l: string }> = {
  idle: { c: 'bg-stone-200 text-stone-700', l: 'IDLE' },
  moving: { c: 'bg-emerald-700 text-white', l: 'MOVING' },
  arrived: { c: 'bg-emerald-700 text-white', l: 'ARRIVED' },
  stopped: { c: 'bg-amber-700 text-white', l: 'STOPPED' },
}

export function StatusPill({ status }: { status: RobotStatus }) {
  const m = STATUS_STYLE[status] ?? STATUS_STYLE.idle
  return <span className={`upper-mono px-1.5 py-[1px] ${m.c}`}>● {m.l}</span>
}

export function Toggle({
  label,
  on,
  onChange,
}: {
  label: string
  on: boolean
  onChange: (next: boolean) => void
}) {
  return (
    <button
      type="button"
      onClick={() => onChange(!on)}
      className="inline-flex items-center gap-2 text-[12px] py-0.5 group"
    >
      <span className="text-stone-700">{label}</span>
      <span
        className={`w-7 h-3.5 border hairline-strong relative transition-colors ${on ? 'bg-stone-900' : 'bg-white'}`}
      >
        <span
          className={`absolute top-[1px] w-[12px] h-[10px] bg-white border hairline-strong transition-all ${on ? 'left-[14px] bg-stone-200' : 'left-[1px]'}`}
        />
      </span>
    </button>
  )
}

export function MicIcon({ active }: { active?: boolean }) {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
      <rect x="9" y="3" width="6" height="11" rx="3" fill="currentColor" />
      <path
        d="M5 11a7 7 0 0 0 14 0"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
      <line
        x1="12"
        y1="18"
        x2="12"
        y2="22"
        stroke="currentColor"
        strokeWidth="2"
        strokeLinecap="round"
      />
      {active && <circle cx="20" cy="5" r="3" fill="#fff" className="blink" />}
    </svg>
  )
}

export function AudioBars({ level }: { level: number }) {
  const bars = 8
  return (
    <div className="flex items-end gap-[2px] h-[14px]">
      {Array.from({ length: bars }).map((_, i) => {
        const phase = (i / bars) * Math.PI * 2
        const h = 3 + Math.abs(Math.sin(phase + level * 6)) * (level * 11 + 2)
        return (
          <span
            key={i}
            className="w-[3px] bg-stone-900"
            style={{ height: `${h}px` }}
          />
        )
      })}
    </div>
  )
}

export function Stat({
  n,
  l,
  small,
}: {
  n: ReactNode
  l: string
  small?: string
}) {
  return (
    <div className="border hairline-strong p-2 bg-stone-50">
      <div className="font-display text-2xl font-light text-stone-900 leading-none num">
        {n}
      </div>
      <div className="upper-mono text-stone-600 mt-1">{l}</div>
      {small && (
        <div className="text-[9px] font-mono text-stone-400 mt-0.5">{small}</div>
      )}
    </div>
  )
}

export function Clock() {
  const [t, setT] = useState(() => new Date())
  useEffect(() => {
    const i = setInterval(() => setT(new Date()), 60_000)
    return () => clearInterval(i)
  }, [])
  return (
    <span className="font-mono num text-stone-900">
      {t.toISOString().slice(0, 10)}
    </span>
  )
}

export type PipelineState = 'idle' | 'recording' | 'uploading' | 'parsing' | 'done'

const PIPELINE_ORDER: PipelineState[] = ['idle', 'recording', 'uploading', 'parsing', 'done']

const PIPELINE_STEPS: { k: Exclude<PipelineState, 'idle'>; l: string; s: string }[] = [
  { k: 'recording', l: 'capture', s: 'MediaRecorder · opus webm' },
  { k: 'uploading', l: 'upload', s: 'POST /api/voice · multipart' },
  { k: 'parsing', l: 'parse', s: 'Llama-4 · audio-in' },
  { k: 'done', l: 'dispatch', s: 'intent → action' },
]

export function PipelineSteps({ state }: { state: PipelineState }) {
  const idx = PIPELINE_ORDER.indexOf(state)
  return (
    <ol className="space-y-1.5">
      {PIPELINE_STEPS.map((st, i) => {
        const active = state === st.k
        const past = idx > PIPELINE_ORDER.indexOf(st.k)
        return (
          <li
            key={st.k}
            className={`flex items-center gap-2 text-[12px] ${active ? 'text-stone-900' : past ? 'text-stone-700' : 'text-stone-400'}`}
          >
            <span
              className={`w-2 h-2 shrink-0 ${active ? 'bg-red-600' : past ? 'bg-stone-900' : 'bg-stone-300'} ${active ? 'pulse-rec' : ''}`}
            />
            <span className="font-mono w-[60px] uppercase tracking-wider text-[10px]">
              {i + 1} · {st.l}
            </span>
            <span className="font-mono text-[10.5px] flex-1 text-stone-500 truncate">
              {st.s}
            </span>
          </li>
        )
      })}
    </ol>
  )
}

export function LogEntry({
  e,
}: {
  e: {
    ts: string
    intent: string
    params: Record<string, unknown>
    ok: boolean
    msg: string
    src: string
    latencyMs?: number
  }
}) {
  const intentColor = e.ok ? 'text-emerald-800' : 'text-red-700'
  return (
    <div className="px-3 py-2 border-b hairline last:border-b-0 hover:bg-stone-50">
      <div className="flex items-center justify-between font-mono text-[10.5px]">
        <span className="text-stone-500 num">{e.ts}</span>
        <span className={`upper-mono ${intentColor}`}>
          {e.ok ? '✓' : '✗'} {e.intent}
        </span>
      </div>
      <div className="font-mono text-[11px] text-stone-700 mt-0.5 truncate">
        {Object.keys(e.params || {}).length ? (
          <code>{JSON.stringify(e.params)}</code>
        ) : (
          <span className="text-stone-400">— no params —</span>
        )}
      </div>
      <div className="flex items-center justify-between mt-0.5">
        <span className="text-[10.5px] italic text-stone-500 truncate flex-1">
          « {e.src} »
        </span>
        {e.latencyMs != null && (
          <span className="font-mono text-[10px] text-stone-500 num shrink-0 ml-2">
            {Math.round(e.latencyMs)} ms
          </span>
        )}
      </div>
      <div className="text-[10.5px] text-stone-600 mt-0.5">{e.msg}</div>
    </div>
  )
}
