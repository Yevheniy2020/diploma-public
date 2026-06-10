import { useCallback, useEffect, useRef, useState } from 'react'
import { useAppStore } from '../state/useAppStore'
import { toast } from '../state/useToaster'
import MicPermissionDialog from './MicPermissionDialog'
import { MicIcon } from './console/primitives'

const PREFERRED_MIMES = [
  'audio/webm;codecs=opus',
  'audio/webm',
  'audio/ogg;codecs=opus',
]

const MAX_RECORD_MS = 15_000
const MIN_RECORD_MS = 300

function pickMime(): string | null {
  if (typeof MediaRecorder === 'undefined') return null
  for (const m of PREFERRED_MIMES) {
    if (MediaRecorder.isTypeSupported(m)) return m
  }
  return null
}

interface Props {
  onAudio?: (blob: Blob, mime: string) => void | Promise<void>
  onLevel?: (level: number) => void
  processing?: boolean
}

function VoiceButton({ onAudio, onLevel, processing }: Props) {
  const currentMap = useAppStore((s) => s.currentMap)
  const setRecording = useAppStore((s) => s.setRecording)
  const isRecording = useAppStore((s) => s.isRecording)
  const setVoicePipelineState = useAppStore((s) => s.setVoicePipelineState)

  const [recState, setRecState] = useState<'idle' | 'recording'>('idle')
  const [supported] = useState(() => pickMime() !== null)
  const [permDialog, setPermDialog] = useState(false)

  const recorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const chunksRef = useRef<BlobPart[]>([])
  const stopTimerRef = useRef<number | null>(null)
  const recordStartRef = useRef<number>(0)
  const audioCtxRef = useRef<AudioContext | null>(null)
  const analyserRafRef = useRef<number | null>(null)
  const onAudioRef = useRef(onAudio)
  const onLevelRef = useRef(onLevel)
  onAudioRef.current = onAudio
  onLevelRef.current = onLevel

  const visual: 'idle' | 'recording' | 'processing' = processing
    ? 'processing'
    : recState

  const stopAnalyser = useCallback(() => {
    if (analyserRafRef.current !== null) {
      cancelAnimationFrame(analyserRafRef.current)
      analyserRafRef.current = null
    }
    audioCtxRef.current?.close().catch(() => {})
    audioCtxRef.current = null
    onLevelRef.current?.(0)
  }, [])

  const stopStream = useCallback(() => {
    if (stopTimerRef.current !== null) {
      window.clearTimeout(stopTimerRef.current)
      stopTimerRef.current = null
    }
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    recorderRef.current = null
    stopAnalyser()
  }, [stopAnalyser])

  const stopRecording = useCallback(() => {
    const rec = recorderRef.current
    if (!rec) return
    if (rec.state !== 'inactive') {
      try {
        rec.stop()
      } catch {
      }
    }
    if (stopTimerRef.current !== null) {
      window.clearTimeout(stopTimerRef.current)
      stopTimerRef.current = null
    }
  }, [])

  const startRecording = useCallback(async () => {
    if (!supported) {
      toast.error('recording not supported in this browser')
      return
    }
    if (!currentMap) return
    if (recState !== 'idle' || isRecording || processing) return

    const mime = pickMime()
    if (!mime) {
      toast.error('recording not supported in this browser')
      return
    }

    try {
      const perms = navigator.permissions as Permissions | undefined
      if (perms && typeof perms.query === 'function') {
        const status = await perms.query({ name: 'microphone' as PermissionName })
        if (status.state === 'denied') {
          setPermDialog(true)
          return
        }
      }
    } catch {
    }

    let stream: MediaStream
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    } catch (err) {
      const e = err as DOMException
      toast.error(`microphone: ${e.name || 'access denied'}`)
      if (e.name === 'NotAllowedError' || e.name === 'SecurityError') {
        setPermDialog(true)
      }
      return
    }
    streamRef.current = stream

    let rec: MediaRecorder
    try {
      rec = new MediaRecorder(stream, { mimeType: mime, audioBitsPerSecond: 128000 })
    } catch (err) {
      stopStream()
      toast.error(`recorder: ${(err as Error).message}`)
      return
    }
    recorderRef.current = rec
    chunksRef.current = []

    try {
      const Ctx =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
      const ctx = new Ctx()
      audioCtxRef.current = ctx
      const src = ctx.createMediaStreamSource(stream)
      const analyser = ctx.createAnalyser()
      analyser.fftSize = 512
      src.connect(analyser)
      const buf = new Uint8Array(analyser.fftSize)
      const tick = () => {
        analyser.getByteTimeDomainData(buf)
        let sum = 0
        for (let i = 0; i < buf.length; i++) {
          const v = (buf[i] - 128) / 128
          sum += v * v
        }
        const rms = Math.sqrt(sum / buf.length)
        onLevelRef.current?.(Math.min(1, rms * 4))
        analyserRafRef.current = requestAnimationFrame(tick)
      }
      tick()
    } catch {
    }

    rec.ondataavailable = (e) => {
      if (e.data && e.data.size > 0) chunksRef.current.push(e.data)
    }
    rec.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: mime })
      const elapsed = Date.now() - recordStartRef.current
      chunksRef.current = []
      stopStream()
      setRecording(false)
      setRecState('idle')
      if (elapsed < MIN_RECORD_MS || blob.size < 1500) {
        toast.error('hold the button a bit longer')
        setVoicePipelineState('idle')
        return
      }
      void onAudioRef.current?.(blob, mime)
    }
    rec.onerror = (ev) => {
      const e = (ev as { error?: DOMException }).error
      toast.error(`recorder error: ${e?.name ?? 'unknown'}`)
      stopStream()
      setRecording(false)
      setRecState('idle')
      setVoicePipelineState('idle')
    }

    rec.start()
    recordStartRef.current = Date.now()
    setRecording(true)
    setRecState('recording')
    setVoicePipelineState('recording')
    stopTimerRef.current = window.setTimeout(() => {
      stopRecording()
    }, MAX_RECORD_MS)
  }, [
    supported,
    currentMap,
    recState,
    isRecording,
    processing,
    stopStream,
    stopRecording,
    setRecording,
    setVoicePipelineState,
  ])

  useEffect(() => {
    const isTypingTarget = (t: EventTarget | null) => {
      if (!(t instanceof HTMLElement)) return false
      const tag = t.tagName
      return tag === 'INPUT' || tag === 'TEXTAREA' || t.isContentEditable
    }
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.code !== 'Space' || e.repeat) return
      if (isTypingTarget(e.target)) return
      e.preventDefault()
      void startRecording()
    }
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.code !== 'Space') return
      if (isTypingTarget(e.target)) return
      e.preventDefault()
      stopRecording()
    }
    window.addEventListener('keydown', onKeyDown)
    window.addEventListener('keyup', onKeyUp)
    return () => {
      window.removeEventListener('keydown', onKeyDown)
      window.removeEventListener('keyup', onKeyUp)
    }
  }, [startRecording, stopRecording])

  useEffect(() => () => stopStream(), [stopStream])

  const disabled = !currentMap || !supported || visual === 'processing'

  const colorClass =
    visual === 'recording'
      ? 'bg-red-600 border-red-700 pulse-rec'
      : visual === 'processing'
        ? 'bg-amber-600 border-amber-700'
        : 'bg-stone-900 border-stone-900 hover:bg-stone-800'

  return (
    <>
      <button
        type="button"
        onPointerDown={(e) => {
          e.preventDefault()
          void startRecording()
        }}
        onPointerUp={() => stopRecording()}
        onPointerLeave={() => stopRecording()}
        onPointerCancel={() => stopRecording()}
        disabled={disabled}
        title={
          supported
            ? 'hold to talk (or hold space)'
            : 'mic not supported in this browser'
        }
        className={`relative shrink-0 w-16 h-16 rounded-full border-2 ${colorClass} text-white transition-colors flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed`}
      >
        <MicIcon active={visual === 'recording'} />
        {visual === 'idle' && supported && (
          <span className="absolute -bottom-1 -right-1 bg-red-600 text-white text-[8px] font-mono uppercase px-1 py-[1px] rounded-sm">
            PTT
          </span>
        )}
      </button>
      <MicPermissionDialog open={permDialog} onClose={() => setPermDialog(false)} />
    </>
  )
}

export default VoiceButton
