import { useCallback, useEffect, useState } from 'react'
import { getRobotStatus, sendVoice } from './api/client'
import type { RobotStatus } from './api/client'
import { setLocale, useLocale, useT } from './i18n'
import CorrectionDialog from './components/CorrectionDialog'
import DebugControls from './components/DebugControls'
import ErrorBoundary from './components/ErrorBoundary'
import Toaster from './components/Toaster'
import GoalPanel from './components/console/GoalPanel'
import MapImportPanel from './components/console/MapImportPanel'
import SpacesPanel from './components/console/SpacesPanel'
import TrainingPanel from './components/console/TrainingPanel'
import VoiceBar from './components/console/VoiceBar'
import { Clock, Toggle } from './components/console/primitives'
import Scene2D from './scenes/Scene2D'
import Scene3D from './scenes/Scene3D'
import KinematicsRunner from './sim/KinematicsRunner'
import { dispatchVoiceResponse } from './sim/voiceDispatch'
import { useAppStore } from './state/useAppStore'
import { toast } from './state/useToaster'

function isVoiceOk(intent: string, ar: Record<string, unknown>): boolean {
  if (intent === 'UNKNOWN') return false
  if (typeof ar.error === 'string' && ar.error) return false
  if (ar.reason === 'no_path') return false
  return true
}

function App() {
  const t = useT()
  const currentMap = useAppStore((s) => s.currentMap)
  const mode = useAppStore((s) => s.mode)
  const switchMode = useAppStore((s) => s.switchMode)
  const refreshMaps = useAppStore((s) => s.refreshMaps)
  const loadMap = useAppStore((s) => s.loadMap)
  const showInflation = useAppStore((s) => s.showInflation)
  const showGrid2D = useAppStore((s) => s.showGrid2D)
  const setShowInflation = useAppStore((s) => s.setShowInflation)
  const setShowGrid2D = useAppStore((s) => s.setShowGrid2D)
  const editMode = useAppStore((s) => s.editMode)
  const setEditMode = useAppStore((s) => s.setEditMode)

  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [voiceProcessing, setVoiceProcessing] = useState(false)

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const ms = await refreshMaps()
        if (cancelled) return
        if (ms.length === 0) return
        const param = new URLSearchParams(window.location.search).get('map')
        let pick = ms[0]
        if (param) {
          const byId = ms.find((m) => String(m.id) === param)
          const byName = byId ?? ms.find((m) => m.name === param)
          if (byName) pick = byName
        }
        await loadMap(pick.id)
      } catch (e) {
        if (!cancelled) setError((e as Error).message)
      } finally {
        if (!cancelled) setLoading(false)
      }
    })()
    return () => {
      cancelled = true
    }
  }, [refreshMaps, loadMap])

  const isMoving = useAppStore((s) => s.isMoving)
  const targetThetaSel = useAppStore((s) => s.targetTheta)
  const queueLen = useAppStore((s) => s.queuedActions.length)
  useEffect(() => {
    if (isMoving || targetThetaSel !== null) return
    if (queueLen === 0) return
    const next = useAppStore.getState().shiftQueuedAction()
    if (!next) return
    dispatchVoiceResponse({
      intent: next.intent,
      params: next.params,
      action_result: next.action_result,
    })
  }, [isMoving, targetThetaSel, queueLen])

  const onVoiceAudio = useCallback(async (blob: Blob) => {
    const map = useAppStore.getState().currentMap
    if (!map) return
    const robotPose = useAppStore.getState().robot
    const setVoicePipelineState = useAppStore.getState().setVoicePipelineState
    const setLastTranscription = useAppStore.getState().setLastTranscription
    const setLastLatencyMs = useAppStore.getState().setLastLatencyMs
    const setLastVoiceResponse = useAppStore.getState().setLastVoiceResponse
    const appendLog = useAppStore.getState().appendLog

    setVoiceProcessing(true)
    setVoicePipelineState('uploading')
    setLastTranscription('')
    const started = Date.now()

    const parsingTimer = window.setTimeout(() => {
      if (useAppStore.getState().voicePipelineState === 'uploading') {
        setVoicePipelineState('parsing')
      }
    }, 250)

    const memoryToSend = useAppStore.getState().voiceMemory

    try {
      const r = await sendVoice(blob, robotPose, map.id, memoryToSend)
      const ms = Date.now() - started
      window.clearTimeout(parsingTimer)
      setLastLatencyMs(ms)
      setLastVoiceResponse(r)

      const ar = r.action_result ?? {}
      const transcription =
        (typeof ar.original_text === 'string' && ar.original_text) ||
        (typeof r.params?.original_text === 'string' && r.params.original_text) ||
        ''
      setLastTranscription(transcription)

      const ok = isVoiceOk(r.intent, ar)
      const errMsg =
        typeof ar.error === 'string'
          ? ar.error
          : ar.reason === 'no_path'
            ? 'no path'
            : r.intent === 'UNKNOWN'
              ? 'not understood'
              : 'ok'
      appendLog({
        intent: r.intent,
        params: r.params ?? {},
        ok,
        msg: errMsg,
        src: transcription || '(voice)',
        latencyMs: ms,
      })

      dispatchVoiceResponse(r)

      if (Array.isArray(r.follow_ups) && r.follow_ups.length > 0) {
        useAppStore.getState().enqueueActions(r.follow_ups)
      }

      if (r.intent === 'NAVIGATE' || r.intent === 'RETURN_HOME') {
        const wp = ar.waypoints
        if (Array.isArray(wp) && wp.length > 0) {
          const last = wp[wp.length - 1] as { x: number; y: number }
          const targetName =
            typeof ar.target === 'string' && ar.target ? ar.target : 'goal'
          useAppStore.getState().setVoiceMemory({
            startedAt: { x: robotPose.x, y: robotPose.y, theta: robotPose.theta },
            goal: { x: last.x, y: last.y, name: targetName },
            ts: Date.now(),
          })
        }
      }

      setVoicePipelineState('done')
      window.setTimeout(() => {
        if (useAppStore.getState().voicePipelineState === 'done') {
          setVoicePipelineState('idle')
        }
      }, 900)
    } catch (e) {
      const ms = Date.now() - started
      window.clearTimeout(parsingTimer)
      setLastLatencyMs(ms)
      const msg = (e as Error).message
      toast.error(t('voice.error', { msg }))
      appendLog({
        intent: 'UNKNOWN',
        params: {},
        ok: false,
        msg,
        src: '(voice)',
        latencyMs: ms,
      })
      setVoicePipelineState('idle')
    } finally {
      setVoiceProcessing(false)
    }
  }, [t])

  const toolbarHint = mode !== '2d' ? t('toolbar.hint.3d') : t('toolbar.hint.2d')

  return (
    <div className="w-full h-screen flex flex-col bg-[#fafaf7] text-stone-900 antialiased overflow-hidden">
      <KinematicsRunner />
      <DebugControls />
      <Toaster />
      <CorrectionDialog />

      <TopMetaBar />
      <Header />

      <div className="grid grid-cols-12 gap-0 flex-1 min-h-0">
        <section className="col-span-12 md:col-span-9 border-r hairline bg-[#fafaf7] flex flex-col min-h-0">
          <div className="border-b hairline bg-white px-4 py-2 flex items-center justify-between gap-3 flex-wrap">
            <div className="flex items-center gap-2 flex-wrap">
              <div className="border hairline-strong flex">
                <button
                  type="button"
                  onClick={() => mode !== '2d' && switchMode()}
                  className={`px-3 py-1.5 upper-mono ${mode === '2d' ? 'bg-stone-900 text-white' : 'text-stone-600 hover:bg-stone-50'}`}
                >
                  {t('mode.2d')}
                </button>
                <button
                  type="button"
                  onClick={() => mode !== '3d' && switchMode()}
                  className={`px-3 py-1.5 upper-mono border-l hairline-strong ${mode === '3d' ? 'bg-stone-900 text-white' : 'text-stone-600 hover:bg-stone-50'}`}
                >
                  {t('mode.3d')}
                </button>
              </div>
              {mode === '2d' && (
                <>
                  <Toggle label={t('toggle.grid')} on={showGrid2D} onChange={setShowGrid2D} />
                  <Toggle
                    label={t('toggle.inflation')}
                    on={showInflation}
                    onChange={setShowInflation}
                  />
                  <div
                    className="border hairline-strong flex"
                    role="group"
                    aria-label={t('edit.title')}
                  >
                    <span className="px-2 py-1.5 upper-mono text-stone-500 border-r hairline-strong">
                      {t('edit.title')}
                    </span>
                    {(['off', 'paint', 'erase'] as const).map((m) => (
                      <button
                        key={m}
                        type="button"
                        onClick={() => setEditMode(m)}
                        className={`px-2 py-1.5 upper-mono ${
                          m !== 'off' ? 'border-l hairline-strong' : ''
                        } ${
                          editMode === m
                            ? m === 'paint'
                              ? 'bg-stone-900 text-white'
                              : m === 'erase'
                                ? 'bg-red-700 text-white'
                                : 'bg-stone-200 text-stone-900'
                            : 'text-stone-600 hover:bg-stone-50'
                        }`}
                      >
                        {t(`edit.${m}` as const)}
                      </button>
                    ))}
                  </div>
                </>
              )}
              <span className="hidden xl:inline upper-mono text-stone-400 ml-1">
                {mode === '2d' && editMode !== 'off' ? t('edit.hint') : toolbarHint}
              </span>
            </div>
          </div>

          <div className="flex-1 relative overflow-hidden min-h-0">
            {loading ? (
              <div className="absolute inset-0 flex items-center justify-center upper-mono text-stone-500 animate-pulse">
                {t('scene.loading')}
              </div>
            ) : error ? (
              <div className="absolute inset-0 flex items-center justify-center upper-mono text-red-700">
                {t('scene.error')} · {error}
              </div>
            ) : currentMap ? (
              <ErrorBoundary fallbackLabel={t('scene.renderError', { mode })}>
                {mode === '2d' ? <Scene2D /> : <Scene3D />}
              </ErrorBoundary>
            ) : (
              <div className="absolute inset-0 flex items-center justify-center upper-mono text-stone-400">
                {t('scene.noMap')}
              </div>
            )}
          </div>
        </section>

        <aside className="col-span-12 md:col-span-3 bg-white overflow-y-auto scrollbar-slim">
          <MapImportPanel />
          <TrainingPanel />
          <VoiceBar onAudio={onVoiceAudio} processing={voiceProcessing} />
          <SpacesPanel />
          <GoalPanel />
        </aside>
      </div>

      <BottomTicker />
    </div>
  )
}

function TopMetaBar() {
  const t = useT()
  const backendOk = useBackendHealth()
  const robot = useRobotStatus()
  return (
    <div className="border-b hairline bg-white">
      <div className="px-5 py-1.5 flex items-center justify-between upper-mono text-stone-600 num">
        <span className="text-stone-900 font-semibold">Diploma</span>
        <div className="flex items-center gap-5">
          <LangSwitch />
          <Clock />
          {robot && robot.mode === 'http' && (
            <span className="flex items-center gap-1.5">
              <span
                className={`w-1.5 h-1.5 rounded-full ${
                  robot.connected ? 'bg-emerald-600' : 'bg-red-600'
                }`}
              />
              <span className={robot.connected ? 'text-stone-900' : 'text-red-700'}>
                {robot.connected ? t('robot.online') : t('robot.offline')}
              </span>
            </span>
          )}
          <span className="flex items-center gap-1.5">
            <span
              className={`w-1.5 h-1.5 rounded-full ${backendOk ? 'bg-emerald-600' : 'bg-red-600'}`}
            />
            <span className={backendOk ? 'text-stone-900' : 'text-red-700'}>
              {backendOk ? t('meta.backend.ok') : t('meta.backend.down')}
            </span>
          </span>
        </div>
      </div>
    </div>
  )
}

function useRobotStatus(): RobotStatus | null {
  const [status, setStatus] = useState<RobotStatus | null>(null)
  useEffect(() => {
    let cancelled = false
    const tick = async () => {
      try {
        const r = await getRobotStatus()
        if (!cancelled) setStatus(r)
      } catch {
      }
    }
    void tick()
    const id = window.setInterval(tick, 5_000)
    return () => {
      cancelled = true
      window.clearInterval(id)
    }
  }, [])
  return status
}

function LangSwitch() {
  const locale = useLocale()
  const t = useT()
  return (
    <div className="border hairline-strong flex" role="group" aria-label={t('lang.toggle')}>
      {(['en', 'ua'] as const).map((l) => (
        <button
          key={l}
          type="button"
          onClick={() => setLocale(l)}
          className={`px-2 py-0.5 upper-mono text-[10.5px] ${
            locale === l ? 'bg-stone-900 text-white' : 'text-stone-600 hover:bg-stone-50'
          }`}
        >
          {l.toUpperCase()}
        </button>
      ))}
    </div>
  )
}

function Header() {
  const t = useT()
  return (
    <div className="border-b hairline bg-white">
      <div className="px-5 pt-4 pb-3">
        <div className="upper-mono text-stone-500 mb-1">{t('header.tag')}</div>
        <h1 className="font-display text-3xl font-light leading-none">
          {t('header.title')}{' '}
          <span className="italic text-stone-500">{t('header.subtitle')}</span>
        </h1>
      </div>
    </div>
  )
}

function BottomTicker() {
  return (
    <div className="border-t-2 border-stone-900 bg-stone-900 text-stone-300">
      <div className="px-5 py-2 flex items-center justify-between upper-mono num">
        <span className="text-stone-100">KPI / IK-23</span>
        <span className="text-stone-100">Y. Mynenko</span>
      </div>
    </div>
  )
}

function useBackendHealth(): boolean {
  const [ok, setOk] = useState(true)
  useEffect(() => {
    const base = (import.meta.env.VITE_API_URL as string) ?? 'http://localhost:8000'
    let cancelled = false
    const ping = async () => {
      try {
        const r = await fetch(`${base}/api/health`, { method: 'GET' })
        if (!cancelled) setOk(r.ok)
      } catch {
        if (!cancelled) setOk(false)
      }
    }
    void ping()
    const id = window.setInterval(ping, 15_000)
    return () => {
      cancelled = true
      window.clearInterval(id)
    }
  }, [])
  return ok
}

export default App
