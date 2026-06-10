import type {
  Intent,
  MapCreate,
  MapResponse,
  MapSummary,
  MapUpdate,
  PlanResponse,
  Point2D,
  RobotPose,
  SpaceCreate,
  SpaceResponse,
  SpaceUpdate,
  VoiceFeedbackResponse,
  VoiceResponse,
} from '../types'

const BASE_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${BASE_URL}${path}`
  let response: Response
  try {
    response = await fetch(url, init)
  } catch (err) {
    console.error(`[api] ${init?.method ?? 'GET'} ${url} — network error`, err)
    throw err
  }

  if (!response.ok) {
    let detail: string | undefined
    try {
      const body = (await response.clone().json()) as { detail?: string }
      detail = typeof body.detail === 'string' ? body.detail : undefined
    } catch {
    }
    const message = `${response.status} ${response.statusText}${detail ? ` — ${detail}` : ''}`
    console.error(`[api] ${init?.method ?? 'GET'} ${url} — ${message}`)
    throw new Error(message)
  }

  if (response.status === 204) {
    return undefined as T
  }
  return (await response.json()) as T
}

function jsonInit(method: string, body: unknown): RequestInit {
  return {
    method,
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  }
}

export function listMaps(): Promise<MapSummary[]> {
  return request<MapSummary[]>('/api/maps')
}

export function getMap(id: number): Promise<MapResponse> {
  return request<MapResponse>(`/api/maps/${id}`)
}

export function createMap(payload: MapCreate): Promise<MapResponse> {
  return request<MapResponse>('/api/maps', jsonInit('POST', payload))
}

export function updateMap(id: number, payload: MapUpdate): Promise<MapResponse> {
  return request<MapResponse>(`/api/maps/${id}`, jsonInit('PUT', payload))
}

export function deleteMap(id: number): Promise<void> {
  return request<void>(`/api/maps/${id}`, { method: 'DELETE' })
}

export interface MapFromImageOptions {
  cellSizeM?: number
  maxCells?: number
  invert?: boolean
  dilate?: number
}

export function createMapFromImage(
  file: File,
  name: string,
  options: MapFromImageOptions = {},
): Promise<MapResponse> {
  const fd = new FormData()
  fd.append('file', file)
  fd.append('name', name)
  if (options.cellSizeM !== undefined) fd.append('cell_size_m', String(options.cellSizeM))
  if (options.maxCells !== undefined) fd.append('max_cells', String(options.maxCells))
  if (options.invert !== undefined) fd.append('invert', String(options.invert))
  if (options.dilate !== undefined) fd.append('dilate', String(options.dilate))
  return request<MapResponse>('/api/maps/from-image', { method: 'POST', body: fd })
}

export function listSpaces(mapId: number): Promise<SpaceResponse[]> {
  return request<SpaceResponse[]>(`/api/maps/${mapId}/spaces`)
}

export function createSpace(payload: SpaceCreate): Promise<SpaceResponse> {
  return request<SpaceResponse>('/api/spaces', jsonInit('POST', payload))
}

export function patchSpace(id: number, payload: SpaceUpdate): Promise<SpaceResponse> {
  return request<SpaceResponse>(`/api/spaces/${id}`, jsonInit('PATCH', payload))
}

export function deleteSpace(id: number): Promise<void> {
  return request<void>(`/api/spaces/${id}`, { method: 'DELETE' })
}

export function plan(mapId: number, from: Point2D, to: Point2D): Promise<PlanResponse> {
  return request<PlanResponse>(
    '/api/plan',
    jsonInit('POST', { map_id: mapId, from, to }),
  )
}

export function sendVoice(
  audio: Blob,
  robotPose: RobotPose,
  mapId: number,
  voiceMemory?: unknown,
): Promise<VoiceResponse> {
  const fd = new FormData()
  fd.append('audio', audio, 'voice.webm')
  fd.append('robot_pos', JSON.stringify(robotPose))
  fd.append('map_id', String(mapId))
  if (voiceMemory) fd.append('voice_memory', JSON.stringify(voiceMemory))
  return request<VoiceResponse>('/api/voice', { method: 'POST', body: fd })
}

export function sendVoiceFeedback(
  commandLogId: number,
  wasCorrect: boolean,
  robotPose: RobotPose,
  correctedIntent?: Intent,
  correctedParams?: Record<string, unknown>,
): Promise<VoiceFeedbackResponse> {
  return request<VoiceFeedbackResponse>(
    '/api/voice/feedback',
    jsonInit('POST', {
      command_log_id: commandLogId,
      was_correct: wasCorrect,
      corrected_intent: correctedIntent,
      corrected_params: correctedParams,
      robot_pose: robotPose,
    }),
  )
}

export interface FeedbackStats {
  pending_count: number
  last_retrained_at: string | null
  retrain_state: 'idle' | 'running' | 'completed' | 'failed'
}

export interface RetrainStatus {
  state: 'idle' | 'running' | 'completed' | 'failed'
  phase: string | null
  started_at: string | null
  finished_at: string | null
  elapsed_seconds: number
  error: string | null
}

export function getFeedbackStats(): Promise<FeedbackStats> {
  return request<FeedbackStats>('/api/voice/feedback/stats')
}

export function triggerRetrain(): Promise<RetrainStatus> {
  return request<RetrainStatus>(
    '/api/voice/feedback/retrain',
    jsonInit('POST', {}),
  )
}

export function getRetrainStatus(): Promise<RetrainStatus> {
  return request<RetrainStatus>('/api/voice/feedback/retrain/status')
}

export interface RobotStatus {
  mode: 'sim' | 'http'
  connected: boolean
  last_pose: { x: number; y: number; theta: number } | null
  age_s: number | null
  last_error: string | null
}

export function getRobotStatus(): Promise<RobotStatus> {
  return request<RobotStatus>('/api/robot/status')
}

export interface CorrectionRow {
  id: number
  transcription: string
  predicted_intent: string
  predicted_confidence: number
  predicted_params: Record<string, unknown>
  was_correct: boolean
  corrected_intent: string | null
  corrected_params: Record<string, unknown> | null
  created_at: string
  effective_intent: string
  effective_params: Record<string, unknown>
}

export function listCorrections(pendingOnly = true): Promise<CorrectionRow[]> {
  const qs = pendingOnly ? '' : '?pending_only=false'
  return request<CorrectionRow[]>(`/api/voice/feedback/corrections${qs}`)
}

export function patchCorrection(
  id: number,
  body: { corrected_intent?: Intent; corrected_params?: Record<string, unknown> },
): Promise<CorrectionRow> {
  return request<CorrectionRow>(
    `/api/voice/feedback/corrections/${id}`,
    jsonInit('PATCH', body),
  )
}

export function deleteCorrection(id: number): Promise<void> {
  return request<void>(`/api/voice/feedback/corrections/${id}`, {
    method: 'DELETE',
  })
}
