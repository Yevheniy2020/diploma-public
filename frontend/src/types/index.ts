export type Intent =
  | 'NAVIGATE'
  | 'DRIVE_RELATIVE'
  | 'ROTATE'
  | 'DELETE_SPACE'
  | 'RENAME_SPACE'
  | 'START_SPACE'
  | 'FINISH_SPACE'
  | 'CANCEL_SPACE'
  | 'STOP'
  | 'RETURN_HOME'
  | 'UNKNOWN'
  | 'UNCERTAIN'

// Intents the operator can pick when overriding an UNCERTAIN prediction.
// UNCERTAIN itself is not selectable (it's the band, not a target).
// UNKNOWN is added as a separate option in the dropdown rendering.
// All 10 other Intent union members are user-selectable here.
export const ACTION_INTENTS: Intent[] = [
  'NAVIGATE',
  'DRIVE_RELATIVE',
  'ROTATE',
  'DELETE_SPACE',
  'RENAME_SPACE',
  'START_SPACE',
  'FINISH_SPACE',
  'CANCEL_SPACE',
  'STOP',
  'RETURN_HOME',
]

export interface Point2D {
  x: number
  y: number
}

export interface RobotPose {
  x: number
  y: number
  theta: number
}

export interface MapSummary {
  id: number
  name: string
  width_cells: number
  height_cells: number
  cell_size_m: number
  scene_glb_path: string | null
}

export interface MapResponse {
  id: number
  name: string
  width_cells: number
  height_cells: number
  cell_size_m: number
  grid_data_b64: string | null
  scene_glb_path: string | null
  scene_origin_x: number
  scene_origin_y: number
  scene_scale: number
}

export interface MapCreate {
  name: string
  width_cells: number
  height_cells: number
  cell_size_m?: number
}

export interface MapUpdate {
  name?: string
  grid_data_b64?: string
  scene_glb_path?: string
  scene_origin_x?: number
  scene_origin_y?: number
  scene_scale?: number
}

export interface SpaceResponse {
  id: number
  map_id: number
  name: string
  vertices: [number, number][]
  is_home: boolean
}

export interface SpaceCreate {
  map_id: number
  name: string
  vertices: [number, number][]
  is_home?: boolean
}

export interface SpaceUpdate {
  name?: string
  is_home?: boolean
}

export interface PlanRequest {
  map_id: number
  from: Point2D
  to: Point2D
}

export interface PlanResponse {
  waypoints: Point2D[]
  reason?: string | null
}

export interface VoiceFollowUp {
  intent: Intent
  params: Record<string, unknown>
  action_result: Record<string, unknown>
}

export interface VoiceResponse {
  intent: Intent
  params: Record<string, unknown>
  action_result: Record<string, unknown>
  follow_ups?: VoiceFollowUp[]
  // Set when intent === 'UNCERTAIN'. Frontend POSTs this back to
  // /api/voice/feedback with the operator's verdict.
  command_log_id?: number
}

export interface VoiceFeedbackResponse {
  recorded: boolean
  intent?: Intent
  params?: Record<string, unknown>
  action_result?: Record<string, unknown>
}

// Mid-confidence prediction parked in the Zustand store while the
// operator decides via CorrectionDialog.
export interface PendingCorrection {
  command_log_id: number
  transcription: string
  predicted_intent: Intent
  predicted_params: Record<string, unknown>
  confidence: number
}
