import { create } from 'zustand'
import { getMap, listMaps, listSpaces } from '../api/client'
import type {
  Intent,
  MapResponse,
  MapSummary,
  PendingCorrection,
  Point2D,
  RobotPose,
  SpaceResponse,
  SpaceUpdate,
  VoiceFollowUp,
  VoiceResponse,
} from '../types'
type Mode = '2d' | '3d'
export type EditMode = 'off' | 'paint' | 'erase'

export interface ActiveGoal {
  x: number
  y: number
  name: string
}

// Anaphora memory: lets the next voice command refer to "назад" / "ще раз".
// startedAt is the robot pose at the moment the last NAVIGATE was issued;
// goal is where that navigation aimed.
export interface VoiceMemory {
  startedAt: { x: number; y: number; theta: number }
  goal: { x: number; y: number; name: string }
  ts: number
}

export interface CommandLogEntry {
  id: number
  ts: string
  intent: Intent | 'CLICK_NAV' | 'STOP_BTN'
  params: Record<string, unknown>
  ok: boolean
  msg: string
  src: string
  latencyMs?: number
}

interface AppState {
  maps: MapSummary[]
  currentMap: MapResponse | null
  spaces: SpaceResponse[]
  robot: RobotPose
  robotV: number
  robotW: number
  path: Point2D[]
  pathIndex: number
  mode: Mode
  isMoving: boolean
  isRecording: boolean
  lastVoiceResponse: VoiceResponse | null
  currentGoal: ActiveGoal | null
  commandLog: CommandLogEntry[]
  showInflation: boolean
  showGrid2D: boolean
  editMode: EditMode
  voicePipelineState: 'idle' | 'recording' | 'uploading' | 'parsing' | 'done'
  lastTranscription: string
  lastLatencyMs: number | null
  voiceMemory: VoiceMemory | null
  // Pure rotation target (radians, world frame). When set, KinematicsRunner
  // lerps theta toward it without translating; cleared on arrival or on
  // any other motion command.
  targetTheta: number | null
  // Queued follow-up actions from a multi-step voice command. App.tsx
  // drains this whenever the robot becomes idle.
  queuedActions: VoiceFollowUp[]
  // Mid-confidence prediction parked for operator confirmation. While
  // non-null the CorrectionDialog modal is open. Cleared by setting back
  // to null on either Yes / No / override.
  pendingCorrection: PendingCorrection | null
  // Voice-perimeter drafting state. When non-null the user has said
  // START_SPACE and the robot's pose (or canvas clicks) accumulate into
  // `points`. Cleared on FINISH_SPACE (after POST) or CANCEL_SPACE.
  draftSpace: { name: string; points: [number, number][] } | null

  refreshMaps: () => Promise<MapSummary[]>
  loadMap: (id: number) => Promise<void>
  setRobotPose: (pose: RobotPose) => void
  setRobotKinematics: (v: number, w: number) => void
  setPath: (path: Point2D[], goal?: ActiveGoal | null) => void
  setGoal: (goal: ActiveGoal | null) => void
  stopMovement: () => void
  // Natural finish of a path. Like stopMovement but keeps queuedActions
  // intact so a multi-step voice command can continue dispatching the
  // next frame when the robot becomes idle.
  pathCompleted: () => void
  switchMode: () => void
  setRecording: (v: boolean) => void
  setLastVoiceResponse: (v: VoiceResponse | null) => void
  appendLog: (entry: Omit<CommandLogEntry, 'id' | 'ts'>) => void
  clearLog: () => void
  setShowInflation: (v: boolean) => void
  setShowGrid2D: (v: boolean) => void
  setEditMode: (m: EditMode) => void
  updateMapGridLocal: (b64: string) => void
  setVoicePipelineState: (s: AppState['voicePipelineState']) => void
  setLastTranscription: (s: string) => void
  setLastLatencyMs: (n: number | null) => void
  setVoiceMemory: (m: VoiceMemory | null) => void
  rotateTo: (theta: number) => void
  enqueueActions: (actions: VoiceFollowUp[]) => void
  shiftQueuedAction: () => VoiceFollowUp | undefined
  clearActionQueue: () => void
  setPendingCorrection: (c: PendingCorrection | null) => void
  addSpace: (space: SpaceResponse) => void
  removeSpace: (id: number) => void
  patchSpaceLocal: (id: number, patch: SpaceUpdate) => void
  startSpaceDraft: (name: string) => void
  appendDraftPoint: (pt: [number, number]) => void
  finishSpaceDraft: () => [number, number][] | null
  cancelSpaceDraft: () => void
}

const LOG_LIMIT = 30
let _logSeq = 0

function nowStr(): string {
  return new Date().toTimeString().slice(0, 8)
}

function polygonCentroid(vertices: [number, number][]): { x: number; y: number } {
  // Vertex-average centroid — matches the backend's helper, good enough
  // for spawn placement on roughly convex spaces.
  if (vertices.length === 0) return { x: 0, y: 0 }
  let sx = 0
  let sy = 0
  for (const [vx, vy] of vertices) {
    sx += vx
    sy += vy
  }
  return { x: sx / vertices.length, y: sy / vertices.length }
}

export const useAppStore = create<AppState>((set, get) => ({
  maps: [],
  currentMap: null,
  spaces: [],
  robot: { x: 0, y: 0, theta: 0 },
  robotV: 0,
  robotW: 0,
  path: [],
  pathIndex: 0,
  mode: '2d',
  isMoving: false,
  isRecording: false,
  lastVoiceResponse: null,
  currentGoal: null,
  commandLog: [],
  showInflation: true,
  showGrid2D: true,
  editMode: 'off',
  voicePipelineState: 'idle',
  lastTranscription: '',
  lastLatencyMs: null,
  voiceMemory: null,
  targetTheta: null,
  queuedActions: [],
  pendingCorrection: null,
  draftSpace: null,

  refreshMaps: async () => {
    const maps = await listMaps()
    set({ maps })
    return maps
  },

  loadMap: async (id) => {
    const [map, spaces] = await Promise.all([getMap(id), listSpaces(id)])
    // Spawn at the centroid of the home space. If no space is flagged
    // is_home (shouldn't happen for seeded maps, but be defensive),
    // fall back to the first space's centroid, then to origin.
    const homeSpace = spaces.find((r) => r.is_home) ?? spaces[0] ?? null
    const spawn = homeSpace ? polygonCentroid(homeSpace.vertices) : { x: 0, y: 0 }
    set({
      currentMap: map,
      spaces,
      robot: { x: spawn.x, y: spawn.y, theta: 0 },
      path: [],
      pathIndex: 0,
      isMoving: false,
      currentGoal: null,
      voiceMemory: null,
      queuedActions: [],
      targetTheta: null,
      draftSpace: null,
    })
  },

  setRobotPose: (pose) => set({ robot: pose }),

  setRobotKinematics: (v, w) => set({ robotV: v, robotW: w }),

  setPath: (path, goal) =>
    set((s) => {
      // Drop any waypoint that is non-finite — a bad offset / planner glitch
      // used to NaN-out the robot pose downstream and crash the 3D scene.
      const safe = path.filter(
        (p) =>
          Number.isFinite(p?.x) &&
          Number.isFinite(p?.y),
      )
      return {
        path: safe,
        pathIndex: 0,
        isMoving: safe.length > 0,
        currentGoal: goal !== undefined ? goal : s.currentGoal,
      }
    }),

  setGoal: (goal) => set({ currentGoal: goal }),

  stopMovement: () =>
    set({
      isMoving: false,
      path: [],
      pathIndex: 0,
      currentGoal: null,
      robotV: 0,
      robotW: 0,
      targetTheta: null,
      // STOP intent must also drop any queued chain — user explicitly
      // halted, don't keep executing what came before.
      queuedActions: [],
    }),

  // Natural path completion (kinematics reached the last waypoint).
  // Same as stopMovement except the multi-step action queue is
  // preserved so the App.tsx watcher can pop the next frame and the
  // chain continues («поверни ліворуч і вперед на 2 метри» actually
  // runs the second segment).
  pathCompleted: () =>
    set({
      isMoving: false,
      path: [],
      pathIndex: 0,
      currentGoal: null,
      robotV: 0,
      robotW: 0,
      targetTheta: null,
    }),

  addSpace: (space) =>
    set((s) => {
      // If the new space is flagged is_home, unset the flag on any other
      // space locally to mirror the backend's partial-unique invariant.
      const next = space.is_home
        ? s.spaces.map((r) => (r.is_home ? { ...r, is_home: false } : r))
        : s.spaces.slice()
      return { spaces: [...next, space] }
    }),

  removeSpace: (id) =>
    set((s) => ({ spaces: s.spaces.filter((r) => r.id !== id) })),

  patchSpaceLocal: (id, patch) =>
    set((s) => {
      const willBeHome = patch.is_home === true
      return {
        spaces: s.spaces.map((r) => {
          if (r.id === id) return { ...r, ...patch }
          if (willBeHome && r.is_home) return { ...r, is_home: false }
          return r
        }),
      }
    }),

  startSpaceDraft: (name) =>
    set({ draftSpace: { name: name.trim().toLowerCase(), points: [] } }),

  appendDraftPoint: (pt) =>
    set((s) => {
      if (s.draftSpace === null) return s
      // De-bounce duplicate / near-duplicate points (0.05 m) — coalesces
      // static jitter while still picking up real motion.
      const pts = s.draftSpace.points
      if (pts.length > 0) {
        const [lx, ly] = pts[pts.length - 1]
        const dx = pt[0] - lx
        const dy = pt[1] - ly
        if (dx * dx + dy * dy < 0.05 * 0.05) return s
      }
      return { draftSpace: { ...s.draftSpace, points: [...pts, pt] } }
    }),

  finishSpaceDraft: () => {
    // Returns the raw points for the caller (voiceDispatch) to simplify
    // and POST. We don't clear state here so a failed POST can retry.
    const draft = get().draftSpace
    return draft ? draft.points.slice() : null
  },

  cancelSpaceDraft: () => set({ draftSpace: null }),

  switchMode: () => set((s) => ({ mode: s.mode === '2d' ? '3d' : '2d' })),

  setRecording: (v) => set({ isRecording: v }),

  setLastVoiceResponse: (v) => set({ lastVoiceResponse: v }),

  appendLog: (entry) =>
    set((s) => ({
      commandLog: [
        { ...entry, id: ++_logSeq, ts: nowStr() },
        ...s.commandLog,
      ].slice(0, LOG_LIMIT),
    })),

  clearLog: () => set({ commandLog: [] }),

  setShowInflation: (v) => set({ showInflation: v }),

  setShowGrid2D: (v) => set({ showGrid2D: v }),

  setEditMode: (editMode) => set({ editMode }),

  updateMapGridLocal: (b64) =>
    set((s) =>
      s.currentMap ? { currentMap: { ...s.currentMap, grid_data_b64: b64 } } : s,
    ),

  setVoicePipelineState: (voicePipelineState) => set({ voicePipelineState }),

  setLastTranscription: (lastTranscription) => set({ lastTranscription }),

  setLastLatencyMs: (lastLatencyMs) => set({ lastLatencyMs }),

  setVoiceMemory: (voiceMemory) => set({ voiceMemory }),

  rotateTo: (theta) =>
    // Pure rotation cancels any in-flight path so the robot doesn't fight
    // itself. KinematicsRunner picks up targetTheta and lerps to it.
    set({
      targetTheta: theta,
      path: [],
      pathIndex: 0,
      isMoving: false,
      currentGoal: null,
    }),

  enqueueActions: (actions) =>
    set((s) => ({ queuedActions: [...s.queuedActions, ...actions] })),

  shiftQueuedAction: () => {
    const q = useAppStore.getState().queuedActions
    if (q.length === 0) return undefined
    const [next, ...rest] = q
    set({ queuedActions: rest })
    return next
  },

  clearActionQueue: () => set({ queuedActions: [] }),

  setPendingCorrection: (c) => set({ pendingCorrection: c }),
}))
