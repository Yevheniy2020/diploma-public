import { useT } from '../../i18n'
import type { Intent } from '../../types'

export interface SlotState {
  delta_deg: number
  direction: 'forward' | 'backward' | 'left' | 'right'
  distance_m: number
  target_or_name: string
  new_name: string
  old_name: string
  radius: number
}

const INPUT_CLS =
  'bg-white border hairline-strong px-3 py-2 outline-none focus:border-stone-700 text-stone-900 text-[13px] font-mono w-full'

const ROTATE_CHIPS = [-360, -270, -180, -90, 90, 180, 270, 360]
const DIRECTION_CHIPS: Array<{
  value: SlotState['direction']
  label: string
}> = [
  { value: 'forward', label: 'forward' },
  { value: 'backward', label: 'backward' },
  { value: 'left', label: 'left' },
  { value: 'right', label: 'right' },
]

function clamp(n: number, lo: number, hi: number): number {
  if (Number.isNaN(n)) return lo
  return Math.max(lo, Math.min(hi, n))
}

interface Props {
  intent: Intent
  slots: SlotState
  setSlots: (next: SlotState) => void
  labels: string[]
}

function SlotEditor({ intent, slots, setSlots, labels }: Props) {
  const t = useT()
  const set = (patch: Partial<SlotState>) => setSlots({ ...slots, ...patch })

  if (
    intent === 'STOP' ||
    intent === 'RETURN_HOME' ||
    intent === 'FINISH_SPACE' ||
    intent === 'CANCEL_SPACE' ||
    intent === 'UNKNOWN'
  ) {
    return null
  }

  return (
    <div className="flex flex-col gap-2 text-[12px]">
      <span className="upper-mono text-stone-500">{t('slot.parameters')}</span>

      {intent === 'ROTATE' ? (
        <>
          <label className="flex items-center gap-2">
            <span className="font-mono text-stone-700 w-20 shrink-0">delta_deg</span>
            <input
              type="number"
              min={-360}
              max={360}
              step={1}
              value={Number.isFinite(slots.delta_deg) ? slots.delta_deg : 0}
              onChange={(e) =>
                set({ delta_deg: clamp(Number(e.target.value), -360, 360) })
              }
              className={INPUT_CLS}
            />
          </label>
          <div className="flex flex-wrap gap-1">
            {ROTATE_CHIPS.map((deg) => (
              <button
                key={deg}
                type="button"
                onClick={() => set({ delta_deg: deg })}
                className={`px-2 py-0.5 border text-[11px] font-mono num transition-colors ${
                  slots.delta_deg === deg
                    ? 'bg-stone-900 text-white border-stone-900'
                    : 'hairline-strong text-stone-700 hover:bg-stone-50'
                }`}
              >
                {deg > 0 ? `+${deg}` : deg}°
              </button>
            ))}
          </div>
        </>
      ) : null}

      {intent === 'DRIVE_RELATIVE' ? (
        <>
          <label className="flex items-center gap-2">
            <span className="font-mono text-stone-700 w-20 shrink-0">direction</span>
            <select
              value={slots.direction}
              onChange={(e) => set({ direction: e.target.value as SlotState['direction'] })}
              className={INPUT_CLS}
            >
              {DIRECTION_CHIPS.map((d) => (
                <option key={d.value} value={d.value}>
                  {d.label}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-2">
            <span className="font-mono text-stone-700 w-20 shrink-0">distance_m</span>
            <input
              type="number"
              step="0.1"
              min="0.05"
              max="10"
              value={Number.isFinite(slots.distance_m) ? slots.distance_m : 1.0}
              onChange={(e) =>
                set({ distance_m: clamp(Number(e.target.value), 0.05, 10) })
              }
              className={INPUT_CLS}
            />
          </label>
        </>
      ) : null}

      {intent === 'NAVIGATE' || intent === 'DELETE_SPACE' ? (
        <label className="flex items-center gap-2">
          <span className="font-mono text-stone-700 w-20 shrink-0">
            {intent === 'NAVIGATE' ? 'target' : 'name'}
          </span>
          <select
            value={slots.target_or_name}
            onChange={(e) => set({ target_or_name: e.target.value })}
            className={INPUT_CLS}
          >
            <option value="">{t('slot.choose')}</option>
            {labels.map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </label>
      ) : null}

      {intent === 'RENAME_SPACE' ? (
        <>
          <label className="flex items-center gap-2">
            <span className="font-mono text-stone-700 w-20 shrink-0">old_name</span>
            <select
              value={slots.old_name}
              onChange={(e) => set({ old_name: e.target.value })}
              className={INPUT_CLS}
            >
              <option value="">{t('slot.choose')}</option>
              {labels.map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-2">
            <span className="font-mono text-stone-700 w-20 shrink-0">new_name</span>
            <input
              type="text"
              value={slots.new_name}
              onChange={(e) => set({ new_name: e.target.value })}
              placeholder="їдальня"
              className={INPUT_CLS}
            />
          </label>
        </>
      ) : null}

      {intent === 'START_SPACE' ? (
        <label className="flex items-center gap-2">
          <span className="font-mono text-stone-700 w-20 shrink-0">name</span>
          <input
            type="text"
            value={slots.target_or_name}
            onChange={(e) => set({ target_or_name: e.target.value })}
            placeholder="спальня"
            className={INPUT_CLS}
          />
        </label>
      ) : null}
    </div>
  )
}

export default SlotEditor
