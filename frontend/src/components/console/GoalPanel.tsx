import { useT } from '../../i18n'
import { useAppStore } from '../../state/useAppStore'
import { KV, Section } from './primitives'
import { LINEAR_SPEED } from '../../sim/kinematics'

function GoalPanel() {
  const t = useT()
  const goal = useAppStore((s) => s.currentGoal)
  const robot = useAppStore((s) => s.robot)
  const path = useAppStore((s) => s.path)
  const pathIndex = useAppStore((s) => s.pathIndex)
  const stopMovement = useAppStore((s) => s.stopMovement)
  const appendLog = useAppStore((s) => s.appendLog)

  const onStop = () => {
    if (!goal) return
    stopMovement()
    appendLog({
      intent: 'STOP_BTN',
      params: { name: goal.name },
      ok: true,
      msg: 'manual stop',
      src: '(UI)',
    })
  }

  return (
    <Section title={t('goal.title')}>
      {goal ? (
        <div className="px-3 pb-3 text-[12.5px]">
          <div className="font-display text-2xl text-stone-900 mb-1">{goal.name}</div>
          <KV k={t('goal.x')} v={`${goal.x.toFixed(3)} m`} />
          <KV k={t('goal.y')} v={`${goal.y.toFixed(3)} m`} />
          <KV
            k={t('goal.distance')}
            v={`${Math.hypot(goal.x - robot.x, goal.y - robot.y).toFixed(2)} m`}
          />
          <KV k={t('goal.waypoints')} v={`${Math.min(pathIndex + 1, path.length)} / ${path.length}`} />
          <KV
            k={t('goal.eta')}
            v={`~ ${Math.max(0, ((path.length - pathIndex) * 0.1) / LINEAR_SPEED).toFixed(1)} s`}
          />
          <button
            type="button"
            onClick={onStop}
            className="mt-3 w-full border-2 border-red-700 text-red-700 py-1.5 upper-mono hover:bg-red-700 hover:text-white transition-colors"
          >
            {t('goal.stop')}
          </button>
        </div>
      ) : (
        <div className="px-3 pb-3 text-[12.5px] text-stone-500 italic">
          {t('goal.empty')}
        </div>
      )}
    </Section>
  )
}

export default GoalPanel
