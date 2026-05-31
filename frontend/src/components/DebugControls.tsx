// Hotkey-only debug helpers — no UI.
// Arrows = manual move/turn, S = stop, R = reset to (0,0,0).
// Space is reserved for push-to-talk (VoiceButton owns it).
import { useEffect } from 'react'
import { useAppStore } from '../state/useAppStore'

const STEP_M = 0.2
const TURN_RAD = 0.1

function isTextInputFocused(): boolean {
  const a = document.activeElement
  if (!a) return false
  const tag = a.tagName
  return (
    tag === 'INPUT' ||
    tag === 'TEXTAREA' ||
    (a as HTMLElement).isContentEditable === true
  )
}

function move(forward: number) {
  const s = useAppStore.getState()
  const r = s.robot
  s.setRobotPose({
    x: r.x + Math.cos(r.theta) * forward,
    y: r.y + Math.sin(r.theta) * forward,
    theta: r.theta,
  })
}

function turn(delta: number) {
  const s = useAppStore.getState()
  const r = s.robot
  s.setRobotPose({ ...r, theta: r.theta + delta })
}

function reset() {
  const s = useAppStore.getState()
  s.setRobotPose({ x: 0, y: 0, theta: 0 })
  s.stopMovement()
}

export default function DebugControls() {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (isTextInputFocused()) return
      switch (e.key) {
        case 'ArrowUp':
          move(STEP_M)
          break
        case 'ArrowDown':
          move(-STEP_M)
          break
        case 'ArrowLeft':
          turn(TURN_RAD)
          break
        case 'ArrowRight':
          turn(-TURN_RAD)
          break
        case 's':
        case 'S':
          useAppStore.getState().stopMovement()
          break
        case 'r':
        case 'R':
          reset()
          break
        default:
          return
      }
      e.preventDefault()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  return null
}
