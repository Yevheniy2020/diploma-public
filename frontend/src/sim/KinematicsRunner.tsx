// Single rAF source of truth for robot motion — runs regardless of mode (2D/3D).
// Uses useAppStore.getState() inside the loop instead of selectors to avoid
// React re-renders on every tick. Also derives v/ω from pose deltas so the
// pose panel has live numbers without rewriting kinematics.ts.
import { useEffect } from 'react'
import { useAppStore } from '../state/useAppStore'
import { ANGLE_THRESHOLD, ANGULAR_SPEED, DT_MAX, lerpAngle, stepRobot, wrapAngle } from './kinematics'

function shortestAngle(a: number, b: number): number {
  return wrapAngle(b - a)
}

export default function KinematicsRunner() {
  useEffect(() => {
    let last = performance.now()
    let raf = 0

    const tick = (now: number) => {
      const dt = (now - last) / 1000
      last = now
      const s = useAppStore.getState()

      // Pure-rotation mode (ROTATE intent). Takes priority over path-follow
      // because rotateTo() clears the path anyway. lerpAngle handles wrap.
      if (s.targetTheta !== null && dt > 0) {
        const before = s.robot
        const safeDt = Math.min(dt, DT_MAX)
        const newTheta = lerpAngle(before.theta, s.targetTheta, ANGULAR_SPEED * safeDt)
        const remaining = Math.abs(wrapAngle(s.targetTheta - newTheta))
        s.setRobotPose({ ...before, theta: newTheta })
        s.setRobotKinematics(0, shortestAngle(before.theta, newTheta) / dt)
        if (remaining < ANGLE_THRESHOLD) {
          useAppStore.setState({ targetTheta: null })
          s.setRobotKinematics(0, 0)
        }
        raf = requestAnimationFrame(tick)
        return
      }

      if (s.isMoving && s.path.length > 0 && dt > 0) {
        const before = s.robot
        const res = stepRobot(before, s.path, s.pathIndex, dt)
        // Guard against NaN/Infinity pose (e.g. malformed waypoint slipped
        // through). One bad pose used to crash the whole 3D scene because
        // R3F has no recovery path for non-finite transforms.
        if (
          !Number.isFinite(res.robot.x) ||
          !Number.isFinite(res.robot.y) ||
          !Number.isFinite(res.robot.theta)
        ) {
          console.warn('[KinematicsRunner] bad pose, stopping motion', res.robot)
          s.stopMovement()
        } else {
          const dx = res.robot.x - before.x
          const dy = res.robot.y - before.y
          const v = Math.hypot(dx, dy) / dt
          const w = shortestAngle(before.theta, res.robot.theta) / dt
          s.setRobotPose(res.robot)
          s.setRobotKinematics(v, w)
          if (res.pathIndex !== s.pathIndex) {
            useAppStore.setState({ pathIndex: res.pathIndex })
          }
          // Voice-perimeter sampling: while drafting a space, append the
          // robot's pose every ≥0.15 m of motion. The store action
          // additionally de-bounces 0.05 m duplicates.
          if (s.draftSpace !== null) {
            const pts = s.draftSpace.points
            if (pts.length === 0) {
              s.appendDraftPoint([res.robot.x, res.robot.y])
            } else {
              const [lx, ly] = pts[pts.length - 1]
              const ddx = res.robot.x - lx
              const ddy = res.robot.y - ly
              if (ddx * ddx + ddy * ddy >= 0.15 * 0.15) {
                s.appendDraftPoint([res.robot.x, res.robot.y])
              }
            }
          }
          if (res.done) {
            // Natural finish — keep queuedActions so the next chain
            // frame can dispatch via the App.tsx idle watcher.
            s.pathCompleted()
          }
        }
      } else if (s.robotV !== 0 || s.robotW !== 0) {
        s.setRobotKinematics(0, 0)
      }
      raf = requestAnimationFrame(tick)
    }

    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [])

  return null
}
