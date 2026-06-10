import type { Point2D, RobotPose } from '../types'

export const LINEAR_SPEED = 0.5
export const ANGULAR_SPEED = 1.5
export const ARRIVAL_THRESHOLD = 0.05
export const ANGLE_THRESHOLD = 0.05
export const DT_MAX = 0.05

const FORWARD_GATE = 0.7

export function wrapAngle(a: number): number {
  let n = ((a + Math.PI) % (2 * Math.PI)) - Math.PI
  if (n <= -Math.PI) n += 2 * Math.PI
  return n
}

export function lerpAngle(current: number, target: number, maxDelta: number): number {
  const diff = wrapAngle(target - current)
  const step = Math.sign(diff) * Math.min(Math.abs(diff), maxDelta)
  return current + step
}

export interface StepResult {
  robot: RobotPose
  pathIndex: number
  done: boolean
}

export function stepRobot(
  robot: RobotPose,
  path: Point2D[],
  pathIndex: number,
  dt: number,
): StepResult {
  const safeDt = Math.min(dt, DT_MAX)

  if (path.length === 0 || pathIndex >= path.length) {
    return { robot, pathIndex, done: true }
  }

  const target = path[pathIndex]
  const dx = target.x - robot.x
  const dy = target.y - robot.y
  const distance = Math.hypot(dx, dy)

  let nextIndex = pathIndex
  if (distance < ARRIVAL_THRESHOLD) {
    nextIndex = pathIndex + 1
    if (nextIndex >= path.length) {
      return { robot, pathIndex: nextIndex, done: true }
    }
    return stepRobot(robot, path, nextIndex, dt)
  }

  const targetTheta = Math.atan2(dy, dx)
  const newTheta = lerpAngle(robot.theta, targetTheta, ANGULAR_SPEED * safeDt)

  const angleErr = Math.abs(wrapAngle(targetTheta - newTheta))
  const forwardScale = angleErr > FORWARD_GATE ? 0 : 1
  const forwardStep = Math.min(distance, LINEAR_SPEED * safeDt * forwardScale)

  const nextRobot: RobotPose = {
    x: robot.x + Math.cos(newTheta) * forwardStep,
    y: robot.y + Math.sin(newTheta) * forwardStep,
    theta: wrapAngle(newTheta),
  }

  return { robot: nextRobot, pathIndex: nextIndex, done: false }
}
