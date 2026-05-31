/**
 * Kinematics module — diff-drive style motion of the robot along a planned path.
 *
 * COORDINATE CONVENTIONS (load-bearing — the #1 source of bugs):
 *
 *   World frame:
 *     x = east, y = north, origin at LOWER-LEFT of the map.
 *     theta in radians, 0 = east (+x), +pi/2 = north (+y), CCW positive.
 *
 *   three.js frame (used by Scene3D):
 *     x = right, y = UP (wall height / vertical), z = forward (= our world y).
 *     Mapping:  (world.x, 0, world.y)  ->  (three.x, three.y, three.z).
 *     Heading:  rotation.y = -theta    (sign flip; +theta CCW around world Y
 *                                        is -rotation around three's Y).
 *
 *   Grid cells:
 *     arr[row * width + col], row 0 = TOP, col 0 = LEFT.
 *     cell (col, row) -> world center ((col + 0.5) * c, (H - 1 - row + 0.5) * c).
 *     (See utils/grid.ts and backend services/planner.py for the canonical map.)
 *
 * dt CLAMP (also load-bearing):
 *   useFrame / requestAnimationFrame can deliver dt > 0.1s after a tab pause,
 *   teleporting the robot. Always clamp dt <= DT_MAX in the simulator.
 */
import type { Point2D, RobotPose } from '../types'

export const LINEAR_SPEED = 0.5
export const ANGULAR_SPEED = 1.5
export const ARRIVAL_THRESHOLD = 0.05
export const ANGLE_THRESHOLD = 0.05
export const DT_MAX = 0.05

// Threshold beyond which we suppress forward motion to avoid sliding-while-spinning.
// At ~0.7 rad (~40°) the angle error is dominant; rotate first, then translate.
const FORWARD_GATE = 0.7

/** Wrap an angle to (-π, π]. Robust against JS's sign-preserving modulo
 * (`-1 % 5 === -1`) which otherwise breaks naive `((a + π) % 2π) - π`. */
export function wrapAngle(a: number): number {
  let n = ((a + Math.PI) % (2 * Math.PI)) - Math.PI
  if (n <= -Math.PI) n += 2 * Math.PI
  return n
}

/** Shortest signed angular delta on the unit circle, clamped to ±maxDelta. */
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

  // Use wrapAngle (not naive `% 2π`) — JS's signed modulo used to make this
  // huge when the heading drifted past ±π, gating forward motion forever.
  const angleErr = Math.abs(wrapAngle(targetTheta - newTheta))
  // Gate forward motion until the heading is close enough — feels natural and
  // matches a diff-drive robot that turns in place before driving.
  const forwardScale = angleErr > FORWARD_GATE ? 0 : 1
  const forwardStep = Math.min(distance, LINEAR_SPEED * safeDt * forwardScale)

  const nextRobot: RobotPose = {
    x: robot.x + Math.cos(newTheta) * forwardStep,
    y: robot.y + Math.sin(newTheta) * forwardStep,
    // Keep theta canonical in (-π, π] so downstream math never drifts past
    // the wrap point and reawakens the broken-modulo bug.
    theta: wrapAngle(newTheta),
  }

  return { robot: nextRobot, pathIndex: nextIndex, done: false }
}
