import * as THREE from 'three';

export const GAZE_ACTION_TARGETS = {
  look_at_you: 'camera',
  glance_left: 'left',
  glance_right: 'right',
  look_up: 'up',
  look_down: 'down',
} as const;

export type GazeActionName = keyof typeof GAZE_ACTION_TARGETS;
export type GazeTarget = (typeof GAZE_ACTION_TARGETS)[GazeActionName];

export interface GazePose {
  readonly headPitch: number;
  readonly headYaw: number;
  readonly headRoll: number;
  readonly neckPitch: number;
  readonly neckYaw: number;
  readonly neckRoll: number;
  readonly chestPitch: number;
  readonly chestYaw: number;
  readonly chestRoll: number;
}

export const NEUTRAL_GAZE_POSE: GazePose = {
  headPitch: 0,
  headYaw: 0,
  headRoll: 0,
  neckPitch: 0,
  neckYaw: 0,
  neckRoll: 0,
  chestPitch: 0,
  chestYaw: 0,
  chestRoll: 0,
};

export function resolveGazeOffset(target: GazeTarget, offset: THREE.Vector3): THREE.Vector3 {
  offset.set(0, 0, 1.05);
  switch (target) {
    case 'left':
      offset.x = -0.92;
      offset.y = 0.02;
      return offset;
    case 'right':
      offset.x = 0.92;
      offset.y = 0.02;
      return offset;
    case 'up':
      offset.y = 0.72;
      return offset;
    case 'down':
      offset.y = -0.52;
      return offset;
    case 'camera':
    default:
      return offset;
  }
}

export function resolveGazePose(target: GazeTarget): GazePose {
  switch (target) {
    case 'left':
      return {
        headPitch: 0.015,
        headYaw: -0.26,
        headRoll: 0.035,
        neckPitch: 0.01,
        neckYaw: -0.14,
        neckRoll: 0.018,
        chestPitch: 0,
        chestYaw: -0.045,
        chestRoll: 0,
      };
    case 'right':
      return {
        headPitch: 0.015,
        headYaw: 0.26,
        headRoll: -0.035,
        neckPitch: 0.01,
        neckYaw: 0.14,
        neckRoll: -0.018,
        chestPitch: 0,
        chestYaw: 0.045,
        chestRoll: 0,
      };
    case 'up':
      return {
        headPitch: 0.21,
        headYaw: 0,
        headRoll: 0,
        neckPitch: 0.11,
        neckYaw: 0,
        neckRoll: 0,
        chestPitch: 0.028,
        chestYaw: 0,
        chestRoll: 0,
      };
    case 'down':
      return {
        headPitch: -0.18,
        headYaw: 0.035,
        headRoll: -0.012,
        neckPitch: -0.09,
        neckYaw: 0.018,
        neckRoll: 0,
        chestPitch: -0.024,
        chestYaw: 0,
        chestRoll: 0,
      };
    case 'camera':
    default:
      return {
        headPitch: 0.015,
        headYaw: 0,
        headRoll: 0,
        neckPitch: 0.008,
        neckYaw: 0,
        neckRoll: 0,
        chestPitch: 0.006,
        chestYaw: 0,
        chestRoll: 0,
      };
  }
}
