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

export function resolveGazeOffset(target: GazeTarget, offset: THREE.Vector3): THREE.Vector3 {
  offset.set(0, 0, 1.35);
  switch (target) {
    case 'left':
      offset.x = -0.38;
      offset.y = 0.02;
      return offset;
    case 'right':
      offset.x = 0.38;
      offset.y = 0.02;
      return offset;
    case 'up':
      offset.y = 0.34;
      return offset;
    case 'down':
      offset.y = -0.24;
      return offset;
    case 'camera':
    default:
      return offset;
  }
}
