import type { AvatarActionRuntime } from './runtime';
import { GAZE_ACTION_TARGETS, type GazeActionName } from './gaze';
import { isClipActionName, type ClipActionName } from './mixamo';
import {
  TRANSIENT_EXPRESSION_ACTIONS,
  type ExpressionActionName,
} from './expressions';

export type SupportedAvatarActionName =
  | ClipActionName
  | ExpressionActionName
  | GazeActionName
  | 'settle';

interface AvatarActionDefinition {
  readonly requiresClip: boolean;
  execute(runtime: AvatarActionRuntime): void;
}

const AVATAR_ACTIONS: Record<SupportedAvatarActionName, AvatarActionDefinition> = {
  wave: {
    requiresClip: true,
    execute(runtime) {
      runtime.playClip('wave');
    },
  },
  nod: {
    requiresClip: true,
    execute(runtime) {
      runtime.playClip('nod');
    },
  },
  shake_head: {
    requiresClip: true,
    execute(runtime) {
      runtime.playClip('shake_head');
    },
  },
  smile: {
    requiresClip: false,
    execute(runtime) {
      runtime.playTransientExpression('smile');
    },
  },
  look_sad: {
    requiresClip: false,
    execute(runtime) {
      runtime.playTransientExpression('look_sad');
    },
  },
  gasp: {
    requiresClip: false,
    execute(runtime) {
      runtime.playTransientExpression('gasp');
    },
  },
  look_angry: {
    requiresClip: false,
    execute(runtime) {
      runtime.playTransientExpression('look_angry');
    },
  },
  settle: {
    requiresClip: false,
    execute(runtime) {
      runtime.settle();
    },
  },
  look_at_you: {
    requiresClip: false,
    execute(runtime) {
      runtime.applyLookAt(GAZE_ACTION_TARGETS.look_at_you);
    },
  },
  glance_left: {
    requiresClip: false,
    execute(runtime) {
      runtime.applyLookAt(GAZE_ACTION_TARGETS.glance_left);
    },
  },
  glance_right: {
    requiresClip: false,
    execute(runtime) {
      runtime.applyLookAt(GAZE_ACTION_TARGETS.glance_right);
    },
  },
  look_up: {
    requiresClip: false,
    execute(runtime) {
      runtime.applyLookAt(GAZE_ACTION_TARGETS.look_up);
    },
  },
  look_down: {
    requiresClip: false,
    execute(runtime) {
      runtime.applyLookAt(GAZE_ACTION_TARGETS.look_down);
    },
  },
};

export function dispatchAvatarAction(name: string, runtime: AvatarActionRuntime): boolean {
  const action = AVATAR_ACTIONS[name as SupportedAvatarActionName];
  if (!action) {
    return false;
  }
  action.execute(runtime);
  return true;
}

export function getUnsupportedAvatarActions(names: readonly string[]): readonly string[] {
  return names.filter((name) => AVATAR_ACTIONS[name as SupportedAvatarActionName] == null);
}

export function getRequiredClipActions(names: readonly string[]): readonly ClipActionName[] {
  return names.filter((name): name is ClipActionName => {
    const action = AVATAR_ACTIONS[name as SupportedAvatarActionName];
    return action?.requiresClip === true && isClipActionName(name);
  });
}

export const SUPPORTED_EXPRESSION_ACTIONS = TRANSIENT_EXPRESSION_ACTIONS;
