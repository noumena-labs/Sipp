import type { ExpressionActionName } from './expressions';
import type { GazeTarget } from './gaze';
import type { ClipActionName } from './mixamo';
import type { WorldEffectActionName } from './world-effects';

export interface AvatarActionRuntime {
  playClip(name: ClipActionName): void;
  playTransientExpression(name: ExpressionActionName): void;
  playWorldEffect(name: WorldEffectActionName): void;
  settle(): void;
  applyLookAt(target: GazeTarget): void;
}
