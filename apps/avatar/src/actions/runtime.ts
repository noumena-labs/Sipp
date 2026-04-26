import type { ExpressionActionName } from './expressions';
import type { GazeTarget } from './gaze';
import type { ClipActionName } from './mixamo';

export interface AvatarActionRuntime {
  playClip(name: ClipActionName): void;
  playTransientExpression(name: ExpressionActionName): void;
  settle(): void;
  applyLookAt(target: GazeTarget): void;
}
