//////////////////////////////////////////////////////////////////////////////
//
// three-vrm-binding.ts
//
// - Bridges character ActionBus events onto a loaded VRM (or primitive
//   fallback) avatar. The binding owns any in-flight animation state and
//   exposes a `tick(dt)` callback that the scene's animation loop drives.
//
// - Gesture implementations are intentionally tiny and procedural: the goal
//   is to show the end-to-end plumbing cleanly, not to ship a production
//   motion library. Swapping in VRMA clips is an exercise for the reader.
//
//////////////////////////////////////////////////////////////////////////////

import * as THREE from 'three';
import { VRMExpressionPresetName, VRMHumanBoneName } from '@pixiv/three-vrm';
import type { ActionEvent } from 'cogent-engine/character';
import { ActionBus } from 'cogent-engine/character';
import type { LoadedAvatar } from '../scene/vrm-loader';

interface ActiveAnimation {
  readonly name: string;
  readonly durationSeconds: number;
  elapsedSeconds: number;
  update(progress: number): void;
  cleanup(): void;
}

const MOOD_TO_EXPRESSION: Record<string, VRMExpressionPresetName> = {
  happy: VRMExpressionPresetName.Happy,
  sad: VRMExpressionPresetName.Sad,
  surprised: VRMExpressionPresetName.Surprised,
  angry: VRMExpressionPresetName.Angry,
  neutral: VRMExpressionPresetName.Neutral,
};

export class ThreeVRMBinding {
  private readonly bus: ActionBus;
  private readonly avatar: LoadedAvatar;
  private readonly disposers: Array<() => void> = [];
  private active: ActiveAnimation | null = null;
  private currentMoodKey: VRMExpressionPresetName = VRMExpressionPresetName.Neutral;

  public constructor(bus: ActionBus, avatar: LoadedAvatar) {
    this.bus = bus;
    this.avatar = avatar;
    this.disposers.push(this.bus.on('action', (event) => this.handleAction(event)));
  }

  /** Per-frame update. Forward `deltaSeconds` from the scene loop. */
  public tick(deltaSeconds: number): void {
    this.avatar.update(deltaSeconds);
    if (!this.active) {
      return;
    }
    this.active.elapsedSeconds += deltaSeconds;
    const progress = Math.min(1, this.active.elapsedSeconds / this.active.durationSeconds);
    this.active.update(progress);
    if (progress >= 1) {
      this.active.cleanup();
      this.active = null;
    }
  }

  public dispose(): void {
    for (const disposer of this.disposers) {
      disposer();
    }
    if (this.active) {
      this.active.cleanup();
      this.active = null;
    }
  }

  private handleAction(event: ActionEvent): void {
    switch (event.name) {
      case 'wave':
        this.startAnimation(this.buildWaveAnimation());
        return;
      case 'nod':
        this.startAnimation(this.buildNodAnimation(1));
        return;
      case 'shake_head':
        this.startAnimation(this.buildNodAnimation(-1, /*axisY=*/ true));
        return;
      case 'set_mood': {
        const mood = String(event.args.mood ?? '').toLowerCase();
        this.applyMood(mood);
        return;
      }
      case 'look_at': {
        const target = String(event.args.target ?? 'camera').toLowerCase();
        this.applyLookAt(target);
        return;
      }
      default:
        // Unknown action — log and ignore. Unknown actions are not a bug in
        // the harness; they mean the schema drifted from what the binding
        // knows how to render.
        console.info(`[binding] no handler for action "${event.name}"`, event.args);
    }
  }

  private startAnimation(next: ActiveAnimation): void {
    if (this.active) {
      this.active.cleanup();
    }
    this.active = next;
  }

  private buildWaveAnimation(): ActiveAnimation {
    const rightArm = this.avatar.vrm?.humanoid?.getNormalizedBoneNode(VRMHumanBoneName.RightUpperArm);
    const fallback = this.avatar.root.getObjectByName('armR');
    const target = rightArm ?? fallback ?? null;
    const initialRotation = target ? target.rotation.clone() : null;
    const durationSeconds = 1.4;
    return {
      name: 'wave',
      durationSeconds,
      elapsedSeconds: 0,
      update(progress: number) {
        if (!target || !initialRotation) {
          return;
        }
        // Raise the arm then oscillate.
        const raise = Math.sin(Math.min(1, progress * 2) * Math.PI * 0.5) * -1.4;
        const oscillate = progress > 0.3 ? Math.sin(progress * Math.PI * 6) * 0.3 : 0;
        target.rotation.set(
          initialRotation.x,
          initialRotation.y,
          initialRotation.z + raise + oscillate
        );
      },
      cleanup() {
        if (target && initialRotation) {
          target.rotation.copy(initialRotation);
        }
      },
    };
  }

  private buildNodAnimation(direction: number, axisY = false): ActiveAnimation {
    const head =
      this.avatar.vrm?.humanoid?.getNormalizedBoneNode(VRMHumanBoneName.Head) ??
      this.avatar.root.getObjectByName('head');
    const initialRotation = head ? head.rotation.clone() : null;
    const durationSeconds = 1.0;
    return {
      name: axisY ? 'shake_head' : 'nod',
      durationSeconds,
      elapsedSeconds: 0,
      update(progress: number) {
        if (!head || !initialRotation) {
          return;
        }
        const swing = Math.sin(progress * Math.PI * 2) * 0.3 * direction;
        if (axisY) {
          head.rotation.set(initialRotation.x, initialRotation.y + swing, initialRotation.z);
        } else {
          head.rotation.set(initialRotation.x + swing, initialRotation.y, initialRotation.z);
        }
      },
      cleanup() {
        if (head && initialRotation) {
          head.rotation.copy(initialRotation);
        }
      },
    };
  }

  private applyMood(mood: string): void {
    const vrm = this.avatar.vrm;
    if (!vrm || !vrm.expressionManager) {
      return;
    }
    const next = MOOD_TO_EXPRESSION[mood];
    if (!next) {
      return;
    }
    // Reset previous preset then apply new one.
    vrm.expressionManager.setValue(this.currentMoodKey, 0);
    vrm.expressionManager.setValue(next, 1);
    this.currentMoodKey = next;
  }

  private applyLookAt(target: string): void {
    const vrm = this.avatar.vrm;
    if (!vrm || !vrm.lookAt) {
      return;
    }
    const offset = new THREE.Vector3();
    switch (target) {
      case 'left':
        offset.set(-0.5, 0, 1);
        break;
      case 'right':
        offset.set(0.5, 0, 1);
        break;
      case 'up':
        offset.set(0, 0.5, 1);
        break;
      case 'down':
        offset.set(0, -0.5, 1);
        break;
      case 'camera':
      default:
        offset.set(0, 0, 1);
    }
    const headPos = new THREE.Vector3();
    const headNode =
      vrm.humanoid?.getNormalizedBoneNode(VRMHumanBoneName.Head) ?? vrm.scene;
    headNode.getWorldPosition(headPos);
    vrm.lookAt.target = makeLookTarget(headPos.clone().add(offset));
  }
}

function makeLookTarget(position: THREE.Vector3): THREE.Object3D {
  const obj = new THREE.Object3D();
  obj.position.copy(position);
  return obj;
}
