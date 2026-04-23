//////////////////////////////////////////////////////////////////////////////
//
// three-vrm-binding.ts
//
// - Bridges character ActionBus events onto a loaded VRM. The binding owns
//   both one-shot gestures and the continuous idle / facial behavior that
//   runs every frame.
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

interface ExpressionEnvelope {
  readonly name: VRMExpressionPresetName | string;
  readonly peak: number;
  readonly attackSeconds: number;
  readonly holdSeconds: number;
  readonly releaseSeconds: number;
  elapsedSeconds: number;
}

interface BoneMotion {
  readonly node: THREE.Object3D;
  readonly rest: THREE.Euler;
}

const EXPRESSION_DAMPING = 18;
const LOOK_TARGET_LERP = 8;
const BLINK_MIN_SECONDS = 2.2;
const BLINK_MAX_SECONDS = 5.1;
const DOUBLE_BLINK_CHANCE = 0.16;
const TALKING_MOUTH_DAMPING = 14;
const GAZE_ACTION_SECONDS = 1.4;

const TRANSIENT_EXPRESSIONS: Partial<Record<string, ExpressionEnvelopeSpec>> = {
  smile: { name: VRMExpressionPresetName.Happy, peak: 0.82, attackSeconds: 0.14, holdSeconds: 1.9, releaseSeconds: 0.45 },
  look_sad: { name: VRMExpressionPresetName.Sad, peak: 0.7, attackSeconds: 0.18, holdSeconds: 2.3, releaseSeconds: 0.5 },
  gasp: { name: VRMExpressionPresetName.Surprised, peak: 0.88, attackSeconds: 0.08, holdSeconds: 0.85, releaseSeconds: 0.32 },
  look_angry: { name: VRMExpressionPresetName.Angry, peak: 0.72, attackSeconds: 0.12, holdSeconds: 1.9, releaseSeconds: 0.45 },
};

interface ExpressionEnvelopeSpec {
  readonly name: VRMExpressionPresetName | string;
  readonly peak: number;
  readonly attackSeconds: number;
  readonly holdSeconds: number;
  readonly releaseSeconds: number;
}

const BASE_MOOD_EXPRESSIONS = [
  VRMExpressionPresetName.Happy,
  VRMExpressionPresetName.Sad,
  VRMExpressionPresetName.Surprised,
  VRMExpressionPresetName.Angry,
  VRMExpressionPresetName.Relaxed,
] as const;

const MOOD_TO_EXPRESSION: Record<string, (typeof BASE_MOOD_EXPRESSIONS)[number] | null> = {
  happy: VRMExpressionPresetName.Happy,
  sad: VRMExpressionPresetName.Sad,
  surprised: VRMExpressionPresetName.Surprised,
  angry: VRMExpressionPresetName.Angry,
  neutral: null,
};

const TALKING_MOUTH_EXPRESSIONS = [
  VRMExpressionPresetName.Aa,
  VRMExpressionPresetName.Ih,
  VRMExpressionPresetName.Ou,
  VRMExpressionPresetName.Ee,
  VRMExpressionPresetName.Oh,
] as const;

export class ThreeVRMBinding {
  private readonly bus: ActionBus;
  private readonly avatar: LoadedAvatar;
  private readonly disposers: Array<() => void> = [];
  private readonly expressionValues = new Map<VRMExpressionPresetName | string, number>();
  private readonly lookTarget = new THREE.Object3D();
  private readonly desiredLookTarget = new THREE.Vector3();
  private readonly tempVec = new THREE.Vector3();
  private readonly headWorldPos = new THREE.Vector3();
  private readonly gazeAnchor = new THREE.Vector3();
  private readonly baseFocus: THREE.Vector3;
  private readonly headMotion: BoneMotion | null;
  private readonly neckMotion: BoneMotion | null;
  private readonly chestMotion: BoneMotion | null;
  private active: ActiveAnimation | null = null;
  private activeMood: (typeof BASE_MOOD_EXPRESSIONS)[number] | null = null;
  private transientExpressions: ExpressionEnvelope[] = [];
  private speaking = false;
  private elapsedSeconds = 0;
  private blinkTimer = randomRange(BLINK_MIN_SECONDS, BLINK_MAX_SECONDS);
  private blinkExpression: ExpressionEnvelope | null = null;
  private gazeOverrideSeconds = 0;
  private readonly gazeOffset = new THREE.Vector3(0, 0, 1.35);

  public constructor(bus: ActionBus, avatar: LoadedAvatar) {
    this.bus = bus;
    this.avatar = avatar;
    this.baseFocus = avatar.layout.focusPoint.clone();
    this.headMotion = this.getBoneMotion(VRMHumanBoneName.Head);
    this.neckMotion = this.getBoneMotion(VRMHumanBoneName.Neck);
    this.chestMotion =
      this.getBoneMotion(VRMHumanBoneName.UpperChest) ??
      this.getBoneMotion(VRMHumanBoneName.Chest) ??
      this.getBoneMotion(VRMHumanBoneName.Spine);
    this.lookTarget.position.copy(this.baseFocus).add(this.gazeOffset);
    this.desiredLookTarget.copy(this.lookTarget.position);
    this.disposers.push(this.bus.on('action', (event) => this.handleAction(event)));

    if (this.avatar.vrm.lookAt) {
      this.avatar.vrm.lookAt.target = this.lookTarget;
    }
  }

  /** Per-frame update. Forward `deltaSeconds` from the scene loop. */
  public tick(deltaSeconds: number): void {
    this.elapsedSeconds += deltaSeconds;
    this.avatar.update(deltaSeconds);
    this.updateOneShotAnimation(deltaSeconds);
    this.updateTransientExpressions(deltaSeconds);
    this.updateBlink(deltaSeconds);
    this.updateIdlePose();
    this.updateLookAt(deltaSeconds);
    this.updateMouthExpressions(deltaSeconds);
    this.updateExpressionWeights(deltaSeconds);
  }

  public setSpeaking(active: boolean): void {
    this.speaking = active;
  }

  public dispose(): void {
    for (const disposer of this.disposers) {
      disposer();
    }
    if (this.active) {
      this.active.cleanup();
      this.active = null;
    }
    this.resetBoneMotion(this.chestMotion);
    this.resetBoneMotion(this.neckMotion);
    this.resetBoneMotion(this.headMotion);
    const expressionManager = this.avatar.vrm.expressionManager;
    if (expressionManager) {
      expressionManager.resetValues();
      expressionManager.update();
    }
    if (this.avatar.vrm.lookAt) {
      this.avatar.vrm.lookAt.target = null;
      this.avatar.vrm.lookAt.reset();
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
        this.startAnimation(this.buildNodAnimation(-1, true));
        return;
      case 'smile':
      case 'look_sad':
      case 'gasp':
      case 'look_angry':
        this.playTransientExpression(event.name);
        return;
      case 'settle':
        this.transientExpressions = [];
        this.setMood('neutral');
        return;
      case 'look_at_you':
        this.applyLookAt('camera');
        return;
      case 'glance_left':
        this.applyLookAt('left');
        return;
      case 'glance_right':
        this.applyLookAt('right');
        return;
      case 'look_up':
        this.applyLookAt('up');
        return;
      case 'look_down':
        this.applyLookAt('down');
        return;
      default:
        console.info(`[binding] no handler for action "${event.name}"`);
    }
  }

  private startAnimation(next: ActiveAnimation): void {
    if (this.active) {
      this.active.cleanup();
    }
    this.active = next;
  }

  private updateOneShotAnimation(deltaSeconds: number): void {
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

  private buildWaveAnimation(): ActiveAnimation {
    const target = this.avatar.vrm.humanoid?.getNormalizedBoneNode(VRMHumanBoneName.RightUpperArm) ?? null;
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
    const head = this.headMotion?.node ?? null;
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

  private setMood(mood: string): void {
    this.activeMood = MOOD_TO_EXPRESSION[mood] ?? null;
  }

  private playTransientExpression(actionName: string): void {
    const next = TRANSIENT_EXPRESSIONS[actionName];
    if (!next) {
      return;
    }
    this.transientExpressions.push({ ...next, elapsedSeconds: 0 });
  }

  private updateTransientExpressions(deltaSeconds: number): void {
    this.transientExpressions = this.transientExpressions.filter((expression) => {
      expression.elapsedSeconds += deltaSeconds;
      return expression.elapsedSeconds <
        expression.attackSeconds + expression.holdSeconds + expression.releaseSeconds;
    });
  }

  private updateBlink(deltaSeconds: number): void {
    if (this.blinkExpression) {
      this.blinkExpression.elapsedSeconds += deltaSeconds;
      const total =
        this.blinkExpression.attackSeconds +
        this.blinkExpression.holdSeconds +
        this.blinkExpression.releaseSeconds;
      if (this.blinkExpression.elapsedSeconds >= total) {
        this.blinkExpression = null;
      }
      return;
    }

    this.blinkTimer -= deltaSeconds;
    if (this.blinkTimer > 0) {
      return;
    }

    this.blinkExpression = {
      name: VRMExpressionPresetName.Blink,
      peak: 1,
      attackSeconds: 0.045,
      holdSeconds: 0.028,
      releaseSeconds: 0.06,
      elapsedSeconds: 0,
    };
    this.blinkTimer = randomRange(BLINK_MIN_SECONDS, BLINK_MAX_SECONDS);
    if (Math.random() < DOUBLE_BLINK_CHANCE) {
      this.blinkTimer = 0.12 + Math.random() * 0.1;
    }
  }

  private updateIdlePose(): void {
    const talkFactor = this.speaking ? 1 : 0;
    const swaySlow = Math.sin(this.elapsedSeconds * 0.72);
    const swayFast = Math.sin(this.elapsedSeconds * 1.47 + 0.9);
    const breathe = Math.sin(this.elapsedSeconds * 1.1 + 0.25);
    const micro = Math.sin(this.elapsedSeconds * 2.3 + 1.8);

    if (this.chestMotion) {
      const rest = this.chestMotion.rest;
      this.chestMotion.node.rotation.set(
        rest.x + breathe * 0.018 + swayFast * 0.004,
        rest.y + swaySlow * 0.02,
        rest.z + swayFast * 0.014
      );
    }

    if (this.neckMotion) {
      const rest = this.neckMotion.rest;
      this.neckMotion.node.rotation.set(
        rest.x + breathe * 0.012 + micro * 0.01 + talkFactor * 0.01 * Math.sin(this.elapsedSeconds * 3.4),
        rest.y + swaySlow * 0.03 + talkFactor * 0.012 * Math.sin(this.elapsedSeconds * 4.2 + 0.8),
        rest.z + swayFast * 0.012
      );
    }

    if (this.headMotion && !this.active) {
      const rest = this.headMotion.rest;
      this.headMotion.node.rotation.set(
        rest.x + swayFast * 0.018 + micro * 0.01 + talkFactor * 0.018 * Math.sin(this.elapsedSeconds * 5.1),
        rest.y + swaySlow * 0.045 + talkFactor * 0.02 * Math.sin(this.elapsedSeconds * 4.5 + 0.35),
        rest.z + swayFast * 0.016 + micro * 0.008
      );
    }
  }

  private updateLookAt(deltaSeconds: number): void {
    const vrm = this.avatar.vrm;
    if (!vrm.lookAt) {
      return;
    }

    if (this.gazeOverrideSeconds > 0) {
      this.gazeOverrideSeconds = Math.max(0, this.gazeOverrideSeconds - deltaSeconds);
    }

    const headNode = this.headMotion?.node ?? vrm.humanoid?.getNormalizedBoneNode(VRMHumanBoneName.Head) ?? vrm.scene;
    headNode.getWorldPosition(this.headWorldPos);
    this.gazeAnchor.copy(this.baseFocus).setZ(this.headWorldPos.z + 1.35);

    if (this.gazeOverrideSeconds === 0) {
      const orbitX = Math.sin(this.elapsedSeconds * 0.4) * (this.speaking ? 0.06 : 0.03);
      const orbitY = Math.sin(this.elapsedSeconds * 0.63 + 0.7) * (this.speaking ? 0.045 : 0.022);
      this.desiredLookTarget.copy(this.gazeAnchor);
      this.desiredLookTarget.x += orbitX;
      this.desiredLookTarget.y += orbitY;
    }

    this.lookTarget.position.lerp(this.desiredLookTarget, 1 - Math.exp(-LOOK_TARGET_LERP * deltaSeconds));
    this.lookTarget.updateMatrixWorld();
  }

  private updateMouthExpressions(deltaSeconds: number): void {
    const mouthOpen = this.speaking
      ? 0.18 + 0.14 * (Math.sin(this.elapsedSeconds * 12.5) * 0.5 + 0.5) + 0.08 * (Math.sin(this.elapsedSeconds * 18.7 + 0.8) * 0.5 + 0.5)
      : 0;
    const mouthTargets: Record<(typeof TALKING_MOUTH_EXPRESSIONS)[number], number> = {
      aa: mouthOpen * (0.6 + 0.4 * (Math.sin(this.elapsedSeconds * 9.7) * 0.5 + 0.5)),
      ih: mouthOpen * (0.18 + 0.18 * (Math.sin(this.elapsedSeconds * 13.1 + 1.2) * 0.5 + 0.5)),
      ou: mouthOpen * (0.22 + 0.16 * (Math.sin(this.elapsedSeconds * 8.4 + 0.4) * 0.5 + 0.5)),
      ee: mouthOpen * (0.12 + 0.12 * (Math.sin(this.elapsedSeconds * 14.4 + 2.1) * 0.5 + 0.5)),
      oh: mouthOpen * (0.16 + 0.18 * (Math.sin(this.elapsedSeconds * 11.3 + 0.2) * 0.5 + 0.5)),
    };

    for (const name of TALKING_MOUTH_EXPRESSIONS) {
      const current = this.expressionValues.get(name) ?? 0;
      const next = THREE.MathUtils.damp(current, mouthTargets[name], TALKING_MOUTH_DAMPING, deltaSeconds);
      this.expressionValues.set(name, next);
    }
  }

  private updateExpressionWeights(deltaSeconds: number): void {
    const expressionManager = this.avatar.vrm.expressionManager;
    if (!expressionManager) {
      return;
    }

    const targets = new Map<VRMExpressionPresetName | string, number>();
    for (const name of BASE_MOOD_EXPRESSIONS) {
      targets.set(name, this.activeMood === name ? 0.35 : 0);
    }
    for (const transient of this.transientExpressions) {
      targets.set(transient.name, Math.max(targets.get(transient.name) ?? 0, getEnvelopeValue(transient)));
    }
    if (this.blinkExpression) {
      targets.set(this.blinkExpression.name, getEnvelopeValue(this.blinkExpression));
    }
    for (const name of TALKING_MOUTH_EXPRESSIONS) {
      targets.set(name, Math.max(targets.get(name) ?? 0, this.expressionValues.get(name) ?? 0));
    }

    const names = new Set<VRMExpressionPresetName | string>([
      ...targets.keys(),
      ...this.expressionValues.keys(),
      ...BASE_MOOD_EXPRESSIONS,
      VRMExpressionPresetName.Blink,
    ]);

    for (const name of names) {
      const current = this.expressionValues.get(name) ?? expressionManager.getValue(name) ?? 0;
      const target = targets.get(name) ?? 0;
      const next = THREE.MathUtils.damp(current, target, EXPRESSION_DAMPING, deltaSeconds);
      this.expressionValues.set(name, next);
      expressionManager.setValue(name, next);
    }

    expressionManager.update();
  }

  private applyLookAt(target: 'camera' | 'left' | 'right' | 'up' | 'down'): void {
    const headNode = this.headMotion?.node ?? this.avatar.vrm.humanoid?.getNormalizedBoneNode(VRMHumanBoneName.Head) ?? this.avatar.root;
    headNode.getWorldPosition(this.headWorldPos);
    const offset = this.tempVec.set(0, 0, 1.35);
    switch (target) {
      case 'left':
        offset.x = -0.38;
        offset.y = 0.02;
        break;
      case 'right':
        offset.x = 0.38;
        offset.y = 0.02;
        break;
      case 'up':
        offset.x = 0;
        offset.y = 0.34;
        break;
      case 'down':
        offset.x = 0;
        offset.y = -0.24;
        break;
      case 'camera':
      default:
        offset.x = 0;
        offset.y = 0;
        break;
    }
    this.desiredLookTarget.copy(this.headWorldPos).add(offset);
    this.gazeOverrideSeconds = GAZE_ACTION_SECONDS;
  }

  private getBoneMotion(humanBoneName: VRMHumanBoneName): BoneMotion | null {
    const node = this.avatar.vrm.humanoid?.getNormalizedBoneNode(humanBoneName) ?? null;
    if (!node) {
      return null;
    }
    return {
      node,
      rest: node.rotation.clone(),
    };
  }

  private resetBoneMotion(motion: BoneMotion | null): void {
    if (!motion) {
      return;
    }
    motion.node.rotation.copy(motion.rest);
  }
}

function getEnvelopeValue(envelope: ExpressionEnvelope): number {
  const attackEnd = envelope.attackSeconds;
  const holdEnd = attackEnd + envelope.holdSeconds;
  const releaseEnd = holdEnd + envelope.releaseSeconds;
  const elapsed = envelope.elapsedSeconds;

  if (elapsed <= attackEnd) {
    return envelope.peak * easeOutCubic(safeDivide(elapsed, envelope.attackSeconds));
  }
  if (elapsed <= holdEnd) {
    return envelope.peak;
  }
  if (elapsed <= releaseEnd) {
    return envelope.peak * (1 - easeInOutCubic(safeDivide(elapsed - holdEnd, envelope.releaseSeconds)));
  }
  return 0;
}

function easeOutCubic(t: number): number {
  return 1 - Math.pow(1 - t, 3);
}

function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

function randomRange(min: number, max: number): number {
  return min + Math.random() * (max - min);
}

function safeDivide(value: number, by: number): number {
  if (by <= 0) {
    return 1;
  }
  return THREE.MathUtils.clamp(value / by, 0, 1);
}
