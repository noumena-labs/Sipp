//////////////////////////////////////////////////////////////////////////////
//
// three-vrm-binding.ts
//
// - Bridges character ActionBus events onto a loaded VRM. Body gestures are
//   loaded from Mixamo .fbx clips by action name; facial and gaze behaviors
//   remain code-driven.
//
//////////////////////////////////////////////////////////////////////////////

import * as THREE from 'three';
import { VRMExpressionPresetName, VRMHumanBoneName } from '@pixiv/three-vrm';
import type { ActionEvent } from '@noumena-labs/cogent-engine/character';
import { ActionBus } from '@noumena-labs/cogent-engine/character';
import {
  dispatchAvatarAction,
  getRequiredClipActions,
} from '../actions';
import {
  BASE_MOOD_EXPRESSIONS,
  MOOD_TO_EXPRESSION,
  TALKING_MOUTH_EXPRESSIONS,
  TRANSIENT_EXPRESSIONS,
  type ExpressionActionName,
  type ExpressionEnvelope,
} from '../actions/expressions';
import { resolveGazeOffset, type GazeTarget } from '../actions/gaze';
import {
  loadMixamoAnimationClip,
  type ClipActionName,
} from '../actions/mixamo';
import type { AvatarActionRuntime } from '../actions/runtime';
import {
  resolveActionClipUrl,
  type AvatarRenderAssets,
} from '../characters/render-assets';
import type { LoadedAvatar } from '../scene/vrm-loader';

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

const CLIP_FADE_SECONDS = 0.18;
const CLIP_STOP_DELAY_MS = Math.ceil(CLIP_FADE_SECONDS * 1000);

export class ThreeVRMBinding implements AvatarActionRuntime {
  private readonly bus: ActionBus;
  private readonly avatar: LoadedAvatar;
  private readonly renderAssets: AvatarRenderAssets;
  private readonly disposers: Array<() => void> = [];
  private readonly expressionValues = new Map<VRMExpressionPresetName | string, number>();
  private readonly clipActions = new Map<ClipActionName, THREE.AnimationAction>();
  private readonly lookTarget = new THREE.Object3D();
  private readonly desiredLookTarget = new THREE.Vector3();
  private readonly tempVec = new THREE.Vector3();
  private readonly headWorldPos = new THREE.Vector3();
  private readonly gazeAnchor = new THREE.Vector3();
  private readonly baseFocus: THREE.Vector3;
  private readonly headMotion: BoneMotion | null;
  private readonly neckMotion: BoneMotion | null;
  private readonly chestMotion: BoneMotion | null;
  private readonly mixer: THREE.AnimationMixer;
  private activeMood: (typeof BASE_MOOD_EXPRESSIONS)[number] | null = null;
  private transientExpressions: ExpressionEnvelope[] = [];
  private speaking = false;
  private elapsedSeconds = 0;
  private blinkTimer = randomRange(BLINK_MIN_SECONDS, BLINK_MAX_SECONDS);
  private blinkExpression: ExpressionEnvelope | null = null;
  private gazeOverrideSeconds = 0;
  private readonly gazeOffset = new THREE.Vector3(0, 0, 1.35);
  private currentClipAction: THREE.AnimationAction | null = null;
  private idleAction: THREE.AnimationAction | null = null;
  private readonly pendingClipStops = new Map<
    THREE.AnimationAction,
    ReturnType<typeof window.setTimeout>
  >();

  public constructor(bus: ActionBus, avatar: LoadedAvatar, renderAssets: AvatarRenderAssets) {
    this.bus = bus;
    this.avatar = avatar;
    this.renderAssets = renderAssets;
    this.mixer = new THREE.AnimationMixer(this.avatar.vrm.scene);
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
    this.mixer.addEventListener('finished', this.handleMixerFinished);

    if (this.avatar.vrm.lookAt) {
      this.avatar.vrm.lookAt.target = this.lookTarget;
    }
  }

  public async init(actionNames: readonly string[]): Promise<void> {
    const clipActions = getRequiredClipActions(actionNames);
    await Promise.all([
      this.preloadIdleAction(),
      ...clipActions.map((actionName) => this.preloadClipAction(actionName)),
    ]);
    this.playIdle();
  }

  /** Per-frame update. Forward `deltaSeconds` from the scene loop. */
  public tick(deltaSeconds: number): void {
    this.elapsedSeconds += deltaSeconds;
    this.mixer.update(deltaSeconds);
    this.updateTransientExpressions(deltaSeconds);
    this.updateBlink(deltaSeconds);
    this.updateIdlePose();
    this.updateLookAt(deltaSeconds);
    this.updateMouthExpressions(deltaSeconds);
    this.updateExpressionWeights(deltaSeconds);
    this.avatar.update(deltaSeconds);
  }

  public setSpeaking(active: boolean): void {
    this.speaking = active;
  }

  public dispose(): void {
    for (const timeoutId of this.pendingClipStops.values()) {
      window.clearTimeout(timeoutId);
    }
    this.pendingClipStops.clear();
    for (const disposer of this.disposers) {
      disposer();
    }
    this.mixer.removeEventListener('finished', this.handleMixerFinished);
    this.idleAction?.stop();
    for (const action of this.clipActions.values()) {
      action.stop();
    }
    this.clipActions.clear();
    this.idleAction = null;
    this.currentClipAction = null;
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

  public playClip(name: ClipActionName): void {
    const next = this.clipActions.get(name);
    if (!next) {
      throw new Error(`Clip action \"${name}\" was triggered before it finished loading.`);
    }

    this.cancelScheduledClipStop(next);
    const previousAction = this.currentClipAction;

    next.reset();
    next.enabled = true;
    next.setLoop(THREE.LoopOnce, 1);
    next.clampWhenFinished = true;
    next.play();

    if (previousAction && previousAction !== next) {
      previousAction.crossFadeTo(next, CLIP_FADE_SECONDS, false);
      this.scheduleClipStop(previousAction);
    } else if (this.idleAction) {
      this.idleAction.crossFadeTo(next, CLIP_FADE_SECONDS, false);
    } else {
      next.fadeIn(CLIP_FADE_SECONDS);
    }

    this.currentClipAction = next;
  }

  public playTransientExpression(actionName: ExpressionActionName): void {
    const next = TRANSIENT_EXPRESSIONS[actionName];
    this.transientExpressions.push({ ...next, elapsedSeconds: 0 });
  }

  public settle(): void {
    this.transientExpressions = [];
    this.setMood('neutral');
  }

  public applyLookAt(target: GazeTarget): void {
    const headNode =
      this.headMotion?.node ??
      this.avatar.vrm.humanoid?.getNormalizedBoneNode(VRMHumanBoneName.Head) ??
      this.avatar.root;
    headNode.getWorldPosition(this.headWorldPos);
    resolveGazeOffset(target, this.tempVec);
    this.desiredLookTarget.copy(this.headWorldPos).add(this.tempVec);
    this.gazeOverrideSeconds = GAZE_ACTION_SECONDS;
  }

  private handleAction(event: ActionEvent): void {
    if (!dispatchAvatarAction(event.name, this)) {
      console.info(`[binding] no handler for action "${event.name}"`);
    }
  }

  private setMood(mood: keyof typeof MOOD_TO_EXPRESSION): void {
    this.activeMood = MOOD_TO_EXPRESSION[mood] ?? null;
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
    if (this.idleAction) {
      return;
    }

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

    if (this.headMotion) {
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

  private readonly handleMixerFinished = (event: { action: THREE.AnimationAction }): void => {
    if (event.action !== this.currentClipAction) {
      event.action.stop();
      return;
    }

    if (this.idleAction) {
      this.playIdle();
      event.action.crossFadeTo(this.idleAction, CLIP_FADE_SECONDS, false);
    } else {
      event.action.fadeOut(CLIP_FADE_SECONDS);
    }

    this.scheduleClipStop(event.action);
    this.currentClipAction = null;
  };

  private async preloadIdleAction(): Promise<void> {
    const clip = await loadMixamoAnimationClip(this.renderAssets.idleUrl, this.avatar.vrm);
    const action = this.mixer.clipAction(clip);
    action.enabled = true;
    action.clampWhenFinished = false;
    action.setLoop(THREE.LoopRepeat, Infinity);
    this.idleAction = action;
  }

  private async preloadClipAction(actionName: ClipActionName): Promise<void> {
    const clipUrl = resolveActionClipUrl(this.renderAssets, actionName);
    const clip = await loadMixamoAnimationClip(clipUrl, this.avatar.vrm);
    const action = this.mixer.clipAction(clip);
    action.enabled = false;
    action.clampWhenFinished = false;
    action.setLoop(THREE.LoopOnce, 1);
    this.clipActions.set(actionName, action);
  }

  private playIdle(): void {
    if (!this.idleAction) {
      return;
    }

    this.cancelScheduledClipStop(this.idleAction);
    this.idleAction.enabled = true;
    this.idleAction.reset();
    this.idleAction.setLoop(THREE.LoopRepeat, Infinity);
    this.idleAction.clampWhenFinished = false;
    this.idleAction.play();
  }

  private scheduleClipStop(action: THREE.AnimationAction): void {
    this.cancelScheduledClipStop(action);
    const timeoutId = window.setTimeout(() => {
      action.stop();
      this.pendingClipStops.delete(action);
    }, CLIP_STOP_DELAY_MS);
    this.pendingClipStops.set(action, timeoutId);
  }

  private cancelScheduledClipStop(action: THREE.AnimationAction): void {
    const timeoutId = this.pendingClipStops.get(action);
    if (timeoutId == null) {
      return;
    }
    window.clearTimeout(timeoutId);
    this.pendingClipStops.delete(action);
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
