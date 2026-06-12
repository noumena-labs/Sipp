//////////////////////////////////////////////////////////////////////////////
//
// three-vrm-binding.ts
//
// - Bridges character event bus events onto a loaded VRM. Body gestures are
//   loaded from Mixamo .fbx clips by action id; facial and gaze behaviors
//   remain code-driven.
//
//////////////////////////////////////////////////////////////////////////////

import * as THREE from 'three';
import { VRMExpressionPresetName, VRMHumanBoneName } from '@pixiv/three-vrm';
import type { ActionEvent } from '@noumena-labs/sipp/character';
import { CharacterEventBus } from '@noumena-labs/sipp/character';
import {
  dispatchAvatarAction,
  getRequiredClipActions,
} from '../actions';
import {
  BASE_MOOD_EXPRESSIONS,
  CLIP_ACTION_EXPRESSIONS,
  MOOD_TO_EXPRESSION,
  TALKING_MOUTH_EXPRESSIONS,
  TRANSIENT_EXPRESSIONS,
  type ExpressionActionName,
  type ExpressionEnvelope,
  type ExpressionEnvelopeSpec,
  type ExpressionName,
} from '../actions/expressions';
import {
  NEUTRAL_GAZE_POSE,
  resolveGazeOffset,
  resolveGazePose,
  type GazePose,
  type GazeTarget,
} from '../actions/gaze';
import {
  loadMixamoAnimationClip,
  type ClipActionName,
} from '../actions/mixamo';
import type { AvatarActionRuntime } from '../actions/runtime';
import type { WorldEffectActionName } from '../actions/world-effects';
import {
  resolveActionClipUrl,
  type AvatarRenderAssets,
} from '../characters/render-assets';
import type { LoadedAvatar } from '../scene/vrm-loader';
import { FantasyWorldEffects } from '../scene/world-effects';

interface BoneMotion {
  readonly node: THREE.Object3D;
  readonly rest: THREE.Euler;
}

interface GazeEnvelope {
  readonly pose: GazePose;
  elapsedSeconds: number;
}

const EXPRESSION_DAMPING = 18;
const LOOK_TARGET_LERP = 8;
const GAZE_POSE_DAMPING = 12;
const BLINK_MIN_SECONDS = 2.2;
const BLINK_MAX_SECONDS = 5.1;
const DOUBLE_BLINK_CHANCE = 0.16;
const TALKING_MOUTH_DAMPING = 14;
const GAZE_ACTION_SECONDS = 2.2;
const GAZE_POSE_ATTACK_SECONDS = 0.18;
const GAZE_POSE_HOLD_SECONDS = 1.05;
const GAZE_POSE_RELEASE_SECONDS = GAZE_ACTION_SECONDS - GAZE_POSE_ATTACK_SECONDS - GAZE_POSE_HOLD_SECONDS;

const CLIP_FADE_SECONDS = 0.18;
const CLIP_STOP_DELAY_MS = Math.ceil(CLIP_FADE_SECONDS * 1000);

export class ThreeVRMBinding implements AvatarActionRuntime {
  private readonly bus: CharacterEventBus;
  private readonly camera: THREE.Camera;
  private readonly avatar: LoadedAvatar;
  private readonly renderAssets: AvatarRenderAssets;
  private readonly disposers: Array<() => void> = [];
  private readonly expressionValues = new Map<VRMExpressionPresetName | string, number>();
  private readonly clipActions = new Map<ClipActionName, THREE.AnimationAction>();
  private readonly lookTarget = new THREE.Object3D();
  private readonly desiredLookTarget = new THREE.Vector3();
  private readonly tempVec = new THREE.Vector3();
  private readonly headWorldPos = new THREE.Vector3();
  private readonly cameraWorldPos = new THREE.Vector3();
  private readonly gazeAnchor = new THREE.Vector3();
  private readonly currentGazePose = { ...NEUTRAL_GAZE_POSE };
  private readonly baseFocus: THREE.Vector3;
  private readonly headMotion: BoneMotion | null;
  private readonly neckMotion: BoneMotion | null;
  private readonly chestMotion: BoneMotion | null;
  private readonly mixer: THREE.AnimationMixer;
  private readonly worldEffects: FantasyWorldEffects;
  private activeMood: (typeof BASE_MOOD_EXPRESSIONS)[number] | null = null;
  private transientExpressions: ExpressionEnvelope[] = [];
  private speaking = false;
  private elapsedSeconds = 0;
  private blinkTimer = randomRange(BLINK_MIN_SECONDS, BLINK_MAX_SECONDS);
  private blinkExpression: ExpressionEnvelope | null = null;
  private gazeOverrideSeconds = 0;
  private gazeEnvelope: GazeEnvelope | null = null;
  private readonly gazeOffset = new THREE.Vector3(0, 0, 1.35);
  private currentClipAction: THREE.AnimationAction | null = null;
  private idleAction: THREE.AnimationAction | null = null;
  private readonly pendingClipStops = new Map<
    THREE.AnimationAction,
    ReturnType<typeof window.setTimeout>
  >();

  public constructor(
    bus: CharacterEventBus,
    scene: THREE.Scene,
    camera: THREE.Camera,
    avatar: LoadedAvatar,
    renderAssets: AvatarRenderAssets
  ) {
    this.bus = bus;
    this.camera = camera;
    this.avatar = avatar;
    this.renderAssets = renderAssets;
    this.mixer = new THREE.AnimationMixer(this.avatar.vrm.scene);
    this.worldEffects = new FantasyWorldEffects(scene, avatar);
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
    this.updateGazePose(deltaSeconds);
    this.updateMouthExpressions(deltaSeconds);
    this.updateExpressionWeights(deltaSeconds);
    this.worldEffects.tick(deltaSeconds);
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
    this.worldEffects.dispose();
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
    this.playClipExpression(name);
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
    this.pushTransientExpression(next);
  }

  public playWorldEffect(name: WorldEffectActionName): void {
    this.worldEffects.trigger(name);
  }

  public settle(): void {
    this.transientExpressions = [];
    this.gazeEnvelope = null;
    this.gazeOverrideSeconds = 0;
    this.setMood('neutral');
  }

  public applyLookAt(target: GazeTarget): void {
    const headNode =
      this.headMotion?.node ??
      this.avatar.vrm.humanoid?.getNormalizedBoneNode(VRMHumanBoneName.Head) ??
      this.avatar.root;
    headNode.getWorldPosition(this.headWorldPos);
    if (target === 'camera') {
      this.camera.getWorldPosition(this.cameraWorldPos);
      this.desiredLookTarget.copy(this.cameraWorldPos);
    } else {
      resolveGazeOffset(target, this.tempVec);
      this.desiredLookTarget.copy(this.headWorldPos).add(this.tempVec);
    }
    this.gazeEnvelope = {
      pose: resolveGazePose(target),
      elapsedSeconds: 0,
    };
    this.gazeOverrideSeconds = GAZE_ACTION_SECONDS;
  }

  private handleAction(event: ActionEvent): void {
    if (!dispatchAvatarAction(event.id, this)) {
      console.info(`[binding] no handler for action "${event.id}"`);
    }
  }

  private setMood(mood: keyof typeof MOOD_TO_EXPRESSION): void {
    this.activeMood = MOOD_TO_EXPRESSION[mood] ?? null;
  }

  private playClipExpression(actionName: ClipActionName): void {
    const expression = CLIP_ACTION_EXPRESSIONS[actionName];
    if (expression) {
      this.pushTransientExpression(expression);
    }
  }

  private pushTransientExpression(spec: ExpressionEnvelopeSpec): void {
    const name = this.resolveExpressionName(spec.name);
    if (!name) {
      return;
    }
    this.transientExpressions.push({
      name,
      peak: spec.peak,
      attackSeconds: spec.attackSeconds,
      holdSeconds: spec.holdSeconds,
      releaseSeconds: spec.releaseSeconds,
      elapsedSeconds: 0,
    });
  }

  private resolveExpressionName(candidate: ExpressionEnvelopeSpec['name']): ExpressionName | null {
    const names = Array.isArray(candidate) ? candidate : [candidate];
    const expressionManager = this.avatar.vrm.expressionManager;
    if (!expressionManager) {
      return names[0] ?? null;
    }
    for (const name of names) {
      if (expressionManager.getExpression(name) != null) {
        return name;
      }
    }
    return null;
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

  private updateGazePose(deltaSeconds: number): void {
    let influence = 0;
    let targetPose = NEUTRAL_GAZE_POSE;

    if (this.gazeEnvelope) {
      this.gazeEnvelope.elapsedSeconds += deltaSeconds;
      targetPose = this.gazeEnvelope.pose;
      influence = getEnvelopeInfluence(
        this.gazeEnvelope.elapsedSeconds,
        GAZE_POSE_ATTACK_SECONDS,
        GAZE_POSE_HOLD_SECONDS,
        GAZE_POSE_RELEASE_SECONDS
      );
      if (influence <= 0) {
        this.gazeEnvelope = null;
      }
    }

    this.dampGazePose(this.currentGazePose, targetPose, influence, deltaSeconds);
    this.applyGazePoseToBone(this.chestMotion, {
      x: this.currentGazePose.chestPitch,
      y: this.currentGazePose.chestYaw,
      z: this.currentGazePose.chestRoll,
    });
    this.applyGazePoseToBone(this.neckMotion, {
      x: this.currentGazePose.neckPitch,
      y: this.currentGazePose.neckYaw,
      z: this.currentGazePose.neckRoll,
    });
    this.applyGazePoseToBone(this.headMotion, {
      x: this.currentGazePose.headPitch,
      y: this.currentGazePose.headYaw,
      z: this.currentGazePose.headRoll,
    });
  }

  private dampGazePose(
    current: Record<keyof GazePose, number>,
    targetPose: GazePose,
    influence: number,
    deltaSeconds: number
  ): void {
    for (const key of Object.keys(current) as Array<keyof GazePose>) {
      current[key] = THREE.MathUtils.damp(
        current[key],
        targetPose[key] * influence,
        GAZE_POSE_DAMPING,
        deltaSeconds
      );
    }
  }

  private applyGazePoseToBone(
    motion: BoneMotion | null,
    rotation: { readonly x: number; readonly y: number; readonly z: number }
  ): void {
    if (!motion) {
      return;
    }
    motion.node.rotation.x += rotation.x;
    motion.node.rotation.y += rotation.y;
    motion.node.rotation.z += rotation.z;
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
  return envelope.peak * getEnvelopeInfluence(
    envelope.elapsedSeconds,
    envelope.attackSeconds,
    envelope.holdSeconds,
    envelope.releaseSeconds
  );
}

function getEnvelopeInfluence(
  elapsed: number,
  attackSeconds: number,
  holdSeconds: number,
  releaseSeconds: number
): number {
  const attackEnd = attackSeconds;
  const holdEnd = attackEnd + holdSeconds;
  const releaseEnd = holdEnd + releaseSeconds;

  if (elapsed <= attackEnd) {
    return easeOutCubic(safeDivide(elapsed, attackSeconds));
  }
  if (elapsed <= holdEnd) {
    return 1;
  }
  if (elapsed <= releaseEnd) {
    return 1 - easeInOutCubic(safeDivide(elapsed - holdEnd, releaseSeconds));
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
