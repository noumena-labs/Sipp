import { VRMExpressionPresetName } from '@pixiv/three-vrm';

export interface ExpressionEnvelope {
  readonly name: VRMExpressionPresetName | string;
  readonly peak: number;
  readonly attackSeconds: number;
  readonly holdSeconds: number;
  readonly releaseSeconds: number;
  elapsedSeconds: number;
}

export interface ExpressionEnvelopeSpec {
  readonly name: VRMExpressionPresetName | string;
  readonly peak: number;
  readonly attackSeconds: number;
  readonly holdSeconds: number;
  readonly releaseSeconds: number;
}

export const TRANSIENT_EXPRESSION_ACTIONS = [
  'smile',
  'look_sad',
  'gasp',
  'look_angry',
] as const;

export type ExpressionActionName = (typeof TRANSIENT_EXPRESSION_ACTIONS)[number];

export const TRANSIENT_EXPRESSIONS: Record<ExpressionActionName, ExpressionEnvelopeSpec> = {
  smile: {
    name: VRMExpressionPresetName.Happy,
    peak: 0.82,
    attackSeconds: 0.14,
    holdSeconds: 1.9,
    releaseSeconds: 0.45,
  },
  look_sad: {
    name: VRMExpressionPresetName.Sad,
    peak: 0.7,
    attackSeconds: 0.18,
    holdSeconds: 2.3,
    releaseSeconds: 0.5,
  },
  gasp: {
    name: VRMExpressionPresetName.Surprised,
    peak: 0.88,
    attackSeconds: 0.08,
    holdSeconds: 0.85,
    releaseSeconds: 0.32,
  },
  look_angry: {
    name: VRMExpressionPresetName.Angry,
    peak: 0.72,
    attackSeconds: 0.12,
    holdSeconds: 1.9,
    releaseSeconds: 0.45,
  },
};

export const BASE_MOOD_EXPRESSIONS = [
  VRMExpressionPresetName.Happy,
  VRMExpressionPresetName.Sad,
  VRMExpressionPresetName.Surprised,
  VRMExpressionPresetName.Angry,
  VRMExpressionPresetName.Relaxed,
] as const;

export type BaseMoodExpressionName = (typeof BASE_MOOD_EXPRESSIONS)[number];

export const MOOD_TO_EXPRESSION: Record<
  'happy' | 'sad' | 'surprised' | 'angry' | 'neutral',
  BaseMoodExpressionName | null
> = {
  happy: VRMExpressionPresetName.Happy,
  sad: VRMExpressionPresetName.Sad,
  surprised: VRMExpressionPresetName.Surprised,
  angry: VRMExpressionPresetName.Angry,
  neutral: null,
};

export const TALKING_MOUTH_EXPRESSIONS = [
  VRMExpressionPresetName.Aa,
  VRMExpressionPresetName.Ih,
  VRMExpressionPresetName.Ou,
  VRMExpressionPresetName.Ee,
  VRMExpressionPresetName.Oh,
] as const;
