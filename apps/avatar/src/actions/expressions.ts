import { VRMExpressionPresetName } from '@pixiv/three-vrm';

export type ExpressionName = VRMExpressionPresetName | string;
export type ExpressionNameCandidate = ExpressionName | readonly ExpressionName[];

export interface ExpressionEnvelope {
  readonly name: ExpressionName;
  readonly peak: number;
  readonly attackSeconds: number;
  readonly holdSeconds: number;
  readonly releaseSeconds: number;
  elapsedSeconds: number;
}

export interface ExpressionEnvelopeSpec {
  readonly name: ExpressionNameCandidate;
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
    name: [VRMExpressionPresetName.Surprised, 'Surprised'],
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

export const CLIP_ACTION_EXPRESSIONS = {
  wave: {
    name: VRMExpressionPresetName.Happy,
    peak: 0.45,
    attackSeconds: 0.18,
    holdSeconds: 1.2,
    releaseSeconds: 0.45,
  },
  salute: {
    name: VRMExpressionPresetName.Relaxed,
    peak: 0.32,
    attackSeconds: 0.16,
    holdSeconds: 1.1,
    releaseSeconds: 0.4,
  },
  nod: {
    name: VRMExpressionPresetName.Relaxed,
    peak: 0.28,
    attackSeconds: 0.12,
    holdSeconds: 0.85,
    releaseSeconds: 0.32,
  },
  shake_head: {
    name: VRMExpressionPresetName.Sad,
    peak: 0.36,
    attackSeconds: 0.12,
    holdSeconds: 1.0,
    releaseSeconds: 0.38,
  },
  thinking: {
    name: VRMExpressionPresetName.Relaxed,
    peak: 0.38,
    attackSeconds: 0.2,
    holdSeconds: 1.7,
    releaseSeconds: 0.45,
  },
  bashful: {
    name: VRMExpressionPresetName.Happy,
    peak: 0.62,
    attackSeconds: 0.14,
    holdSeconds: 1.6,
    releaseSeconds: 0.5,
  },
  excited: {
    name: VRMExpressionPresetName.Happy,
    peak: 0.78,
    attackSeconds: 0.1,
    holdSeconds: 1.4,
    releaseSeconds: 0.38,
  },
  happy_blissful: {
    name: VRMExpressionPresetName.Relaxed,
    peak: 0.7,
    attackSeconds: 0.18,
    holdSeconds: 1.9,
    releaseSeconds: 0.5,
  },
  joy_jump: {
    name: VRMExpressionPresetName.Happy,
    peak: 0.92,
    attackSeconds: 0.08,
    holdSeconds: 1.15,
    releaseSeconds: 0.42,
  },
  upset_angry: {
    name: VRMExpressionPresetName.Angry,
    peak: 0.82,
    attackSeconds: 0.08,
    holdSeconds: 1.5,
    releaseSeconds: 0.48,
  },
  crying: {
    name: VRMExpressionPresetName.Sad,
    peak: 0.82,
    attackSeconds: 0.16,
    holdSeconds: 2.0,
    releaseSeconds: 0.6,
  },
  sad_idle: {
    name: VRMExpressionPresetName.Sad,
    peak: 0.68,
    attackSeconds: 0.2,
    holdSeconds: 2.3,
    releaseSeconds: 0.65,
  },
} as const satisfies Record<string, ExpressionEnvelopeSpec>;

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
