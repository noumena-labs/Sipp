import type { ChatMessage, CogentEngine } from 'cogentlm';

export const DRAWING_COLORS = ['#111827', '#ffffff', '#ef4444', '#f97316', '#facc15', '#22c55e', '#38bdf8', '#8b5cf6'] as const;
export const HECKLE_VOICES = [
  'deadpan art critic',
  'chaos goblin',
  'overconfident museum curator',
  'tired art teacher',
  'sports commentator',
  'noir detective',
] as const;

const MISSING_HECKLE = 'Heckle unavailable; try another stroke.';

export type DrawingColor = typeof DRAWING_COLORS[number];
export type HeckleVoice = typeof HECKLE_VOICES[number];
export type CapturePresetId = 'turbo' | 'trace';

export interface DirectorPolicy {
  readonly maxSubjectChars: number;
  readonly maxFeatureChars: number;
  readonly maxWeirdChars: number;
  readonly maxQualityChars: number;
  readonly maxHeckleChars: number;
}

export interface DrawingDirectorConfig {
  readonly id: string;
  readonly perceptionPersona: string;
  readonly perceptionInstructions: readonly string[];
  readonly hecklePersona: string;
  readonly policy: DirectorPolicy;
}

export interface DrawingState {
  readonly strokeCount: number;
  readonly selectedColor: DrawingColor;
  readonly selectedSize: number;
  readonly canvasWidth: number;
  readonly canvasHeight: number;
  readonly voice: HeckleVoice;
}

export interface CapturedDrawing {
  readonly bytes: Uint8Array;
  readonly width: number;
  readonly height: number;
  readonly byteLength: number;
  readonly preset: CapturePresetId;
  readonly cropX: number;
  readonly cropY: number;
  readonly cropWidth: number;
  readonly cropHeight: number;
}

export interface SketchPerception {
  readonly subject: string;
  readonly features: readonly string[];
  readonly weirdDetail: string;
  readonly lineQuality: string;
  readonly parseStatus: 'parsed' | 'fallback';
  readonly parseNote?: string;
}

export interface SketchHeckle {
  readonly comment: string;
  readonly parseStatus: 'parsed' | 'fallback';
  readonly parseNote?: string;
}

export interface DrawingDirectorResult {
  readonly perception: SketchPerception;
  readonly heckle: SketchHeckle;
  readonly perceptionRawText: string;
  readonly heckleRawText: string;
  readonly perceptionPromptPreview: string;
  readonly hecklePromptPreview: string;
  readonly perceptionMs: number;
  readonly heckleMs: number;
}

interface RunDrawingDirectorArgs {
  readonly capture: CapturedDrawing;
  readonly state: DrawingState;
  readonly signal?: AbortSignal;
}

export const DEFAULT_DRAWING_DIRECTOR_CONFIG: DrawingDirectorConfig = {
  id: 'sketch-two-pass-heckle-loop',
  perceptionPersona: 'Sketch Perception, a fast vision model that extracts concrete visual facts from rough drawings.',
  perceptionInstructions: [
    'Return visual facts only. Do not try to be funny.',
    'Name the likely subject as a short noun phrase, even when uncertain.',
    'List concrete visible features such as shapes, facial parts, line direction, color, count, and placement.',
    'Do not answer with digits only, numbering, markdown, jokes, apologies, or uncertainty phrases.',
  ],
  hecklePersona: 'You are a snarky British comedy commentator watching someone draw badly. You speak in one short witty sentence in the spirit of Monty Python. You never use labels, lists, colons, or all-caps words. You never describe the drawing literally; you make a joke about it.',
  policy: {
    maxSubjectChars: 52,
    maxFeatureChars: 42,
    maxWeirdChars: 64,
    maxQualityChars: 64,
    maxHeckleChars: 160,
  },
};

export async function loadDrawingDirectorConfig(url: string): Promise<DrawingDirectorConfig> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`drawing-director.json HTTP ${response.status}`);
  }
  return normalizeConfig(await response.json());
}

export class DrawingDirector {
  public constructor(
    private readonly engine: CogentEngine,
    private readonly config: DrawingDirectorConfig
  ) { }

  public async run(args: RunDrawingDirectorArgs): Promise<DrawingDirectorResult> {
    const perceptionSystemPrompt = renderPerceptionSystemPrompt(this.config);
    const perceptionUserPrompt = renderPerceptionUserPrompt(args.capture, args.state);
    const perceptionMessages: ChatMessage[] = [
      { role: 'system', content: perceptionSystemPrompt },
      { role: 'user', content: perceptionUserPrompt },
    ];

    const perceptionStarted = performance.now();
    const perceptionRawText = await this.engine.chat(
      { messages: perceptionMessages, media: [args.capture.bytes] },
      {
        session: `proactive-ui:${this.config.id}:perception-v1`,
        maxTokens: args.capture.preset === 'turbo' ? 96 : 128,
        signal: args.signal,
      }
    );
    const perceptionMs = Math.round(performance.now() - perceptionStarted);
    const perception = parsePerceptionResponse(perceptionRawText, this.config.policy, args.state);

    const heckleExemplars = pickExemplars(args.state, 3);
    const heckleMessages = buildHeckleMessages(this.config, perception, args.state, heckleExemplars);
    const heckleSession = `proactive-ui:${this.config.id}:heckle:${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;

    const heckleStarted = performance.now();
    let heckleRawText = await this.engine.chat(
      { messages: heckleMessages },
      {
        session: heckleSession,
        maxTokens: 48,
        signal: args.signal,
      }
    );
    let heckleMs = Math.round(performance.now() - heckleStarted);
    let heckle = parseHeckleResponse(heckleRawText, this.config.policy);
    let hecklePromptPreview = renderMessagesPreview(heckleMessages);

    if (heckle.comment === MISSING_HECKLE) {
      const retryExemplars = pickExemplars(
        { ...args.state, strokeCount: args.state.strokeCount + 1 },
        1
      );
      const retryMessages = buildHeckleMessages(this.config, perception, args.state, retryExemplars);
      const retrySession = `proactive-ui:${this.config.id}:heckle-retry:${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
      const retryStarted = performance.now();
      const retryRawText = await this.engine.chat(
        { messages: retryMessages },
        {
          session: retrySession,
          maxTokens: 48,
          signal: args.signal,
        }
      );
      heckleMs += Math.round(performance.now() - retryStarted);
      heckle = parseHeckleResponse(retryRawText, this.config.policy);
      heckleRawText = `${heckleRawText.trim()}\n\n--- retry ---\n${retryRawText.trim()}`;
      hecklePromptPreview = `${hecklePromptPreview}\n\n--- retry messages ---\n${renderMessagesPreview(retryMessages)}`;
    }

    return {
      perception,
      heckle,
      perceptionRawText,
      heckleRawText,
      perceptionPromptPreview: `${perceptionSystemPrompt}\n\n${perceptionUserPrompt}`.slice(0, 1400),
      hecklePromptPreview: hecklePromptPreview.slice(0, 2200),
      perceptionMs,
      heckleMs,
    };
  }
}

function normalizeConfig(raw: unknown): DrawingDirectorConfig {
  const record = asRecord(raw);
  const policy = asRecord(record.policy);
  return {
    id: readString(record.id, DEFAULT_DRAWING_DIRECTOR_CONFIG.id),
    perceptionPersona: readString(record.perceptionPersona, DEFAULT_DRAWING_DIRECTOR_CONFIG.perceptionPersona),
    perceptionInstructions: readStringArray(record.perceptionInstructions, DEFAULT_DRAWING_DIRECTOR_CONFIG.perceptionInstructions),
    hecklePersona: readString(record.hecklePersona, DEFAULT_DRAWING_DIRECTOR_CONFIG.hecklePersona),
    policy: {
      maxSubjectChars: readPositiveInt(policy.maxSubjectChars, DEFAULT_DRAWING_DIRECTOR_CONFIG.policy.maxSubjectChars),
      maxFeatureChars: readPositiveInt(policy.maxFeatureChars, DEFAULT_DRAWING_DIRECTOR_CONFIG.policy.maxFeatureChars),
      maxWeirdChars: readPositiveInt(policy.maxWeirdChars, DEFAULT_DRAWING_DIRECTOR_CONFIG.policy.maxWeirdChars),
      maxQualityChars: readPositiveInt(policy.maxQualityChars, DEFAULT_DRAWING_DIRECTOR_CONFIG.policy.maxQualityChars),
      maxHeckleChars: readPositiveInt(policy.maxHeckleChars, DEFAULT_DRAWING_DIRECTOR_CONFIG.policy.maxHeckleChars),
    },
  };
}

function renderPerceptionSystemPrompt(config: DrawingDirectorConfig): string {
  return [
    config.perceptionPersona,
    'Return exactly four newline-separated fields named SUBJECT, FEATURES, WEIRD, and QUALITY.',
    'SUBJECT is a 2-5 word noun phrase. Never write a sentence.',
    'FEATURES is 3-6 visible details separated by commas.',
    'WEIRD is the single oddest or most roastable visible detail.',
    'QUALITY is the line quality or drawing style in a few words.',
    'No jokes. No numbering. No markdown. No digit-only answers. No apologies. No uncertainty phrases.',
    ...config.perceptionInstructions,
  ].join('\n');
}

function renderPerceptionUserPrompt(capture: CapturedDrawing, state: DrawingState): string {
  return [
    'Analyze only the visible ink in the image.',
    `Canvas: ${state.canvasWidth}x${state.canvasHeight}; ink crop: ${capture.cropWidth}x${capture.cropHeight} at ${capture.cropX},${capture.cropY}; capture: ${capture.width}x${capture.height}/${capture.preset}; strokes: ${state.strokeCount}; pen: ${state.selectedColor}/${state.selectedSize}px.`,
    'Use this field order: SUBJECT, FEATURES, WEIRD, QUALITY.',
  ].join('\n');
}

function renderHeckleSystemPrompt(config: DrawingDirectorConfig): string {
  return [
    config.hecklePersona,
    `Reply with exactly one short witty sentence, max ${config.policy.maxHeckleChars} characters, ending with . ? or !.`,
    'No labels. No colons. No lists. No ALL CAPS words. Do not restate the drawing facts. Do not repeat the user message.',
  ].join('\n');
}

interface HeckleExemplar {
  readonly drawing: string;
  readonly quip: string;
}

const HECKLE_EXEMPLARS: readonly HeckleExemplar[] = [
  { drawing: 'a lopsided cat with three whiskers.', quip: 'Ah, a rare British shorthair after seventeen pints and a bad divorce.' },
  { drawing: 'a stick figure holding a balloon.', quip: 'Behold, the proudest knight in all the land, defeated at last by a single party balloon.' },
  { drawing: 'a wobbly house with smoke coming out the wrong side.', quip: 'A charming family home, currently being haunted by an extremely confused chimney inspector.' },
  { drawing: 'a duck wearing a tiny crown.', quip: 'His Majesty is, regrettably, made entirely of leftover bath sponges.' },
];

function buildHeckleMessages(config: DrawingDirectorConfig, perception: SketchPerception, state: DrawingState, exemplars: readonly HeckleExemplar[]): ChatMessage[] {
  const messages: ChatMessage[] = [
    { role: 'system', content: renderHeckleSystemPrompt(config) },
  ];
  for (const example of exemplars) {
    messages.push({ role: 'user', content: `Drawing: ${example.drawing}` });
    messages.push({ role: 'assistant', content: example.quip });
  }
  void state;
  messages.push({ role: 'user', content: `Drawing: ${describePerceptionForPrompt(perception)}.` });
  return messages;
}

function pickExemplars(state: DrawingState, count: number): readonly HeckleExemplar[] {
  const start = Math.abs(state.strokeCount) % HECKLE_EXEMPLARS.length;
  const result: HeckleExemplar[] = [];
  for (let i = 0; i < count && i < HECKLE_EXEMPLARS.length; i += 1) {
    result.push(HECKLE_EXEMPLARS[(start + i) % HECKLE_EXEMPLARS.length]);
  }
  return result;
}

function describePerceptionForPrompt(perception: SketchPerception): string {
  const features = perception.features.slice(0, 3).join(', ');
  const weird = perception.weirdDetail && perception.weirdDetail.length > 0 ? `; the oddest bit is ${perception.weirdDetail}` : '';
  return `${perception.subject}${features ? ` with ${features}` : ''}${weird}`;
}

function renderMessagesPreview(messages: readonly ChatMessage[]): string {
  return messages.map((m) => `[${m.role}] ${m.content}`).join('\n');
}

function parsePerceptionResponse(rawText: string, policy: DirectorPolicy, state: DrawingState): SketchPerception {
  const fields = extractFields(rawText, ['SUBJECT', 'FEATURES', 'WEIRD', 'QUALITY']);
  const subject = cleanSubject(fields.get('SUBJECT') ?? '', policy.maxSubjectChars);
  const features = cleanFeatures(fields.get('FEATURES') ?? '', policy.maxFeatureChars);
  const weirdDetail = cleanField(fields.get('WEIRD') ?? '', policy.maxWeirdChars, ['SUBJECT', 'FEATURES', 'WEIRD', 'QUALITY']);
  const lineQuality = cleanField(fields.get('QUALITY') ?? '', policy.maxQualityChars, ['SUBJECT', 'FEATURES', 'WEIRD', 'QUALITY']);
  const parsed = subject !== '' && features.length >= 2 && weirdDetail !== '' && lineQuality !== '';

  if (parsed) {
    return {
      subject,
      features,
      weirdDetail,
      lineQuality,
      parseStatus: 'parsed',
    };
  }

  const fallbackFeatures = inferFallbackFeatures(state);
  return {
    subject: subject || 'mystery sketch',
    features: features.length > 0 ? features : fallbackFeatures,
    weirdDetail: weirdDetail || fallbackFeatures[0],
    lineQuality: lineQuality || `${state.strokeCount} visible strokes`,
    parseStatus: 'fallback',
    parseNote: `perception parse failed; raw=${truncateText(rawText.trim() || 'empty', 140)}`,
  };
}

function parseHeckleResponse(rawText: string, policy: DirectorPolicy): SketchHeckle {
  const lines = rawText
    .replace(/\r/g, '\n')
    .split('\n')
    .map((line) => cleanHeckle(line, policy.maxHeckleChars))
    .filter((line) => line.length > 0);

  for (const line of lines) {
    if (isUsableHeckle(line)) {
      return {
        comment: line,
        parseStatus: 'parsed',
      };
    }
  }

  const collapsed = cleanHeckle(rawText.replace(/\s+/g, ' '), policy.maxHeckleChars);
  if (isUsableHeckle(collapsed)) {
    return {
      comment: collapsed,
      parseStatus: 'fallback',
      parseNote: 'heckle accepted from collapsed raw text',
    };
  }

  return {
    comment: MISSING_HECKLE,
    parseStatus: 'fallback',
    parseNote: `heckle parse failed; raw=${truncateText(rawText.trim() || 'empty', 140)}`,
  };
}

function extractFields(rawText: string, names: readonly string[]): Map<string, string> {
  const fields = new Map<string, string>();
  const labels = names.map(escapeRegExp).join('|');
  const labelPattern = new RegExp(`\\b(${labels})\\s*[:|]`, 'gi');
  const linePattern = new RegExp(`^(${labels})\\s*[:|]\\s*(.+)$`, 'i');
  const normalized = rawText
    .replace(/\r/g, '\n')
    .replace(labelPattern, '\n$1:')
    .split('\n')
    .map((line) => line.trim().replace(/^[-*\d.\s]+/, ''))
    .filter(Boolean);

  for (const line of normalized) {
    const match = linePattern.exec(line);
    if (!match) {
      continue;
    }
    fields.set(match[1].toUpperCase(), stripQuotes(match[2]));
  }
  return fields;
}

function cleanSubject(value: string, maxLength: number): string {
  const cleaned = cleanField(value, maxLength, ['SUBJECT', 'FEATURES', 'WEIRD', 'QUALITY'])
    .replace(/^(?:a\s+)?(?:drawing|sketch|picture)\s+of\s+(?:a\s+|an\s+)?/i, '')
    .replace(/^(?:it\s+looks\s+like|looks\s+like|maybe|probably|possibly)\s+(?:a\s+|an\s+)?/i, '')
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 5)
    .join(' ')
    .trim();
  return isBadField(cleaned) ? '' : cleaned;
}

function cleanFeatures(value: string, maxFeatureChars: number): string[] {
  return value
    .split(/[,;|]/)
    .map((part) => cleanField(part, maxFeatureChars, ['SUBJECT', 'FEATURES', 'WEIRD', 'QUALITY']))
    .filter((part) => part.length > 0 && !isBadField(part))
    .slice(0, 6);
}

function cleanField(value: string, maxLength: number, labels: readonly string[]): string {
  const labelPattern = new RegExp(`^(?:${labels.map(escapeRegExp).join('|')})\\s*[:|]\\s*`, 'i');
  const trailingLabelPattern = new RegExp(`\\b(?:${labels.map(escapeRegExp).join('|')})\\s*[:|].*$`, 'i');
  const cleaned = value
    .replace(labelPattern, '')
    .replace(trailingLabelPattern, '')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/^['"]+|['"]+$/g, '')
    .replace(/[,.!?;:]+$/, '')
    .replace(/^['"]+|['"]+$/g, '')
    .trim();
  return truncateText(cleaned, maxLength);
}

function cleanHeckle(value: string, maxLength: number): string {
  const cleaned = value
    .replace(/^\s*(?:HECKLE|QUIP|JOKE|ANSWER|REPLY|RESPONSE|HEECKLE|DRAWING)\s*[:|-]\s*/i, '')
    .replace(/\b(?:SUBJECT|FEATURES|WEIRD|QUALITY|HECKLE|QUIP|DRAWING)\s*[:|].*$/i, '')
    .replace(/\s+/g, ' ')
    .trim()
    .replace(/^[-*\d.\s]+/, '')
    .replace(/^['"]+|['"]+$/g, '')
    .trim();
  return truncateText(cleaned, maxLength);
}

function isUsableHeckle(value: string): boolean {
  if (isBadField(value)) return false;
  if (value.length < 12) return false;
  if (/[<>]/.test(value)) return false;
  if (/\b(?:Rule|Constraint|Task|Style|Target|Output|Sketch|Details|Odd bit|Line style|Voice|SUBJECT|FEATURES|WEIRD|QUALITY|DRAWING CONTEXT|Quip|Drawing)\s*[:=]/i.test(value)) return false;
  if (/\b(?:write one|write exactly|one line only|one single sentence|exactly one short|answer with a joke only|visual facts|drawing only|not the artist|do not quote|do not repeat|repeat the prompt|existing comedy|pompous civic authority|royal bureaucracy|procedural nonsense|begin with heckle|no instructions|no explanation|no labels|no colons|no all caps|do not restate|do not repeat the user|reply with)\b/i.test(value)) return false;
  // Reject exemplar echoes (parser guard against in-context examples being copied verbatim).
  for (const ex of HECKLE_EXEMPLARS) {
    if (value.toLowerCase().includes(ex.quip.slice(0, 24).toLowerCase())) return false;
    if (value.toLowerCase().includes(ex.drawing.slice(0, 18).toLowerCase())) return false;
  }
  // Reject "ALL CAPS WORD: rest" style label lines.
  if (/^[A-Z][A-Z0-9 \-]{2,}:\s*/.test(value)) return false;
  // Reject lines with multiple ALL-CAPS tokens.
  const allCapsTokens = (value.match(/\b[A-Z]{3,}\b/g) ?? []).length;
  if (allCapsTokens >= 2) return false;
  // Reject perception echoes.
  if (/\b\d+\s*(?:strokes|px)\b/i.test(value)) return false;
  if (/#[0-9a-f]{3,8}\b/i.test(value)) return false;
  return true;
}

function inferFallbackFeatures(state: DrawingState): string[] {
  return [
    `${state.strokeCount} visible strokes`,
    `${state.selectedColor} ink`,
    `${state.selectedSize}px pen lines`,
  ];
}

function isBadField(value: string): boolean {
  return value.length === 0 || /^\d+$/.test(value) || /^(none|null|unknown|n\/a|not sure|cannot tell|can't tell)$/i.test(value);
}

function asRecord(value: unknown): Record<string, unknown> {
  return value != null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function readString(value: unknown, fallback: string): string {
  return typeof value === 'string' && value.trim().length > 0 ? value.trim() : fallback;
}

function readStringArray(value: unknown, fallback: readonly string[]): readonly string[] {
  if (!Array.isArray(value)) {
    return fallback;
  }
  const strings = value.filter((item): item is string => typeof item === 'string' && item.trim().length > 0);
  return strings.length > 0 ? strings.map((item) => item.trim()) : fallback;
}

function readPositiveInt(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) && value > 0 ? Math.round(value) : fallback;
}

function stripQuotes(value: string): string {
  return value.trim().replace(/^['"]|['"]$/g, '');
}

function truncateText(text: string, maxLength: number): string {
  return text.length <= maxLength ? text : `${text.slice(0, Math.max(0, maxLength - 3))}...`;
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}
