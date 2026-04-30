import type { ChatMessage, CogentEngine } from '@noumena-labs/cogent-engine';
import DOMPurify from 'dompurify';
import type { FieldKitScore, GearItem } from './game-data';
import { categoryLabel, FIELD_KIT_LIMITS } from './game-data';

const SUPPORTED_OPS = [
  'replaceText',
  'replaceHtml',
  'appendHtml',
  'addClass',
  'removeClass',
  'setAttribute',
  'scrollIntoView',
] as const;

const PATCH_OPERATION_ALIASES: Record<string, NormalizedPatchOperation> = {
  replacetext: { op: 'replaceText' },
  settext: { op: 'replaceText' },
  updatetext: { op: 'replaceText' },
  text: { op: 'replaceText' },
  replacelabel: { op: 'replaceText' },
  replacehtml: { op: 'replaceHtml' },
  sethtml: { op: 'replaceHtml' },
  updatehtml: { op: 'replaceHtml' },
  html: { op: 'replaceHtml' },
  markup: { op: 'replaceHtml' },
  appendhtml: { op: 'appendHtml' },
  inserthtml: { op: 'appendHtml' },
  append: { op: 'appendHtml' },
  insert: { op: 'appendHtml' },
  addclass: { op: 'addClass' },
  highlight: { op: 'addClass', defaultClassName: 'ai-spotlight' },
  spotlight: { op: 'addClass', defaultClassName: 'ai-spotlight' },
  pulse: { op: 'addClass', defaultClassName: 'ai-pulse' },
  warning: { op: 'addClass', defaultClassName: 'ai-warning' },
  warn: { op: 'addClass', defaultClassName: 'ai-warning' },
  success: { op: 'addClass', defaultClassName: 'ai-success' },
  dim: { op: 'addClass', defaultClassName: 'ai-dim' },
  removeclass: { op: 'removeClass' },
  clearclass: { op: 'removeClass' },
  setattribute: { op: 'setAttribute' },
  setattr: { op: 'setAttribute' },
  attribute: { op: 'setAttribute' },
  scrollintoview: { op: 'scrollIntoView' },
  scroll: { op: 'scrollIntoView' },
  reveal: { op: 'scrollIntoView' },
} as const;

const JSON_GRAMMAR = String.raw`
root ::= object
value ::= object | array | string | number | "true" ws | "false" ws | "null" ws
object ::= "{" ws (string ":" ws value ("," ws string ":" ws value)*)? "}" ws
array ::= "[" ws (value ("," ws value)*)? "]" ws
string ::= "\"" ([^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F] [0-9a-fA-F]))* "\"" ws
number ::= ("-"? ([0-9] | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?) ws
ws ::= [ \t\n\r]*
`.trim();

type SupportedPatchOp = typeof SUPPORTED_OPS[number];

interface NormalizedPatchOperation {
  readonly op: SupportedPatchOp;
  readonly defaultClassName?: string;
}

export interface PatchPolicy {
  readonly maxPatches: number;
  readonly maxTextChars: number;
  readonly maxHtmlChars: number;
  readonly allowedClasses: readonly string[];
  readonly allowedHtmlClasses: readonly string[];
  readonly allowedAttributes: readonly string[];
}

export interface DomPatchDirectorConfig {
  readonly id: string;
  readonly scenarioName: string;
  readonly objective: string;
  readonly instructions: readonly string[];
  readonly patchPolicy: PatchPolicy;
}

export interface PatchTargetContract {
  readonly id: string;
  readonly label: string;
  readonly allowedOps: readonly SupportedPatchOp[];
  readonly tagName: string;
}

export interface DirectorGameState {
  readonly score: FieldKitScore;
  readonly selectedItems: readonly GearItem[];
}

export type DomPatch =
  | ({ readonly op: 'replaceText'; readonly targetId: string; readonly text: string } & PatchNote)
  | ({ readonly op: 'replaceHtml'; readonly targetId: string; readonly html: string } & PatchNote)
  | ({ readonly op: 'appendHtml'; readonly targetId: string; readonly html: string } & PatchNote)
  | ({ readonly op: 'addClass'; readonly targetId: string; readonly className: string } & PatchNote)
  | ({ readonly op: 'removeClass'; readonly targetId: string; readonly className: string } & PatchNote)
  | {
    readonly op: 'setAttribute';
    readonly targetId: string;
    readonly attributeName: string;
    readonly value: string;
  } & PatchNote
  | ({ readonly op: 'scrollIntoView'; readonly targetId: string } & PatchNote);

interface PatchNote {
  readonly note?: string;
}

export interface RawPatchResponse {
  readonly observation: string;
  readonly intent: string;
  readonly patches: readonly unknown[];
}

export interface RejectedPatch {
  readonly index: number;
  readonly reason: string;
  readonly patch: unknown;
}

export interface ValidatedPatchResponse {
  readonly observation: string;
  readonly intent: string;
  readonly patches: readonly DomPatch[];
  readonly rejectedPatches: readonly RejectedPatch[];
}

export interface DomPatchRunResult extends ValidatedPatchResponse {
  readonly rawText: string;
  readonly targetCount: number;
  readonly promptPreview: string;
}

export interface AppliedMutation {
  readonly targetId: string;
  readonly summary: string;
}

interface RunDomPatchDirectorArgs {
  readonly screenshot: Uint8Array;
  readonly targets: readonly PatchTargetContract[];
  readonly gameState: DirectorGameState;
  readonly signal?: AbortSignal;
}

export const DEFAULT_PATCH_DIRECTOR_CONFIG: DomPatchDirectorConfig = {
  id: 'dust-ridge-proactive-ui',
  scenarioName: 'Dust Ridge Field Kit',
  objective: 'Inspect the field-kit planner and patch the annotated DOM to help the user finish safely.',
  instructions: [
    'Reason from the screenshot and supplied game state only.',
    'Return only valid JSON with observation, intent, and patches.',
    'Patch only targets from the DOM contract.',
  ],
  patchPolicy: {
    maxPatches: 3,
    maxTextChars: 220,
    maxHtmlChars: 700,
    allowedClasses: ['ai-spotlight', 'ai-warning', 'ai-success', 'ai-dim', 'ai-pulse'],
    allowedHtmlClasses: [
      'ai-gen-card',
      'ai-gen-title',
      'ai-gen-note',
      'ai-gen-action',
      'ai-gen-warning',
      'ai-gen-success',
      'ai-gen-list',
    ],
    allowedAttributes: ['aria-label', 'title', 'data-ai-state', 'data-ai-note'],
  },
};

export async function loadDomPatchDirectorConfig(url: string): Promise<DomPatchDirectorConfig> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`dom-patch-director.json HTTP ${response.status}`);
  }
  return normalizeDirectorConfig(await response.json());
}

export class DomPatchDirector {
  public constructor(
    private readonly engine: CogentEngine,
    private readonly config: DomPatchDirectorConfig
  ) { }

  public async run(args: RunDomPatchDirectorArgs): Promise<DomPatchRunResult> {
    const userPrompt = renderUserPrompt(this.config, args.targets, args.gameState);
    const messages: ChatMessage[] = [
      { role: 'system', content: renderSystemPrompt(this.config) },
      { role: 'user', content: userPrompt },
    ];
    const rawText = await this.engine.chat(
      { messages, media: [args.screenshot] },
      {
        session: `proactive-ui:${this.config.id}`,
        maxTokens: 340,
        signal: args.signal,
        grammar: JSON_GRAMMAR,
      }
    );
    const parsed = parsePatchResponse(rawText);
    const validated = validatePatchResponse(parsed, args.targets, this.config.patchPolicy);

    return {
      rawText,
      targetCount: args.targets.length,
      promptPreview: userPrompt.slice(0, 2400),
      ...validated,
    };
  }
}

export function collectPatchTargets(root: HTMLElement): PatchTargetContract[] {
  const nodes = Array.from(root.querySelectorAll<HTMLElement>('[data-ai-id]'));
  if (root.dataset.aiId) {
    nodes.unshift(root);
  }

  return nodes.flatMap((node) => {
    const id = node.dataset.aiId?.trim();
    if (!id) {
      return [];
    }
    const allowedOps = parseAllowedOps(node.dataset.aiOps ?? '');
    if (allowedOps.length === 0) {
      return [];
    }
    return [
      {
        id,
        label: node.dataset.aiLabel?.trim() || id,
        allowedOps,
        tagName: node.tagName.toLowerCase(),
      },
    ];
  });
}

export function applyDomPatches(
  root: HTMLElement,
  patches: readonly DomPatch[]
): readonly AppliedMutation[] {
  removePatchNotes(root);
  const targetNodes = collectTargetNodes(root);
  const mutations: AppliedMutation[] = [];

  for (const patch of patches) {
    const target = targetNodes.get(patch.targetId);
    if (!target) {
      continue;
    }

    switch (patch.op) {
      case 'replaceText':
        target.textContent = patch.text;
        mutations.push({ targetId: patch.targetId, summary: `Replaced text on ${patch.targetId}` });
        break;
      case 'replaceHtml':
        target.innerHTML = patch.html;
        mutations.push({ targetId: patch.targetId, summary: `Replaced HTML on ${patch.targetId}` });
        break;
      case 'appendHtml':
        target.insertAdjacentHTML('beforeend', patch.html);
        mutations.push({ targetId: patch.targetId, summary: `Appended generated HTML to ${patch.targetId}` });
        break;
      case 'addClass':
        target.classList.add(patch.className);
        mutations.push({ targetId: patch.targetId, summary: `Added .${patch.className} to ${patch.targetId}` });
        break;
      case 'removeClass':
        target.classList.remove(patch.className);
        mutations.push({ targetId: patch.targetId, summary: `Removed .${patch.className} from ${patch.targetId}` });
        break;
      case 'setAttribute':
        target.setAttribute(patch.attributeName, patch.value);
        mutations.push({ targetId: patch.targetId, summary: `Set ${patch.attributeName} on ${patch.targetId}` });
        break;
      case 'scrollIntoView':
        target.scrollIntoView({ block: 'center', behavior: 'smooth' });
        mutations.push({ targetId: patch.targetId, summary: `Scrolled ${patch.targetId} into view` });
        break;
    }

    if (patch.note) {
      addPatchNote(target, patch.note);
      mutations.push({ targetId: patch.targetId, summary: `Added explanatory note to ${patch.targetId}` });
    }
    target.dataset.aiModified = 'true';
  }

  return mutations;
}

function normalizeDirectorConfig(raw: unknown): DomPatchDirectorConfig {
  const record = asRecord(raw);
  const policy = asRecord(record.patchPolicy);
  return {
    id: readString(record.id, DEFAULT_PATCH_DIRECTOR_CONFIG.id),
    scenarioName: readString(record.scenarioName, DEFAULT_PATCH_DIRECTOR_CONFIG.scenarioName),
    objective: readString(record.objective, DEFAULT_PATCH_DIRECTOR_CONFIG.objective),
    instructions: readStringArray(record.instructions, DEFAULT_PATCH_DIRECTOR_CONFIG.instructions),
    patchPolicy: {
      maxPatches: readPositiveInt(policy.maxPatches, DEFAULT_PATCH_DIRECTOR_CONFIG.patchPolicy.maxPatches),
      maxTextChars: readPositiveInt(policy.maxTextChars, DEFAULT_PATCH_DIRECTOR_CONFIG.patchPolicy.maxTextChars),
      maxHtmlChars: readPositiveInt(policy.maxHtmlChars, DEFAULT_PATCH_DIRECTOR_CONFIG.patchPolicy.maxHtmlChars),
      allowedClasses: readStringArray(policy.allowedClasses, DEFAULT_PATCH_DIRECTOR_CONFIG.patchPolicy.allowedClasses),
      allowedHtmlClasses: readStringArray(
        policy.allowedHtmlClasses,
        DEFAULT_PATCH_DIRECTOR_CONFIG.patchPolicy.allowedHtmlClasses
      ),
      allowedAttributes: readStringArray(
        policy.allowedAttributes,
        DEFAULT_PATCH_DIRECTOR_CONFIG.patchPolicy.allowedAttributes
      ),
    },
  };
}

function renderSystemPrompt(config: DomPatchDirectorConfig): string {
  return [
    `You are the proactive UI director for ${config.scenarioName}. Inspect the screenshot first.`,
    'Return only JSON: {"observation":string,"intent":string,"patches":[Patch]}.',
    'Patch examples: {"op":"addClass","targetId":"goal-hydration","className":"ai-warning","note":"Hydration is missing. Add water before launch."} or {"op":"replaceText","targetId":"launch-button","text":"Ready","note":"All constraints are satisfied."}.',
    'Every patch must include op, targetId, and a helpful note. Patch only listed target ids.',
    'Instructions:',
    ...config.instructions.map((instruction) => `- ${instruction}`),
  ].join('\n');
}

function renderUserPrompt(
  config: DomPatchDirectorConfig,
  targets: readonly PatchTargetContract[],
  state: DirectorGameState
): string {
  const selected = state.selectedItems.map((item) => `${item.name}(${categoryLabel(item.category)})`).join(', ') || 'none';
  const missing = state.score.missingCategories.map(categoryLabel).join(', ') || 'none';
  const covered = state.score.coveredCategories.map(categoryLabel).join(', ') || 'none';
  const targetLines = targets
    .map((target) => `${target.id}|${target.label}|ops:${target.allowedOps.join(',')}`)
    .join('\n');

  return [
    'Inspect the screenshot and help the user finish the field kit.',
    `State: readiness=${state.score.readiness}; weight=${state.score.totalWeight}/${FIELD_KIT_LIMITS.maxWeight}kg; budget=$${state.score.totalCost}/$${FIELD_KIT_LIMITS.maxBudget}; ready=${state.score.readyToLaunch}; missing=${missing}; covered=${covered}; selected=${selected}.`,
    `Rules: maxPatches=${config.patchPolicy.maxPatches}; allowedClasses=${config.patchPolicy.allowedClasses.join('|')}; use only target ids below; every patch needs note; prefer the most important missing/risky item; output JSON only.`,
    'Targets:',
    targetLines,
  ].join('\n\n');
}

function parsePatchResponse(rawText: string): RawPatchResponse {
  const parsed = JSON.parse(rawText.trim()) as unknown;
  const record = asRecord(parsed);
  const patches = readPatchArray(record);
  return {
    observation: readStringFrom(
      record,
      ['observation', 'analysis', 'description', 'summary'],
      'The model returned no observation.'
    ),
    intent: readStringFrom(
      record,
      ['intent', 'userIntent', 'inferredIntent', 'goal'],
      'The model returned no intent.'
    ),
    patches,
  };
}

function validatePatchResponse(
  response: RawPatchResponse,
  targets: readonly PatchTargetContract[],
  policy: PatchPolicy
): ValidatedPatchResponse {
  const targetMap = new Map(targets.map((target) => [target.id, target]));
  const accepted: DomPatch[] = [];
  const rejected: RejectedPatch[] = [];
  const candidatePatches = response.patches.slice(0, policy.maxPatches);

  candidatePatches.forEach((patch, index) => {
    const result = validatePatch(patch, targetMap, policy);
    if (typeof result === 'string') {
      rejected.push({ index, patch, reason: result });
      return;
    }
    accepted.push(result);
  });

  return {
    observation: compactText(response.observation, 700),
    intent: compactText(response.intent, 420),
    patches: accepted,
    rejectedPatches: rejected,
  };
}

function validatePatch(
  patch: unknown,
  targetMap: ReadonlyMap<string, PatchTargetContract>,
  policy: PatchPolicy
): DomPatch | string {
  const record = normalizePatchRecord(patch);
  const operation = readPatchOperation(record);
  if (!operation) {
    return `Unsupported op ${JSON.stringify(readStringFrom(record, ['op', 'operation', 'action', 'type', 'kind', 'command'], ''))}; patch keys: ${Object.keys(record).join(', ') || 'none'}.`;
  }
  const op = operation.op;
  const rawTargetId = readStringFrom(
    record,
    ['targetId', 'target', 'id', 'elementId', 'element', 'aiId', 'selector', 'query'],
    ''
  );
  const targetId = resolveTargetId(rawTargetId, targetMap);
  if (!targetId) {
    return `Unknown targetId ${JSON.stringify(rawTargetId)}.`;
  }
  const target = targetMap.get(targetId)!;
  if (!target) {
    return `Unknown targetId ${JSON.stringify(targetId)}.`;
  }
  const note = readPatchNote(record, policy) ?? readContentAsNote(record, policy) ?? defaultPatchNote(op, target);
  if (!target.allowedOps.includes(op)) {
    const fallbackClass = defaultClassForTarget(target);
    if (target.allowedOps.includes('addClass') && policy.allowedClasses.includes(fallbackClass)) {
      return withPatchNote({ op: 'addClass', targetId, className: fallbackClass }, note);
    }
    return `${targetId} does not allow ${op}.`;
  }

  switch (op) {
    case 'replaceText': {
      const text = compactText(readStringFrom(record, ['text', 'content', 'value', 'innerText', 'message', 'copy'], ''), policy.maxTextChars);
      return text ? withPatchNote({ op, targetId, text }, note) : 'replaceText requires non-empty text.';
    }
    case 'replaceHtml':
    case 'appendHtml': {
      const html = sanitizeGeneratedHtml(readStringFrom(record, ['html', 'content', 'value', 'innerHTML', 'markup', 'body'], ''), policy);
      return html ? withPatchNote({ op, targetId, html }, note) : `${op} requires safe non-empty html.`;
    }
    case 'addClass':
    case 'removeClass': {
      const className = readClassName(record, operation.defaultClassName ?? defaultClassForTarget(target));
      return policy.allowedClasses.includes(className)
        ? withPatchNote({ op, targetId, className }, note)
        : `Class ${JSON.stringify(className)} is not allowed.`;
    }
    case 'setAttribute': {
      const attributeName = readStringFrom(record, ['attributeName', 'attribute', 'attr', 'name'], '');
      const value = compactText(readStringFrom(record, ['value', 'attributeValue', 'content', 'text'], ''), 160);
      return policy.allowedAttributes.includes(attributeName)
        ? withPatchNote({ op, targetId, attributeName, value }, note)
        : `Attribute ${JSON.stringify(attributeName)} is not allowed.`;
    }
    case 'scrollIntoView':
      return withPatchNote({ op, targetId }, note);
  }
}

function withPatchNote<TPatch extends Omit<DomPatch, 'note'>>(
  patch: TPatch,
  note: string | undefined
): TPatch & PatchNote {
  return note ? { ...patch, note } : patch;
}

function readPatchNote(record: Record<string, unknown>, policy: PatchPolicy): string | undefined {
  const note = compactText(readStringFrom(record, ['note', 'reason', 'message', 'explanation', 'why'], ''), policy.maxTextChars);
  return note.length > 0 ? note : undefined;
}

function readContentAsNote(record: Record<string, unknown>, policy: PatchPolicy): string | undefined {
  const content = stripHtml(readStringFrom(record, ['text', 'content', 'value', 'html', 'innerHTML'], ''));
  const note = compactText(content, policy.maxTextChars);
  return note.length > 0 ? note : undefined;
}

function defaultPatchNote(op: SupportedPatchOp, target: PatchTargetContract): string {
  if (op === 'addClass') {
    return `The model flagged ${target.label} as the next thing to check for the mission.`;
  }
  if (op === 'replaceHtml' || op === 'replaceText' || op === 'appendHtml') {
    return `The model updated ${target.label} with guidance from the screenshot.`;
  }
  return `The model adjusted ${target.label} based on the current screenshot.`;
}

function defaultClassForTarget(target: PatchTargetContract): string {
  if (target.id.startsWith('goal-') || target.id.startsWith('meter-') || target.id.includes('launch')) {
    return 'ai-warning';
  }
  return 'ai-spotlight';
}

function readPatchArray(record: Record<string, unknown>): readonly unknown[] {
  const aliases = ['patches', 'domPatches', 'operations', 'mutations', 'actions', 'updates'];
  for (const alias of aliases) {
    const value = record[alias];
    if (Array.isArray(value)) {
      return value;
    }
  }
  if (
    hasAnyKey(record, ['op', 'operation', 'action', 'type', 'kind', 'command', 'patchOp']) &&
    hasAnyKey(record, ['targetId', 'target', 'id', 'elementId', 'element', 'aiId', 'selector', 'query'])
  ) {
    return [record];
  }
  for (const alias of ['patch', 'domPatch', 'mutation']) {
    const singlePatch = asRecord(record[alias]);
    if (Object.keys(singlePatch).length > 0) {
      return [singlePatch];
    }
  }
  return [];
}

function normalizePatchRecord(patch: unknown): Record<string, unknown> {
  const record = asRecord(patch);
  const nestedPatch = record.patch ?? record.domPatch ?? record.mutation;
  if (!hasAnyKey(record, ['op', 'operation', 'action', 'type', 'kind', 'command']) && nestedPatch != null) {
    const nested = asRecord(nestedPatch);
    if (Object.keys(nested).length > 0) {
      return normalizePatchRecord(nested);
    }
  }

  const keys = Object.keys(record);
  if (!hasAnyKey(record, ['op', 'operation', 'action', 'type', 'kind', 'command']) && keys.length === 1) {
    const wrapperOp = readNormalizedPatchOperation(keys[0]);
    const payload = asRecord(record[keys[0]]);
    if (wrapperOp && Object.keys(payload).length > 0) {
      return { ...payload, op: keys[0] };
    }
  }

  return record;
}

function readPatchOperation(record: Record<string, unknown>): NormalizedPatchOperation | null {
  const explicitOperation = readStringFrom(
    record,
    ['op', 'operation', 'action', 'type', 'kind', 'command', 'patchOp'],
    ''
  );
  const normalized = readNormalizedPatchOperation(explicitOperation);
  if (normalized) {
    return normalized;
  }
  return inferPatchOperation(record);
}

function readNormalizedPatchOperation(value: string): NormalizedPatchOperation | null {
  const normalized = normalizeIdentifier(value);
  return PATCH_OPERATION_ALIASES[normalized] ?? null;
}

function inferPatchOperation(record: Record<string, unknown>): NormalizedPatchOperation | null {
  if (hasStringLike(record, ['html', 'innerHTML', 'markup', 'body'])) {
    return { op: 'replaceHtml' };
  }
  if (hasStringLike(record, ['content', 'value'])) {
    const content = readStringFrom(record, ['content', 'value'], '');
    return content.includes('<') ? { op: 'replaceHtml' } : { op: 'replaceText' };
  }
  if (hasStringLike(record, ['text', 'innerText', 'message', 'copy'])) {
    return { op: 'replaceText' };
  }
  if (hasStringLike(record, ['className', 'class', 'cssClass'])) {
    return { op: 'addClass' };
  }
  if (hasStringLike(record, ['attributeName', 'attribute', 'attr', 'name'])) {
    return { op: 'setAttribute' };
  }
  return null;
}

function resolveTargetId(
  value: string,
  targetMap: ReadonlyMap<string, PatchTargetContract>
): string | null {
  const targetId = normalizeTargetId(value);
  if (targetMap.has(targetId)) {
    return targetId;
  }

  const variants = [
    `gear-${targetId}`,
    `goal-${targetId}`,
    `meter-${targetId}`,
    `tab-${targetId}`,
  ];
  for (const variant of variants) {
    if (targetMap.has(variant)) {
      return variant;
    }
  }

  const suffixMatch = Array.from(targetMap.keys()).find((candidate) => candidate.endsWith(`-${targetId}`));
  return suffixMatch ?? null;
}

function normalizeTargetId(value: string): string {
  const trimmed = value.trim();
  const dataAiIdMatch = trimmed.match(/data-ai-id\s*=\s*["']?([^"'\]]+)/i);
  if (dataAiIdMatch?.[1]) {
    return dataAiIdMatch[1].trim();
  }
  return trimmed.replace(/^#/, '').replace(/^data-ai-id:/, '').replace(/^data-ai-id=/, '').replace(/^['"]|['"]$/g, '');
}

function sanitizeGeneratedHtml(html: string, policy: PatchPolicy): string {
  const trimmed = html.slice(0, policy.maxHtmlChars);
  const sanitized = DOMPurify.sanitize(trimmed, {
    ALLOWED_TAGS: ['p', 'strong', 'em', 'span', 'ul', 'ol', 'li', 'h3', 'h4', 'button', 'div', 'small', 'br'],
    ALLOWED_ATTR: ['class', 'aria-label', 'data-ai-generated'],
    FORBID_TAGS: ['script', 'style', 'iframe', 'img', 'svg', 'math', 'form', 'input', 'textarea'],
    RETURN_TRUSTED_TYPE: false,
  }) as string;
  const template = document.createElement('template');
  template.innerHTML = sanitized;
  for (const element of Array.from(template.content.querySelectorAll<HTMLElement>('[class]'))) {
    for (const className of Array.from(element.classList)) {
      if (!policy.allowedHtmlClasses.includes(className)) {
        element.classList.remove(className);
      }
    }
    if (element.classList.length === 0) {
      element.removeAttribute('class');
    }
  }
  return template.innerHTML.trim();
}

function collectTargetNodes(root: HTMLElement): Map<string, HTMLElement> {
  const nodes = Array.from(root.querySelectorAll<HTMLElement>('[data-ai-id]'));
  if (root.dataset.aiId) {
    nodes.unshift(root);
  }
  return new Map(
    nodes.flatMap((node) => {
      const id = node.dataset.aiId?.trim();
      return id ? [[id, node] as const] : [];
    })
  );
}

function addPatchNote(target: HTMLElement, note: string): void {
  const noteElement = document.createElement('span');
  noteElement.className = 'ai-patch-note';
  noteElement.dataset.aiPatchNote = 'true';
  noteElement.textContent = note;
  target.appendChild(noteElement);
}

function removePatchNotes(root: HTMLElement): void {
  for (const note of Array.from(root.querySelectorAll('[data-ai-patch-note="true"]'))) {
    note.remove();
  }
}

function parseAllowedOps(source: string): SupportedPatchOp[] {
  const parts = source.split(',').map((part) => part.trim()).filter(Boolean);
  return parts.filter((part): part is SupportedPatchOp => SUPPORTED_OPS.includes(part as SupportedPatchOp));
}

function readStringFrom(
  record: Record<string, unknown>,
  keys: readonly string[],
  fallback: string
): string {
  for (const key of keys) {
    const value = record[key];
    if (typeof value === 'string') {
      return value;
    }
    if (typeof value === 'number' || typeof value === 'boolean') {
      return String(value);
    }
  }
  return fallback;
}

function readClassName(record: Record<string, unknown>, fallback: string): string {
  for (const key of ['className', 'class', 'classes', 'cssClass', 'value']) {
    const value = record[key];
    if (typeof value === 'string') {
      return value.trim().split(/\s+/)[0] ?? fallback;
    }
    if (Array.isArray(value)) {
      const className = value.find((entry): entry is string => typeof entry === 'string');
      if (className) {
        return className.trim().split(/\s+/)[0] ?? fallback;
      }
    }
  }
  return fallback;
}

function hasAnyKey(record: Record<string, unknown>, keys: readonly string[]): boolean {
  return keys.some((key) => record[key] !== undefined);
}

function hasStringLike(record: Record<string, unknown>, keys: readonly string[]): boolean {
  return keys.some((key) => {
    const value = record[key];
    return (
      (typeof value === 'string' && value.trim().length > 0) ||
      typeof value === 'number' ||
      typeof value === 'boolean'
    );
  });
}

function normalizeIdentifier(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]/g, '');
}

function stripHtml(value: string): string {
  return value.replace(/<[^>]*>/g, ' ');
}

function compactText(source: string, maxChars: number): string {
  const compacted = source.replace(/\s+/g, ' ').trim();
  return compacted.length > maxChars ? `${compacted.slice(0, maxChars - 1)}…` : compacted;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value != null && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {};
}

function readString(value: unknown, fallback: string): string {
  return typeof value === 'string' ? value : fallback;
}

function readStringArray(value: unknown, fallback: readonly string[]): readonly string[] {
  return Array.isArray(value) && value.every((entry) => typeof entry === 'string')
    ? value
    : fallback;
}

function readPositiveInt(value: unknown, fallback: number): number {
  return Number.isInteger(value) && Number(value) > 0 ? Number(value) : fallback;
}
