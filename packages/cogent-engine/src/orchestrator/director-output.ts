import type {
  DirectorChoice,
  DirectorChoiceConfig,
  DirectorChoiceSource,
  DirectorOutputConfig,
  DirectorRunRequest,
  DirectorSelection,
} from './director-types.js';

const CHOICE_ID_RE = /^[A-Za-z0-9_.:-]+$/;
export const MAX_DIRECTOR_GRAMMAR_BYTES = 64 * 1024;

export class DirectorOutputError extends Error {
  public constructor(message: string) {
    super(message);
    this.name = 'DirectorOutputError';
  }
}

export interface ResolvedDirectorChoices<TPayload = unknown> {
  readonly choices?: readonly DirectorChoice<TPayload>[];
  readonly slotChoices?: Readonly<Record<string, readonly DirectorChoice<TPayload>[]>>;
  readonly directives?: readonly DirectorChoice<TPayload>[];
}

export interface ParsedDirectorOutput<TPayload = unknown> {
  readonly text: string;
  readonly selections: readonly DirectorSelection<TPayload>[];
}

export function resolveDirectorChoices<TPayload>(
  output: DirectorOutputConfig,
  request: DirectorRunRequest<TPayload>
): ResolvedDirectorChoices<TPayload> {
  switch (output.shape) {
    case 'select_one':
    case 'select_many':
      return {
        choices: resolveChoiceSource(output.choices, request.choices, 'choices'),
      };
    case 'select_slots': {
      const slotChoices: Record<string, readonly DirectorChoice<TPayload>[]> = {};
      for (const slot of output.slots) {
        slotChoices[slot.name] = resolveChoiceSource(
          slot.choices,
          request.slotChoices?.[slot.name],
          `slotChoices.${slot.name}`
        );
      }
      return { slotChoices };
    }
    case 'text_with_directives':
      return {
        directives: resolveChoiceSource(output.directives, request.directives, 'directives'),
      };
    case 'text':
      return {};
  }
}

export function compileDirectorOutputGrammar<TPayload>(
  output: DirectorOutputConfig,
  resolved: ResolvedDirectorChoices<TPayload>
): string | undefined {
  let grammar: string | undefined;
  switch (output.shape) {
    case 'select_one':
      grammar = `root ::= ${choiceAlternation(requireChoices(resolved.choices, 'choices'))}\n`;
      break;
    case 'select_many':
      grammar = compileSelectManyGrammar(output.min ?? 0, output.max, requireChoices(resolved.choices, 'choices'));
      break;
    case 'select_slots':
      grammar = compileSelectSlotsGrammar(output, resolved);
      break;
    case 'text':
      return undefined;
    case 'text_with_directives':
      grammar = compileTextWithDirectivesGrammar(requireChoices(resolved.directives, 'directives'));
      break;
  }
  validateDirectorGrammarSize(grammar);
  return grammar;
}

export function parseDirectorOutput<TPayload>(
  rawText: string,
  output: DirectorOutputConfig,
  resolved: ResolvedDirectorChoices<TPayload>
): ParsedDirectorOutput<TPayload> {
  switch (output.shape) {
    case 'select_one':
      return parseSelectOne(rawText, requireChoices(resolved.choices, 'choices'));
    case 'select_many':
      return parseSelectMany(
        rawText,
        requireChoices(resolved.choices, 'choices'),
        output.min ?? 0,
        output.max
      );
    case 'select_slots':
      return parseSelectSlots(rawText, output, resolved);
    case 'text':
      return parseText<TPayload>(rawText, output.maxLength);
    case 'text_with_directives':
      return parseTextWithDirectives(
        rawText,
        requireChoices(resolved.directives, 'directives'),
        output.maxDirectives,
        output.maxLength
      );
  }
}

function resolveChoiceSource<TPayload>(
  source: DirectorChoiceSource,
  runtimeChoices: readonly DirectorChoice<TPayload>[] | undefined,
  path: string
): readonly DirectorChoice<TPayload>[] {
  const choices = source === 'runtime'
    ? runtimeChoices
    : source.map((choice) => ({ ...choice })) as readonly DirectorChoice<TPayload>[];
  if (!choices || choices.length === 0) {
    throw new DirectorOutputError(`${path} must contain at least one choice.`);
  }
  validateChoices(choices, path);
  return choices;
}

function validateChoices(
  choices: readonly DirectorChoiceConfig[],
  path: string
): void {
  const seen = new Set<string>();
  for (const choice of choices) {
    if (!CHOICE_ID_RE.test(choice.id)) {
      throw new DirectorOutputError(`${path} contains invalid choice id ${JSON.stringify(choice.id)}.`);
    }
    if (seen.has(choice.id)) {
      throw new DirectorOutputError(`${path} contains duplicate choice id ${JSON.stringify(choice.id)}.`);
    }
    seen.add(choice.id);
  }
}

function requireChoices<TPayload>(
  choices: readonly DirectorChoice<TPayload>[] | undefined,
  path: string
): readonly DirectorChoice<TPayload>[] {
  if (!choices || choices.length === 0) {
    throw new DirectorOutputError(`${path} must contain at least one choice.`);
  }
  return choices;
}

function choiceAlternation(choices: readonly DirectorChoiceConfig[]): string {
  return choices.map((choice) => gbnfStringLiteral(choice.id)).join(' | ');
}

function compileSelectManyGrammar(
  min: number,
  max: number | undefined,
  choices: readonly DirectorChoiceConfig[]
): string {
  const boundedMax = Math.max(min, max ?? choices.length);
  const lines = [
    `root ::= ${min === 0 ? '"" | ' : ''}selection-line${boundedMax > 1 ? ` (linebreak selection-line){${Math.max(0, min - 1)},${boundedMax - 1}}` : ''}`,
    `selection-line ::= ${choiceAlternation(choices)}`,
    'linebreak ::= "\\n"',
  ];
  return lines.join('\n') + '\n';
}

function compileSelectSlotsGrammar<TPayload>(
  output: Extract<DirectorOutputConfig, { shape: 'select_slots' }>,
  resolved: ResolvedDirectorChoices<TPayload>
): string {
  const lines: string[] = [];
  const rootParts: string[] = [];
  for (const [index, slot] of output.slots.entries()) {
    const ruleName = `slot${index}-choice`;
    const choices = requireChoices(resolved.slotChoices?.[slot.name], `slotChoices.${slot.name}`);
    lines.push(`${ruleName} ::= ${choiceAlternation(choices)}`);
    if (index > 0) {
      rootParts.push('linebreak');
    }
    rootParts.push(`${gbnfStringLiteral(`${slot.name}=`)} ${ruleName}`);
  }
  lines.unshift(`root ::= ${rootParts.join(' ')}`);
  lines.push('linebreak ::= "\\n"');
  return lines.join('\n') + '\n';
}

function compileTextWithDirectivesGrammar(
  directives: readonly DirectorChoiceConfig[]
): string {
  return [
    'root ::= ( directive-cue | prose-char )+',
    'prose-char ::= [^[]',
    'directive-cue ::= "[" directive-id "]"',
    `directive-id ::= ${choiceAlternation(directives)}`,
  ].join('\n') + '\n';
}

function parseSelectOne<TPayload>(
  rawText: string,
  choices: readonly DirectorChoice<TPayload>[]
): ParsedDirectorOutput<TPayload> {
  const id = rawText.trim();
  const choice = choices.find((entry) => entry.id === id);
  if (!choice) {
    throw new DirectorOutputError(`selection ${JSON.stringify(id)} did not match any available choice.`);
  }
  return { text: '', selections: [selectionFromChoice(choice)] };
}

function parseSelectMany<TPayload>(
  rawText: string,
  choices: readonly DirectorChoice<TPayload>[],
  min: number,
  max: number | undefined
): ParsedDirectorOutput<TPayload> {
  const trimmed = rawText.trim();
  const ids = trimmed.length === 0
    ? []
    : trimmed.split(/\r?\n/).map((line) => line.trim()).filter((line) => line.length > 0);
  const boundedMax = max ?? choices.length;
  if (ids.length < min || ids.length > boundedMax) {
    throw new DirectorOutputError(`selection count must be between ${min} and ${boundedMax}.`);
  }
  const seen = new Set<string>();
  const selections: DirectorSelection<TPayload>[] = [];
  for (const id of ids) {
    if (seen.has(id)) {
      throw new DirectorOutputError(`selection ${JSON.stringify(id)} was repeated.`);
    }
    seen.add(id);
    const choice = choices.find((entry) => entry.id === id);
    if (!choice) {
      throw new DirectorOutputError(`selection ${JSON.stringify(id)} did not match any available choice.`);
    }
    selections.push(selectionFromChoice(choice));
  }
  return { text: '', selections };
}

function parseSelectSlots<TPayload>(
  rawText: string,
  output: Extract<DirectorOutputConfig, { shape: 'select_slots' }>,
  resolved: ResolvedDirectorChoices<TPayload>
): ParsedDirectorOutput<TPayload> {
  const lines = rawText.trim().split(/\r?\n/).map((line) => line.trim()).filter((line) => line.length > 0);
  const bySlot = new Map<string, string>();
  for (const line of lines) {
    const match = line.match(/^([A-Za-z0-9_-]+)=([A-Za-z0-9_.:-]+)$/);
    if (!match) {
      throw new DirectorOutputError(`slot selection line ${JSON.stringify(line)} must use slot=choice format.`);
    }
    const slot = match[1]!;
    const id = match[2]!;
    if (bySlot.has(slot)) {
      throw new DirectorOutputError(`slot ${JSON.stringify(slot)} was repeated.`);
    }
    bySlot.set(slot, id);
  }

  const selections: DirectorSelection<TPayload>[] = [];
  for (const slot of output.slots) {
    const id = bySlot.get(slot.name);
    if (!id) {
      throw new DirectorOutputError(`slot ${JSON.stringify(slot.name)} is required.`);
    }
    const choice = resolved.slotChoices?.[slot.name]?.find((entry) => entry.id === id);
    if (!choice) {
      throw new DirectorOutputError(
        `selection ${JSON.stringify(id)} did not match any choice for slot ${JSON.stringify(slot.name)}.`
      );
    }
    selections.push(selectionFromChoice(choice, slot.name));
    bySlot.delete(slot.name);
  }
  const extra = bySlot.keys().next().value;
  if (extra) {
    throw new DirectorOutputError(`unknown slot ${JSON.stringify(extra)}.`);
  }
  return { text: '', selections };
}

function parseText<TPayload>(
  rawText: string,
  maxLength: number | undefined
): ParsedDirectorOutput<TPayload> {
  const text = rawText.trim();
  validateTextLength(text, maxLength);
  return { text, selections: [] };
}

function parseTextWithDirectives<TPayload>(
  rawText: string,
  directives: readonly DirectorChoice<TPayload>[],
  maxDirectives: number | undefined,
  maxLength: number | undefined
): ParsedDirectorOutput<TPayload> {
  const selections: DirectorSelection<TPayload>[] = [];
  const text = rawText.replace(/\[([A-Za-z0-9_.:-]+)\]/g, (_full, id: string) => {
    const directive = directives.find((entry) => entry.id === id);
    if (!directive) {
      throw new DirectorOutputError(`directive ${JSON.stringify(id)} did not match any available directive.`);
    }
    selections.push(selectionFromChoice(directive));
    return '';
  }).replace(/[ \t]+\n/g, '\n').replace(/\n{3,}/g, '\n\n').trim();
  if (text.includes('[')) {
    throw new DirectorOutputError('text contains an unknown or malformed directive cue.');
  }
  if (maxDirectives != null && selections.length > maxDirectives) {
    throw new DirectorOutputError(`directive count must be at most ${maxDirectives}.`);
  }
  validateTextLength(text, maxLength);
  return { text, selections };
}

function validateTextLength(text: string, maxLength: number | undefined): void {
  if (maxLength != null && text.length > maxLength) {
    throw new DirectorOutputError(`text must be at most ${maxLength} characters.`);
  }
}

function selectionFromChoice<TPayload>(
  choice: DirectorChoice<TPayload>,
  slot?: string
): DirectorSelection<TPayload> {
  return {
    id: choice.id,
    ...(choice.label ? { label: choice.label } : {}),
    ...(slot ? { slot } : {}),
    ...(choice.payload !== undefined ? { payload: choice.payload } : {}),
  };
}

function gbnfStringLiteral(source: string): string {
  return JSON.stringify(source);
}

function validateDirectorGrammarSize(grammar: string): void {
  const byteLength = typeof TextEncoder !== 'undefined'
    ? new TextEncoder().encode(grammar).byteLength
    : grammar.length;
  if (byteLength > MAX_DIRECTOR_GRAMMAR_BYTES) {
    throw new DirectorOutputError(
      `director grammar exceeds maximum size of ${MAX_DIRECTOR_GRAMMAR_BYTES} bytes (got ${byteLength}).`
    );
  }
}
