//////////////////////////////////////////////////////////////////////////////
//
// director-config.ts
//
// - Parser/validator for `director.json`.
// - The director config names high-level tasks and their output shapes.
//
//////////////////////////////////////////////////////////////////////////////

import type {
  DirectorChoiceConfig,
  DirectorChoiceSource,
  DirectorConfig,
  DirectorInputSlotConfig,
  DirectorOutputConfig,
  DirectorProfileConfig,
  DirectorScenarioConfig,
  DirectorSelectSlotConfig,
  DirectorTaskConfig,
} from './director-types.js';

const NAME_RE = /^[A-Za-z0-9_-]+$/;
const CHOICE_ID_RE = /^[A-Za-z0-9_.:-]+$/;

export class DirectorConfigError extends Error {
  public constructor(message: string) {
    super(message);
    this.name = 'DirectorConfigError';
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function parseOptionalString(value: unknown, path: string): string | undefined {
  if (value == null) {
    return undefined;
  }
  if (typeof value !== 'string') {
    throw new DirectorConfigError(`\`${path}\` must be a string if present.`);
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

function parseRequiredString(value: unknown, path: string): string {
  const parsed = parseOptionalString(value, path);
  if (!parsed) {
    throw new DirectorConfigError(`\`${path}\` is required and must be a non-empty string.`);
  }
  return parsed;
}

function parseOptionalStringArray(value: unknown, path: string): readonly string[] | undefined {
  if (value == null) {
    return undefined;
  }
  if (!Array.isArray(value) || value.some((entry) => typeof entry !== 'string')) {
    throw new DirectorConfigError(`\`${path}\` must be an array of strings if present.`);
  }
  const trimmed = value.map((entry) => entry.trim()).filter((entry) => entry.length > 0);
  return trimmed.length > 0 ? trimmed : undefined;
}

function parseOptionalNonNegativeInteger(value: unknown, path: string): number | undefined {
  if (value == null) {
    return undefined;
  }
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || Math.floor(value) !== value) {
    throw new DirectorConfigError(`\`${path}\` must be a non-negative integer if present.`);
  }
  return value;
}

function parsePositiveInteger(value: unknown, path: string): number | undefined {
  const parsed = parseOptionalNonNegativeInteger(value, path);
  if (parsed === 0) {
    throw new DirectorConfigError(`\`${path}\` must be greater than zero if present.`);
  }
  return parsed;
}

function parseScenario(raw: unknown): DirectorScenarioConfig | undefined {
  if (raw == null) {
    return undefined;
  }
  if (!isRecord(raw)) {
    throw new DirectorConfigError('`scenario` must be an object if present.');
  }
  return {
    name: parseOptionalString(raw.name, 'scenario.name'),
    summary: parseOptionalString(raw.summary, 'scenario.summary'),
  } satisfies DirectorScenarioConfig;
}

function parseDirectorProfile(raw: unknown): DirectorProfileConfig {
  if (!isRecord(raw)) {
    throw new DirectorConfigError('`director` must be an object.');
  }
  return {
    role: parseRequiredString(raw.role, 'director.role'),
    objective: parseOptionalString(raw.objective, 'director.objective'),
    instructions: parseOptionalStringArray(raw.instructions, 'director.instructions'),
  } satisfies DirectorProfileConfig;
}

function parseInputs(raw: unknown): Readonly<Record<string, DirectorInputSlotConfig>> | undefined {
  if (raw == null) {
    return undefined;
  }
  if (!isRecord(raw)) {
    throw new DirectorConfigError('`inputs` must be an object mapping input names to input slot configs.');
  }
  const entries: Array<[string, DirectorInputSlotConfig]> = [];
  for (const [name, value] of Object.entries(raw)) {
    if (!NAME_RE.test(name)) {
      throw new DirectorConfigError(`input name ${JSON.stringify(name)} must match [A-Za-z0-9_-]+.`);
    }
    if (!isRecord(value)) {
      throw new DirectorConfigError(`\`inputs.${name}\` must be an object.`);
    }
    const kind = value.kind;
    if (kind !== 'text' && kind !== 'data' && kind !== 'image') {
      throw new DirectorConfigError(`\`inputs.${name}.kind\` must be text, data, or image.`);
    }
    entries.push([
      name,
      {
        kind,
        description: parseRequiredString(value.description, `inputs.${name}.description`),
      },
    ]);
  }
  return Object.fromEntries(entries);
}

function parseChoice(raw: unknown, path: string): DirectorChoiceConfig {
  if (!isRecord(raw)) {
    throw new DirectorConfigError(`\`${path}\` must be an object.`);
  }
  const id = parseRequiredString(raw.id, `${path}.id`);
  if (!CHOICE_ID_RE.test(id)) {
    throw new DirectorConfigError(`\`${path}.id\` must match [A-Za-z0-9_.:-]+.`);
  }
  return {
    id,
    label: parseOptionalString(raw.label, `${path}.label`),
    description: parseOptionalString(raw.description, `${path}.description`),
  };
}

function parseChoiceSource(raw: unknown, path: string): DirectorChoiceSource {
  if (raw === 'runtime') {
    return 'runtime';
  }
  if (!Array.isArray(raw)) {
    throw new DirectorConfigError(`\`${path}\` must be "runtime" or an array of choices.`);
  }
  const choices = raw.map((entry, index) => parseChoice(entry, `${path}[${index}]`));
  assertUniqueChoiceIds(choices, path);
  if (choices.length === 0) {
    throw new DirectorConfigError(`\`${path}\` must contain at least one choice.`);
  }
  return choices;
}

function assertUniqueChoiceIds(choices: readonly DirectorChoiceConfig[], path: string): void {
  const seen = new Set<string>();
  for (const choice of choices) {
    if (seen.has(choice.id)) {
      throw new DirectorConfigError(`\`${path}\` contains duplicate choice id ${JSON.stringify(choice.id)}.`);
    }
    seen.add(choice.id);
  }
}

function parseSlot(raw: unknown, path: string): DirectorSelectSlotConfig {
  if (!isRecord(raw)) {
    throw new DirectorConfigError(`\`${path}\` must be an object.`);
  }
  const name = parseRequiredString(raw.name, `${path}.name`);
  if (!NAME_RE.test(name)) {
    throw new DirectorConfigError(`\`${path}.name\` must match [A-Za-z0-9_-]+.`);
  }
  return {
    name,
    description: parseOptionalString(raw.description, `${path}.description`),
    choices: parseChoiceSource(raw.choices, `${path}.choices`),
  };
}

function parseOutput(raw: unknown, path: string): DirectorOutputConfig {
  if (!isRecord(raw)) {
    throw new DirectorConfigError(`\`${path}\` must be an object.`);
  }
  const shape = raw.shape;
  switch (shape) {
    case 'select_one':
      return {
        shape,
        choices: parseChoiceSource(raw.choices, `${path}.choices`),
      };
    case 'select_many': {
      const min = parseOptionalNonNegativeInteger(raw.min, `${path}.min`);
      const max = parsePositiveInteger(raw.max, `${path}.max`);
      if (min != null && max != null && max < min) {
        throw new DirectorConfigError(`\`${path}.max\` must be greater than or equal to \`${path}.min\`.`);
      }
      return {
        shape,
        choices: parseChoiceSource(raw.choices, `${path}.choices`),
        ...(min != null ? { min } : {}),
        ...(max != null ? { max } : {}),
      };
    }
    case 'select_slots': {
      if (!Array.isArray(raw.slots) || raw.slots.length === 0) {
        throw new DirectorConfigError(`\`${path}.slots\` must contain at least one slot.`);
      }
      const slots = raw.slots.map((slot, index) => parseSlot(slot, `${path}.slots[${index}]`));
      const slotNames = new Set<string>();
      for (const slot of slots) {
        if (slotNames.has(slot.name)) {
          throw new DirectorConfigError(`\`${path}.slots\` contains duplicate slot ${JSON.stringify(slot.name)}.`);
        }
        slotNames.add(slot.name);
      }
      return { shape, slots };
    }
    case 'text':
      return {
        shape,
        minLength: parsePositiveInteger(raw.minLength, `${path}.minLength`),
        maxLength: parsePositiveInteger(raw.maxLength, `${path}.maxLength`),
      };
    case 'text_with_directives':
      return {
        shape,
        directives: parseChoiceSource(raw.directives, `${path}.directives`),
        maxDirectives: parsePositiveInteger(raw.maxDirectives, `${path}.maxDirectives`),
        minLength: parsePositiveInteger(raw.minLength, `${path}.minLength`),
        maxLength: parsePositiveInteger(raw.maxLength, `${path}.maxLength`),
      };
    default:
      throw new DirectorConfigError(`Unsupported output shape at \`${path}.shape\`: ${JSON.stringify(shape)}.`);
  }
}

function parseTask(
  name: string,
  raw: unknown,
  inputs: Readonly<Record<string, DirectorInputSlotConfig>> | undefined
): DirectorTaskConfig {
  if (!isRecord(raw)) {
    throw new DirectorConfigError(`\`tasks.${name}\` must be an object.`);
  }
  const taskInputs = parseOptionalStringArray(raw.inputs, `tasks.${name}.inputs`);
  if (taskInputs) {
    for (const inputName of taskInputs) {
      if (!inputs || !(inputName in inputs)) {
        throw new DirectorConfigError(
          `\`tasks.${name}.inputs\` references unknown input ${JSON.stringify(inputName)}.`
        );
      }
    }
  }
  return {
    purpose: parseOptionalString(raw.purpose, `tasks.${name}.purpose`),
    instructions: parseOptionalStringArray(raw.instructions, `tasks.${name}.instructions`),
    ...(taskInputs ? { inputs: taskInputs } : {}),
    output: parseOutput(raw.output, `tasks.${name}.output`),
  };
}

export function parseDirectorConfig(raw: unknown): DirectorConfig {
  if (!isRecord(raw)) {
    throw new DirectorConfigError('Director config must be a JSON object.');
  }

  if (typeof raw.id !== 'string' || !NAME_RE.test(raw.id)) {
    throw new DirectorConfigError('`id` must be a non-empty identifier ([A-Za-z0-9_-]+).');
  }

  const inputs = parseInputs(raw.inputs);
  if (!isRecord(raw.tasks)) {
    throw new DirectorConfigError('`tasks` must be an object.');
  }
  const taskEntries = Object.entries(raw.tasks);
  if (taskEntries.length === 0) {
    throw new DirectorConfigError('`tasks` must contain at least one named task.');
  }
  const tasks = Object.fromEntries(
    taskEntries.map(([name, task]) => {
      if (!NAME_RE.test(name)) {
        throw new DirectorConfigError(`task name ${JSON.stringify(name)} must match [A-Za-z0-9_-]+.`);
      }
      return [name, parseTask(name, task, inputs)];
    })
  );

  return {
    id: raw.id,
    scenario: parseScenario(raw.scenario),
    director: parseDirectorProfile(raw.director),
    ...(inputs ? { inputs } : {}),
    tasks,
  };
}
