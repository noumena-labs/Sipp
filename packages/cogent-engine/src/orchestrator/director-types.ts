//////////////////////////////////////////////////////////////////////////////
//
// director-types.ts
//
// - Generic types for the director harness public API.
// - A director config defines scenario-level guidance and named tasks.
// - Tasks declare their output shape so callers can choose fast constrained
//   selection paths or expressive text paths without JSON model responses.
//
//////////////////////////////////////////////////////////////////////////////

import type { RunStatus } from '../core/run-status.js';

export type JsonPrimitive = string | number | boolean | null;

export interface JsonObject {
  readonly [key: string]: JsonValue;
}

export type JsonArray = readonly JsonValue[];
export type JsonValue = JsonPrimitive | JsonObject | JsonArray;

export interface DirectorScenarioConfig {
  readonly name?: string;
  readonly summary?: string;
}

export interface DirectorProfileConfig {
  readonly role: string;
  readonly objective?: string;
  readonly instructions?: readonly string[];
}

export type DirectorInputKind = 'text' | 'data' | 'image';

export interface DirectorInputSlotConfig {
  readonly kind: DirectorInputKind;
  readonly description: string;
}

export interface DirectorChoiceConfig {
  readonly id: string;
  readonly label?: string;
  readonly description?: string;
}

export interface DirectorChoice<TPayload = unknown> extends DirectorChoiceConfig {
  readonly payload?: TPayload;
}

export type DirectorChoiceSource = 'runtime' | readonly DirectorChoiceConfig[];

export interface DirectorSelectOneOutputConfig {
  readonly shape: 'select_one';
  readonly choices: DirectorChoiceSource;
}

export interface DirectorSelectManyOutputConfig {
  readonly shape: 'select_many';
  readonly choices: DirectorChoiceSource;
  readonly min?: number;
  readonly max?: number;
}

export interface DirectorSelectSlotConfig {
  readonly name: string;
  readonly description?: string;
  readonly choices: DirectorChoiceSource;
}

export interface DirectorSelectSlotsOutputConfig {
  readonly shape: 'select_slots';
  readonly slots: readonly DirectorSelectSlotConfig[];
}

export interface DirectorTextOutputConfig {
  readonly shape: 'text';
}

export interface DirectorTextWithDirectivesOutputConfig {
  readonly shape: 'text_with_directives';
  readonly directives: DirectorChoiceSource;
  readonly maxDirectives?: number;
}

export type DirectorOutputConfig =
  | DirectorSelectOneOutputConfig
  | DirectorSelectManyOutputConfig
  | DirectorSelectSlotsOutputConfig
  | DirectorTextOutputConfig
  | DirectorTextWithDirectivesOutputConfig;

export interface DirectorTaskConfig {
  readonly purpose?: string;
  readonly instructions?: readonly string[];
  readonly inputs?: readonly string[];
  readonly output: DirectorOutputConfig;
}

export interface DirectorConfig {
  readonly id: string;
  readonly scenario?: DirectorScenarioConfig;
  readonly director: DirectorProfileConfig;
  readonly inputs?: Readonly<Record<string, DirectorInputSlotConfig>>;
  readonly tasks: Readonly<Record<string, DirectorTaskConfig>>;
}

export interface DirectorRuntimeOptions {
  readonly maxOutputTokens?: number;
  readonly contextKey?: string;
}

export interface DirectorTextInput {
  readonly kind: 'text';
  readonly text: string;
}

export interface DirectorDataInput {
  readonly kind: 'data';
  readonly value: JsonValue;
}

export interface DirectorImageInput {
  readonly kind: 'image';
  readonly media: Uint8Array;
  readonly description?: string;
}

export type DirectorInputValue =
  | JsonValue
  | DirectorTextInput
  | DirectorDataInput
  | DirectorImageInput;

export interface DirectorRunRequest<TPayload = unknown> {
  readonly inputs?: Readonly<Record<string, DirectorInputValue | undefined>>;
  readonly choices?: readonly DirectorChoice<TPayload>[];
  readonly slotChoices?: Readonly<Record<string, readonly DirectorChoice<TPayload>[]>>;
  readonly directives?: readonly DirectorChoice<TPayload>[];
  readonly signal?: AbortSignal;
  readonly timeoutMs?: number;
  readonly maxOutputTokens?: number;
}

export interface DirectorSelection<TPayload = unknown> {
  readonly id: string;
  readonly label?: string;
  readonly slot?: string;
  readonly payload?: TPayload;
}

export interface DirectorRunResult<TPayload = unknown> {
  readonly status: RunStatus;
  readonly text: string;
  readonly selections: readonly DirectorSelection<TPayload>[];
  readonly rawText: string;
  readonly errorMessage?: string;
}

export interface DirectorTaskPrompt {
  readonly systemPrompt: string;
  readonly userPrompt: string;
  readonly media: readonly Uint8Array[];
  readonly grammar?: string;
}
