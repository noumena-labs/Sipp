//////////////////////////////////////////////////////////////////////////////
//
// director-types.ts
//
// - Generic types for the `cogent-engine/orchestrator` subpath.
// - This package is intentionally query/runtime-focused: it loads
//   `director.json`, renders prompts from app-supplied state, compiles a
//   matching response grammar, and runs the model through the core engine.
//
//////////////////////////////////////////////////////////////////////////////

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

export interface ResponseSchemaBase {
  readonly description?: string;
  readonly nullable?: boolean;
}

export interface ResponseStringSchema extends ResponseSchemaBase {
  readonly type: 'string';
  readonly maxLength?: number;
  readonly enum?: readonly string[];
}

export interface ResponseNumberSchema extends ResponseSchemaBase {
  readonly type: 'number';
  readonly integer?: boolean;
}

export interface ResponseBooleanSchema extends ResponseSchemaBase {
  readonly type: 'boolean';
}

export interface ResponseNullSchema {
  readonly type: 'null';
}

export interface ResponseArraySchema extends ResponseSchemaBase {
  readonly type: 'array';
  readonly items: ResponseSchema;
  readonly maxItems?: number;
}

export interface ResponseObjectSchema extends ResponseSchemaBase {
  readonly type: 'object';
  readonly properties: Readonly<Record<string, ResponseSchema>>;
}

export type ResponseSchema =
  | ResponseStringSchema
  | ResponseNumberSchema
  | ResponseBooleanSchema
  | ResponseNullSchema
  | ResponseArraySchema
  | ResponseObjectSchema;

export interface DirectorQueryConfig {
  readonly description?: string;
  readonly instructions?: readonly string[];
  readonly hooks?: readonly string[];
  readonly response: ResponseSchema;
}

export interface DirectorConfig {
  readonly id: string;
  readonly scenario?: DirectorScenarioConfig;
  readonly director: DirectorProfileConfig;
  readonly hooks?: Readonly<Record<string, string>>;
  readonly queries: Readonly<Record<string, DirectorQueryConfig>>;
}

export interface DirectorRuntimeOptions {
  readonly maxOutputTokens?: number;
  readonly contextKey?: string;
}

export interface DirectorQueryPayload {
  readonly [sectionName: string]: JsonValue | undefined;
}

export interface DirectorQueryResult {
  readonly data: JsonValue | null;
  readonly cancelled: boolean;
  readonly errorMessage?: string;
  readonly rawText: string;
}

export interface DirectorQueryOptions {
  readonly signal?: AbortSignal;
  readonly timeoutMs?: number;
}
