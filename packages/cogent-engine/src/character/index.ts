//////////////////////////////////////////////////////////////////////////////
//
// character/index.ts
//
// - Barrel export for the `cogent-engine/character` subpath.
// - Everything needed to stand up a character-driven chat loop ships from
//   this single entry point.
//
//////////////////////////////////////////////////////////////////////////////

export type {
  ActionArgSpec,
  ActionArgType,
  ActionSchema,
  ActionSpec,
} from './action-schema.js';
export { renderActionSchemaForPrompt, validateActionSchema } from './action-schema.js';

export { ActionSchemaError, compileActionGrammar } from './action-grammar.js';

export type { ActionEvent, ParsedEvent, ProseEvent } from './action-parser.js';
export {
  ActionParseError,
  StreamingActionParser,
  parseActionTag,
} from './action-parser.js';

export type {
  CharacterEvent,
  CharacterEventKind,
  CharacterEventListener,
  ChatTurnEndEvent,
  ChatTurnStartEvent,
} from './action-bus.js';
export { ActionBus } from './action-bus.js';

export type { PersonaSpec } from './persona.js';
export { renderSystemPrompt } from './persona.js';

export type {
  CharacterAssets,
  CharacterConfig,
  CharacterMemoryConfig,
} from './character-config.js';
export {
  CharacterConfigError,
  DEFAULT_MEMORY_MAX_TURNS,
  parseCharacterConfig,
  resolveMaxMemoryTurns,
} from './character-config.js';

export type {
  CharacterAgentEngine,
  CharacterAgentOptions,
  ChatEvent,
  ChatTurn,
} from './character-agent.js';
export { CharacterAgent } from './character-agent.js';
