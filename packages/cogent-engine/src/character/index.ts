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
  ActionCue,
  ActionSchema,
  ActionSpec,
} from './action-schema.js';

export type { ActionEvent, ParsedEvent, ProseEvent } from './action-parser.js';
export type {
  PersonaCurrentLifeSpec,
  PersonaDialogExample,
  PersonaPersonalitySpec,
  PersonaSpec,
} from './persona.js';

export { ActionBus } from './action-bus.js';

export { type CreateCharacterFromConfigUrlOptions, createCharacterFromConfigUrl } from './create-character.js';

export type {
  CharacterEvent,
  CharacterEventKind,
  CharacterEventListener,
  ChatTurnEndEvent,
  ChatTurnStartEvent,
} from './action-bus.js';

export type { CharacterConfig, CharacterMemoryConfig } from './character-config.js';
export {
  CharacterConfigError,
  parseCharacterConfig,
} from './character-config.js';

export type {
  CharacterAgentEngine,
  CharacterAgentOptions,
  ChatEvent,
  ChatTurn,
} from './character-agent.js';
export { CharacterAgent } from './character-agent.js';
