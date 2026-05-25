//////////////////////////////////////////////////////////////////////////////
//
// character/index.ts
//
// - Barrel export for the character subpath.
// - Everything needed to stand up a character-driven runtime ships from
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

export { CharacterEventBus } from './action-bus.js';

export {
  type CreateCharacterFromConfigUrlOptions,
  createCharacterFromConfigUrl,
} from './create-character.js';

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
  CharacterRuntimeEngine,
  CharacterChooseResult,
  CharacterChooseOptions,
  CharacterRuntimeOptions,
  ChatEvent,
  ChatTurn,
  RunStatus,
} from './character-agent.js';
export { CharacterRuntime } from './character-agent.js';
