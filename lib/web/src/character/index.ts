/** Action schema types used to constrain character runtime outputs. */
export type {
  ActionCue,
  ActionSchema,
  ActionSpec,
} from './action-schema.js';

/** Parsed prose and action events emitted by the incremental action parser. */
export type { ActionEvent, ParsedEvent, ProseEvent } from './action-parser.js';
/** Persona fields used to render character system prompts. */
export type {
  PersonaCurrentLifeSpec,
  PersonaDialogExample,
  PersonaPersonalitySpec,
  PersonaSpec,
} from './persona.js';

/** Event bus for chat-turn and action events from a character runtime. */
export { CharacterEventBus } from './action-bus.js';

/** Loader for character runtime configs fetched from URLs. */
export {
  type CreateCharacterFromConfigUrlOptions,
  createCharacterFromConfigUrl,
} from './create-character.js';

/** Event payloads and listener types published by character runtimes. */
export type {
  CharacterEvent,
  CharacterEventKind,
  CharacterEventListener,
  ChatTurnEndEvent,
  ChatTurnStartEvent,
} from './action-bus.js';

/** Character configuration and memory-window settings. */
export type { CharacterConfig, CharacterMemoryConfig } from './character-config.js';
/** Character configuration validation error and parser. */
export {
  CharacterConfigError,
  parseCharacterConfig,
} from './character-config.js';

/** Character runtime client, run-state, choice, and chat-turn types. */
export type {
  CharacterChoice,
  CharacterRuntimeClient,
  CharacterChooseResult,
  CharacterChooseOptions,
  CharacterRuntimeOptions,
  ChatEvent,
  ChatTurn,
  RunStatus,
} from './character-agent.js';
/** Runtime that turns chat turns into character prose and actions. */
export { CharacterRuntime } from './character-agent.js';
