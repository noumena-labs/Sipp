//////////////////////////////////////////////////////////////////////////////
//
// orchestrator/index.ts
//
// - Implementation barrel for the director harness public API.
//
//////////////////////////////////////////////////////////////////////////////

export type {
  DirectorChoice,
  DirectorChoiceConfig,
  DirectorChoiceSource,
  DirectorConfig,
  DirectorDataInput,
  DirectorImageInput,
  DirectorInputKind,
  DirectorInputSlotConfig,
  DirectorInputValue,
  DirectorOutputConfig,
  DirectorProfileConfig,
  DirectorRunRequest,
  DirectorRunResult,
  DirectorRuntimeOptions,
  DirectorScenarioConfig,
  DirectorSelection,
  DirectorSelectManyOutputConfig,
  DirectorSelectOneOutputConfig,
  DirectorSelectSlotConfig,
  DirectorSelectSlotsOutputConfig,
  DirectorTaskConfig,
  DirectorTaskPrompt,
  DirectorTextInput,
  DirectorTextOutputConfig,
  DirectorTextWithDirectivesOutputConfig,
  JsonArray,
  JsonObject,
  JsonPrimitive,
  JsonValue,
} from './director-types.js';

export { DirectorConfigError, parseDirectorConfig } from './director-config.js';

export { DirectorOutputError } from './director-output.js';

export type { DirectorRuntimeEngine } from './director-runtime.js';
export { DirectorRuntime } from './director-runtime.js';

export type {
  CreateDirectorFromConfigOptions,
  CreateDirectorFromConfigUrlOptions,
} from './create-director-from-config.js';
export { createDirectorFromConfig, createDirectorFromConfigUrl } from './create-director-from-config.js';
export type { RunStatus } from '../core/run-status.js';
