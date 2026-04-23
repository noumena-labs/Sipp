//////////////////////////////////////////////////////////////////////////////
//
// agent-grammar.ts
//
// - Hand-authored GBNF grammar constraining SimulationAgent output to a
//   single JSON object of the form:
//
//     { "intent": { "kind": "...", ...payload }, "status": "..." }
//
//   The `emotion` field inside each intent is one of the fixed
//   SIMULATION_ACTION_NAMES. Numbers are signed decimals and bounded in
//   grammar only by length; the reducer clamps them to the world bounds.
//
//////////////////////////////////////////////////////////////////////////////

import type { AgentIntent, SimulationActionName } from './simulation-types.js';
import {
  DEFAULT_SIMULATION_EMOTION,
  SIMULATION_ACTION_NAMES,
  isSimulationActionName,
} from './simulation-character-actions.js';

export interface AgentOutput {
  readonly intent: AgentIntent;
  readonly status: string;
}

/** Returns the static GBNF source for agent responses. */
export function getAgentGrammar(): string {
  return AGENT_GRAMMAR;
}

/**
 * Parses a (possibly partial / trailing-whitespace) JSON string emitted by
 * the LLM into a validated AgentOutput. Returns `null` when the payload is
 * malformed or fails schema validation — callers fall back to a default.
 */
export function parseAgentOutput(raw: string): AgentOutput | null {
  const trimmed = raw.trim();
  if (trimmed.length === 0) {
    return null;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    return null;
  }
  if (!isRecord(parsed)) {
    return null;
  }
  const intent = coerceIntent(parsed.intent);
  if (!intent) {
    return null;
  }
  const status =
    typeof parsed.status === 'string' ? parsed.status.slice(0, 120) : '';
  return { intent, status };
}

/** Default action when parsing fails or the agent has nothing to say. */
export function defaultAgentOutput(reason = 'confused'): AgentOutput {
  return {
    intent: { kind: 'wait', emotion: DEFAULT_SIMULATION_EMOTION, reason },
    status: '',
  };
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function coerceEmotion(value: unknown): SimulationActionName {
  return isSimulationActionName(value) ? value : DEFAULT_SIMULATION_EMOTION;
}

function coerceVec2(value: unknown): { x: number; z: number } | null {
  if (!isRecord(value)) return null;
  const x = value.x;
  const z = value.z;
  if (typeof x !== 'number' || !Number.isFinite(x)) return null;
  if (typeof z !== 'number' || !Number.isFinite(z)) return null;
  return { x, z };
}

function coerceIntent(value: unknown): AgentIntent | null {
  if (!isRecord(value)) return null;
  const kind = value.kind;
  const emotion = coerceEmotion(value.emotion);
  switch (kind) {
    case 'wait': {
      const reason = typeof value.reason === 'string' ? value.reason.slice(0, 80) : undefined;
      return reason ? { kind: 'wait', emotion, reason } : { kind: 'wait', emotion };
    }
    case 'wander':
      return { kind: 'wander', emotion };
    case 'move_to': {
      const target = coerceVec2(value.target);
      if (!target) return null;
      return { kind: 'move_to', target, emotion };
    }
    case 'approach_agent': {
      if (typeof value.agentId !== 'string' || value.agentId.length === 0) return null;
      return { kind: 'approach_agent', agentId: value.agentId, emotion };
    }
    case 'pick_up': {
      if (typeof value.objectId !== 'string' || value.objectId.length === 0) return null;
      return { kind: 'pick_up', objectId: value.objectId, emotion };
    }
    case 'drop':
      return { kind: 'drop', emotion };
    case 'use': {
      if (typeof value.objectId !== 'string' || value.objectId.length === 0) return null;
      return { kind: 'use', objectId: value.objectId, emotion };
    }
    default:
      return null;
  }
}

// ---------------------------------------------------------------------------
// GBNF source
// ---------------------------------------------------------------------------

function buildEmotionRule(): string {
  // emotion ::= "\"thinking\"" | "\"curious\"" | ...
  return SIMULATION_ACTION_NAMES.map((name) => `"\\"${name}\\""`).join(' | ');
}

const AGENT_GRAMMAR = `root ::= "{" ws "\\"intent\\"" ws ":" ws intent ws "," ws "\\"status\\"" ws ":" ws status-string ws "}"

intent ::= "{" ws "\\"kind\\"" ws ":" ws intent-kind ws intent-tail ws "}"

intent-kind ::= "\\"wait\\"" | "\\"wander\\"" | "\\"move_to\\"" | "\\"approach_agent\\"" | "\\"pick_up\\"" | "\\"drop\\"" | "\\"use\\""

# "intent-tail" covers every payload variant. The model must still emit a
# valid kind+payload pairing; the reducer validates semantic correctness.
intent-tail ::= ("," ws field)+

field ::= emotion-field | target-field | agent-id-field | object-id-field | reason-field

emotion-field ::= "\\"emotion\\"" ws ":" ws emotion
target-field ::= "\\"target\\"" ws ":" ws vec2
agent-id-field ::= "\\"agentId\\"" ws ":" ws short-string
object-id-field ::= "\\"objectId\\"" ws ":" ws short-string
reason-field ::= "\\"reason\\"" ws ":" ws short-string

emotion ::= ${buildEmotionRule()}

vec2 ::= "{" ws "\\"x\\"" ws ":" ws number ws "," ws "\\"z\\"" ws ":" ws number ws "}"

number ::= "-"? [0-9] [0-9]* ("." [0-9]+)?

short-string ::= "\\"" short-char{0,48} "\\""
short-char ::= [a-zA-Z0-9_ .,!?-]

status-string ::= "\\"" short-char{0,80} "\\""

ws ::= [ \\t\\n]*
`;
