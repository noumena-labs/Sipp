//////////////////////////////////////////////////////////////////////////////
//
// agent-glyphs.ts
//
// - Shared emoji selection for agent billboards and the agents panel.
// - Keeps steady-state activity glyphs and transient query/intent/game-event
//   glyph overrides aligned across the DOM inspector and the three.js scene.
//
//////////////////////////////////////////////////////////////////////////////

import type { AgentIntent, SimulationAgentState, SimulationGameEvent } from './runtime/types.js';
import { emotionGlyphFor } from './render/emoji-billboard.js';

export interface AgentGlyphContext {
  readonly bananaObjectId: string;
  readonly tick: number;
}

export interface AgentGlyphOverrideSpec {
  readonly agentId: string;
  readonly glyph: string;
  readonly durationSeconds: number;
}

export const QUERY_GLYPH = '...';
export const INTENT_GLYPH_OVERRIDE_SECONDS = 0.24;

export function resolveAgentGlyph(
  agent: SimulationAgentState,
  context: AgentGlyphContext,
  glyphOverride: string | null = null
): string | null {
  const glyph = glyphOverride ?? activityGlyphFor(agent, context);
  if (glyph) return glyph;
  return agent.emotion ? emotionGlyphFor(agent.emotion) : null;
}

export function activityGlyphFor(
  agent: SimulationAgentState,
  context: AgentGlyphContext
): string | null {
  if (agent.holding === context.bananaObjectId) return '🍌';
  if (agent.frozenUntilTick > context.tick) return '⛄';
  if (agent.powerUp?.kind === 'bat') return '🏏';
  if (agent.powerUp?.kind === 'ice_cube') return '🧊';
  const intent = agent.intent;
  if (!intent) return null;
  switch (intent.kind) {
    case 'go_to_object':
    case 'move_to':
      return '🏃';
    case 'pick_up':
      return '✋';
    case 'deliver':
      return '🏁';
    case 'sabotage':
      return intent.method === 'bat' ? '🏏' : intent.method === 'ice_cube' ? '🧊' : '💥';
    case 'approach_agent':
      return '👀';
    case 'push':
      return '✋';
    case 'wait':
      return '⏳';
    case 'drop':
      return '💢';
    case 'use':
      return '✨';
  }
}

export function glyphForIntent(intent: AgentIntent): string {
  switch (intent.kind) {
    case 'go_to_object':
    case 'move_to':
      return '🏃';
    case 'pick_up':
      return '✋';
    case 'deliver':
      return '🏁';
    case 'sabotage':
      return intent.method === 'bat' ? '🏏' : intent.method === 'ice_cube' ? '🧊' : '💥';
    case 'approach_agent':
      return '👀';
    case 'push':
      return '✋';
    case 'wait':
      return '⏳';
    case 'drop':
      return '💢';
    case 'use':
      return '✨';
  }
}

export function glyphOverridesForGameEvent(
  event: SimulationGameEvent
): readonly AgentGlyphOverrideSpec[] {
  switch (event.kind) {
    case 'pickup':
      return [{ agentId: event.agentId, glyph: '✋', durationSeconds: 0.28 }];
    case 'drop':
      return [{ agentId: event.agentId, glyph: '💢', durationSeconds: 0.35 }];
    case 'forced_drop':
      return [
        { agentId: event.attackerAgentId, glyph: '💥', durationSeconds: 0.32 },
        { agentId: event.targetAgentId, glyph: event.outcome === 'drop' ? '💢' : '‼', durationSeconds: 0.32 },
      ];
    case 'bump_whiff':
      return [{ agentId: event.attackerAgentId, glyph: '💨', durationSeconds: 0.28 }];
    case 'push':
      return [
        { agentId: event.agentId, glyph: '✋', durationSeconds: 0.3 },
        { agentId: event.targetAgentId, glyph: '💨', durationSeconds: 0.38 },
      ];
    case 'power_up_throw':
      return [{ agentId: event.agentId, glyph: '🧊', durationSeconds: 0.28 }];
    case 'bat_swing':
      return [
        { agentId: event.agentId, glyph: '🏏', durationSeconds: 0.42 },
        ...event.hits.map((hit) => ({ agentId: hit.agentId, glyph: '💫', durationSeconds: 0.44 })),
      ];
    case 'power_up_use':
      return [
        { agentId: event.agentId, glyph: '🧊', durationSeconds: 0.36 },
        { agentId: event.targetAgentId, glyph: '⛄', durationSeconds: 0.4 },
      ];
    case 'delivery':
      return [{ agentId: event.agentId, glyph: '🎉', durationSeconds: 0.9 }];
    case 'respawn':
    case 'fallback':
      return [];
  }
}
