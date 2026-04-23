//////////////////////////////////////////////////////////////////////////////
//
// components/AgentInspector.tsx
//
// - Shows per-agent state from the latest snapshot: position, emotion,
//   current intent, and status. Clicking an agent highlights them in
//   the 3D scene.
//
//////////////////////////////////////////////////////////////////////////////

import type { SimulationAgentState } from '../runtime/types.js';
import { emotionGlyphFor } from '../render/emoji-billboard.js';

export interface AgentInspectorProps {
  readonly agents: readonly SimulationAgentState[];
  readonly selectedAgentId: string | null;
  readonly onSelect: (agentId: string | null) => void;
}

function formatIntent(agent: SimulationAgentState): string {
  const intent = agent.intent;
  if (!intent) return '—';
  switch (intent.kind) {
    case 'wait':
      return `wait (${intent.reason ?? ''})`.trim();
    case 'move_to':
      return `move_to (${intent.target.x.toFixed(1)}, ${intent.target.z.toFixed(1)})`;
    case 'go_to_object':
      return `go_to ${intent.objectId}`;
    case 'approach_agent':
      return `approach ${intent.agentId}`;
    case 'pick_up':
      return `pick_up ${intent.objectId}`;
    case 'drop':
      return 'drop';
    case 'deliver':
      return `deliver ${intent.objectId}`;
    case 'sabotage':
      return `bump ${intent.agentId}`;
    case 'use':
      return `use ${intent.objectId}`;
  }
}

function formatGoal(agent: SimulationAgentState): string {
  return agent.goal?.label ?? 'idle';
}

function formatActivity(agent: SimulationAgentState): string {
  if (agent.thinking) return 'thinking through the next move';
  if (agent.holding === 'banana') return 'carrying banana to home base';
  const intent = agent.intent;
  if (!intent) return agent.status || 'watching quietly';
  switch (intent.kind) {
    case 'go_to_object':
      return intent.objectId === 'banana' ? 'rushing the banana' : `moving to ${intent.objectId}`;
    case 'pick_up':
      return `reaching for ${intent.objectId}`;
    case 'approach_agent':
      return `chasing ${intent.agentId}`;
    case 'sabotage':
      return `trying to bump ${intent.agentId}`;
    case 'deliver':
      return 'scoring at home base';
    case 'wait':
      return intent.reason ?? 'waiting';
    case 'drop':
      return 'dropping the banana';
    case 'move_to':
      return 'moving to a waypoint';
    case 'use':
      return `using ${intent.objectId}`;
  }
}

export function AgentInspector(props: AgentInspectorProps) {
  return (
    <div className="agent-inspector glass-panel">
      <div className="panel-eyebrow">Agents</div>
      <ul>
        {props.agents.map((a) => {
          const selected = a.id === props.selectedAgentId;
          const glyph = a.holding === 'banana' ? '🍌' : a.emotion ? emotionGlyphFor(a.emotion) : ' ';
          return (
            <li
              key={a.id}
              className={`agent-row${selected ? ' selected' : ''}`}
              onClick={() => props.onSelect(selected ? null : a.id)}
            >
              <div className="agent-head">
                <span className="agent-glyph">{glyph}</span>
                <span className="agent-name">{a.name}</span>
                <span className="agent-pos">
                  ({a.position.x.toFixed(1)}, {a.position.z.toFixed(1)})
                </span>
              </div>
              <div className="agent-activity">{formatActivity(a)}</div>
              <div className="agent-goal">goal: {a.thinking ? 'thinking...' : formatGoal(a)}</div>
              {a.intent ? <div className="agent-intent">executor: {formatIntent(a)}</div> : null}
              {a.holding ? <div className="agent-holding">holding: {a.holding}</div> : null}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
