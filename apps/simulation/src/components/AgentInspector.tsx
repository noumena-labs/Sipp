//////////////////////////////////////////////////////////////////////////////
//
// components/AgentInspector.tsx
//
// - Shows per-agent state from the latest snapshot: position, emotion,
//   current intent, and status. Clicking an agent highlights them in
//   the 3D scene.
//
//////////////////////////////////////////////////////////////////////////////

import type { SimulationAgentState } from 'cogent-engine/orchestrator';
import { EMOTION_GLYPH } from '../render/emoji-billboard.js';

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
    case 'wander':
      return 'wander';
    case 'move_to':
      return `move_to (${intent.target.x.toFixed(1)}, ${intent.target.z.toFixed(1)})`;
    case 'approach_agent':
      return `approach ${intent.agentId}`;
    case 'pick_up':
      return `pick_up ${intent.objectId}`;
    case 'drop':
      return 'drop';
    case 'use':
      return `use ${intent.objectId}`;
  }
}

export function AgentInspector(props: AgentInspectorProps) {
  return (
    <div className="agent-inspector glass-panel">
      <div className="panel-eyebrow">Agents</div>
      <ul>
        {props.agents.map((a) => {
          const selected = a.id === props.selectedAgentId;
          const glyph = a.emotion ? EMOTION_GLYPH[a.emotion] : ' ';
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
              <div className="agent-intent">intent: {formatIntent(a)}</div>
              {a.status ? <div className="agent-status">"{a.status}"</div> : null}
              {a.holding ? <div className="agent-holding">holding: {a.holding}</div> : null}
            </li>
          );
        })}
      </ul>
    </div>
  );
}
