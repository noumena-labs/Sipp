import type { WorldSnapshot } from '../runtime/types.js';

export interface ScoreboardProps {
  readonly snapshot: WorldSnapshot;
  readonly metaText?: string;
}

export function Scoreboard(props: ScoreboardProps) {
  const rows = props.snapshot.agents
    .map((agent) => ({
      id: agent.id,
      name: agent.name,
      deliveries: props.snapshot.game.score.deliveries[agent.id] ?? 0,
      holding: agent.holding === props.snapshot.game.bananaObjectId,
    }))
    .sort((a, b) => b.deliveries - a.deliveries || a.name.localeCompare(b.name));
  const banana = props.snapshot.objects.find((object) => object.id === props.snapshot.game.bananaObjectId);
  const holder = banana?.heldBy
    ? props.snapshot.agents.find((agent) => agent.id === banana.heldBy)?.name ?? banana.heldBy
    : 'loose';

  return (
    <div className="scoreboard glass-panel">
      <div className="scoreboard-head">
        <span className="panel-eyebrow">Banana Dash</span>
        <span className="scoreboard-banana">banana: {holder}</span>
      </div>
      <div className="scoreboard-grid">
        {rows.map((row) => (
          <div key={row.id} className={`score-row${row.holding ? ' carrying' : ''}`}>
            <span>{row.name}</span>
            <strong>{row.deliveries}</strong>
          </div>
        ))}
      </div>
      <div className="scoreboard-meta">
        {props.metaText ?? (props.snapshot.game.referee.status === 'ruling' ? 'director ruling...' : 'race in progress')}
      </div>
    </div>
  );
}
