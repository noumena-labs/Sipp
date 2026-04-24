import type { CSSProperties } from 'react';
import type { BrainActivityStoreSnapshot } from '../runtime/brain-activity-store.js';

export interface BrainActivityHudProps {
  readonly activity: BrainActivityStoreSnapshot;
  readonly selectedBrainId: string | null;
  readonly onSelectBrain: (brainId: string) => void;
}

export function BrainActivityHud(props: BrainActivityHudProps) {
  const activeBrain = props.activity.brains.find((brain) => brain.status === 'running') ?? null;
  const streamPreview = activeBrain?.responseText.trim() || previewPrompt(activeBrain?.renderedPrompt ?? '');

  return (
    <div className="brain-hud glass-panel">
      <div className="brain-hud-head">
        <div>
          <span className="panel-eyebrow">Brain Activity</span>
          <div className="brain-hud-title">LLM query visualizer</div>
        </div>
        <span className={`brain-hud-status${activeBrain ? ' active' : ''}`}>
          {activeBrain ? `${activeBrain.label} live` : 'Idle'}
        </span>
      </div>

      <div className="brain-metric-grid">
        <Metric label="Total queries" value={String(props.activity.totalQueries)} />
        <Metric label="Queries / sec" value={props.activity.queriesPerSecond.toFixed(2)} />
        <Metric label="Last latency" value={formatMs(props.activity.lastLatencyMs)} />
        <Metric label="Failures" value={String(props.activity.totalFailures)} />
      </div>

      <div className="brain-stream-band">
        <div className="brain-stream-label">
          {activeBrain ? `${activeBrain.label} streaming` : 'Latest brain snapshot'}
        </div>
        <div className="brain-stream-text">{streamPreview || 'No active query yet.'}</div>
      </div>

      <div className="brain-grid">
        {props.activity.brains.map((brain) => {
          const selected = brain.brainId === props.selectedBrainId;
          const running = brain.status === 'running';
          const style = { '--brain-accent': brain.accentColor } as CSSProperties;
          return (
            <button
              key={brain.brainId}
              type="button"
              className={`brain-chip brain-status-${brain.status}${selected ? ' selected' : ''}${running ? ' running' : ''}`}
              style={style}
              onClick={() => props.onSelectBrain(brain.brainId)}
            >
              <span className="brain-chip-head">
                <span className="brain-chip-orb" />
                <span className="brain-chip-name">{brain.label}</span>
                <span className="brain-chip-kind">{brain.kind === 'director' ? 'orch' : 'char'}</span>
              </span>
              <span className="brain-chip-status">{formatStatus(brain.status, brain.queryType)}</span>
              <span className="brain-chip-meta">
                {brain.elapsedMs != null ? formatMs(brain.elapsedMs) : 'waiting'}
                {brain.tick != null ? ` • #${brain.tick}` : ''}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function Metric(props: { label: string; value: string }) {
  return (
    <div className="brain-metric-card">
      <span className="brain-metric-label">{props.label}</span>
      <strong className="brain-metric-value">{props.value}</strong>
    </div>
  );
}

function formatStatus(
  status: BrainActivityStoreSnapshot['brains'][number]['status'],
  queryType: BrainActivityStoreSnapshot['brains'][number]['queryType']
): string {
  const queryLabel = queryType == null ? 'standby' : queryType.replace('_', ' ');
  switch (status) {
    case 'running':
      return `${queryLabel} live`;
    case 'completed':
      return `${queryLabel} complete`;
    case 'cancelled':
      return `${queryLabel} cancelled`;
    case 'timed_out':
      return `${queryLabel} timed out`;
    case 'failed':
      return `${queryLabel} failed`;
    default:
      return queryLabel;
  }
}

function previewPrompt(text: string): string {
  const compact = text.replace(/\s+/g, ' ').trim();
  return compact.slice(0, 140);
}

function formatMs(value: number | null): string {
  if (value == null) {
    return 'n/a';
  }
  if (value >= 1000) {
    return `${(value / 1000).toFixed(2)}s`;
  }
  return `${Math.round(value)}ms`;
}
