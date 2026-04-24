import type { BrainActivityStoreSnapshot } from '../runtime/brain-activity-store.js';

export interface BrainTraceDrawerProps {
  readonly activity: BrainActivityStoreSnapshot;
  readonly selectedBrainId: string | null;
  readonly onClose: () => void;
}

export function BrainTraceDrawer(props: BrainTraceDrawerProps) {
  const brain = props.selectedBrainId
    ? props.activity.brains.find((entry) => entry.brainId === props.selectedBrainId) ?? null
    : null;

  return (
    <>
      <button
        type="button"
        className={`brain-drawer-backdrop${brain ? ' open' : ''}`}
        onClick={props.onClose}
        aria-label="Close brain trace drawer"
      />
      <aside className={`brain-drawer glass-panel${brain ? ' open' : ''}`} aria-hidden={!brain}>
        {brain ? (
          <>
            <div className="brain-drawer-head">
              <div>
                <span className="panel-eyebrow">Latest Snapshot</span>
                <h2>{brain.label}</h2>
                <div className="brain-drawer-subhead">
                  {brain.kind === 'director' ? 'Orchestrator harness' : 'Character harness'}
                </div>
              </div>
              <button type="button" className="brain-drawer-close" onClick={props.onClose}>
                Close
              </button>
            </div>

            <div className="brain-trace-metrics">
              <DrawerMetric label="Status" value={formatStatus(brain.status)} />
              <DrawerMetric label="Query" value={brain.queryType ?? 'n/a'} />
              <DrawerMetric label="Latency" value={formatMs(brain.elapsedMs)} />
              <DrawerMetric label="TTFT" value={formatMs(brain.ttftMs)} />
              <DrawerMetric label="Input" value={formatCount(brain.inputTokenCount)} />
              <DrawerMetric label="Output" value={formatCount(brain.outputTokenCount)} />
            </div>

            <div className="brain-trace-scroll">
              <TraceSection title="Prompt snapshot" body={brain.renderedPrompt} />
              <TraceSection title="System prompt" body={brain.systemPrompt} />
              <TraceSection title="User prompt" body={brain.userPrompt} />
              <TraceSection title="Response snapshot" body={brain.responseText} live={brain.status === 'running'} />
              {brain.grammar ? <TraceSection title="Grammar" body={brain.grammar} /> : null}
              {brain.errorMessage ? <TraceSection title="Error" body={brain.errorMessage} tone="error" /> : null}
            </div>
          </>
        ) : null}
      </aside>
    </>
  );
}

function DrawerMetric(props: { label: string; value: string }) {
  return (
    <div className="brain-trace-metric-card">
      <span className="brain-trace-metric-label">{props.label}</span>
      <strong className="brain-trace-metric-value">{props.value}</strong>
    </div>
  );
}

function TraceSection(props: {
  title: string;
  body: string;
  live?: boolean;
  tone?: 'default' | 'error';
}) {
  return (
    <section className={`brain-trace-section${props.tone === 'error' ? ' error' : ''}`}>
      <div className="brain-trace-section-head">
        <span className="brain-trace-section-title">{props.title}</span>
        {props.live ? <span className="brain-trace-live">Streaming</span> : null}
      </div>
      <pre className="brain-trace-block">{props.body.trim().length > 0 ? props.body : '(empty)'}</pre>
    </section>
  );
}

function formatStatus(status: BrainActivityStoreSnapshot['brains'][number]['status']): string {
  switch (status) {
    case 'timed_out':
      return 'timed out';
    default:
      return status;
  }
}

function formatCount(value: number | null): string {
  return value == null ? 'n/a' : String(value);
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
