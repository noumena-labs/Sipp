//////////////////////////////////////////////////////////////////////////////
//
// components/ControlsPanel.tsx
//
// - Runtime transport panel. Model setup happens on the start screen.
//
//////////////////////////////////////////////////////////////////////////////

export interface ControlsPanelProps {
  readonly onStart: () => void;
  readonly onPause: () => void;
  readonly onStep: () => void;
  readonly onReset: () => void;
  readonly status: string;
  readonly running: boolean;
  readonly tick: number;
  readonly highlightStart: boolean;
}

export function ControlsPanel(props: ControlsPanelProps) {
  return (
    <div className="controls-panel glass-panel">
      <div className="panel-eyebrow">Simulation</div>
      <div className="panel-title">Banana Dash</div>

      <div className="row">
        <button
          type="button"
          className={props.highlightStart ? 'start-button-highlight' : undefined}
          disabled={props.running}
          onClick={props.onStart}
        >
          Start
        </button>
        <button
          type="button"
          disabled={!props.running}
          onClick={props.onPause}
        >
          Pause
        </button>
        <button type="button" onClick={props.onStep}>
          Step
        </button>
        <button type="button" onClick={props.onReset}>
          Reset
        </button>
      </div>

      <div className="status">
        <div>tick #{props.tick}</div>
        <div>{props.status}</div>
      </div>
    </div>
  );
}
