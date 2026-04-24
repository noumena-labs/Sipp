//////////////////////////////////////////////////////////////////////////////
//
// components/ControlsPanel.tsx
//
// - Setup + transport panel. Mirrors the avatar app pattern: review the
//   .gguf URL, click Load, then Start / Pause / Step. Character config
//   URLs are implicit (scenario-defined).
//
//////////////////////////////////////////////////////////////////////////////

export interface ControlsPanelProps {
  readonly modelUrl: string;
  readonly onModelUrlChange: (modelUrl: string) => void;
  readonly onLoad: (modelUrl: string) => void | Promise<void>;
  readonly onStart: () => void;
  readonly onPause: () => void;
  readonly onStep: () => void;
  readonly onReset: () => void;
  readonly status: string;
  readonly busy: boolean;
  readonly loaded: boolean;
  readonly running: boolean;
  readonly tick: number;
  readonly highlightStart: boolean;
}

export function ControlsPanel(props: ControlsPanelProps) {
  const trimmedModelUrl = props.modelUrl.trim();

  return (
    <div className="controls-panel glass-panel">
      <div className="panel-eyebrow">Simulation</div>
      <div className="panel-title">Banana Dash</div>

      <label className="field">
        <span>Model GGUF</span>
        <input
          type="text"
          value={props.modelUrl}
          disabled={props.busy}
          placeholder="https://…/model.gguf"
          onChange={(e) => props.onModelUrlChange(e.target.value)}
        />
      </label>

      <div className="row">
        <button
          type="button"
          disabled={props.busy || trimmedModelUrl.length === 0}
          onClick={() => props.onLoad(trimmedModelUrl)}
        >
          {props.loaded ? 'Reload' : 'Load'}
        </button>
      </div>

      <div className="row">
        <button
          type="button"
          className={props.highlightStart ? 'start-button-highlight' : undefined}
          disabled={!props.loaded || props.running}
          onClick={props.onStart}
        >
          Start
        </button>
        <button
          type="button"
          disabled={!props.loaded || !props.running}
          onClick={props.onPause}
        >
          Pause
        </button>
        <button type="button" disabled={!props.loaded} onClick={props.onStep}>
          Step
        </button>
        <button type="button" disabled={!props.loaded} onClick={props.onReset}>
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
