//////////////////////////////////////////////////////////////////////////////
//
// components/StartPanel.tsx
//
// - Pre-simulation start screen. Lets the user choose the local GGUF model
//   before entering the Banana Dash simulation.
//
//////////////////////////////////////////////////////////////////////////////

export interface StartPanelProps {
  readonly modelUrl: string;
  readonly onModelUrlChange: (modelUrl: string) => void;
  readonly onLoad: (modelUrl: string) => void | Promise<void>;
  readonly status: string;
  readonly busy: boolean;
}

export function StartPanel(props: StartPanelProps) {
  const trimmedModelUrl = props.modelUrl.trim();

  return (
    <div className="start-panel glass-panel">
      <div className="panel-eyebrow">Simulation</div>
      <h1 className="start-title">Banana Dash</h1>
      <p className="start-copy">
        Watch four agents and one director run a fast Banana Dash match. A local LLM acts as their decision brain,
        with every choice funneled through the model for consistent monitoring, decision-making, and interaction at
        fast, low latency.
      </p>

      <label className="field start-field">
        <span>Model GGUF</span>
        <input
          type="text"
          value={props.modelUrl}
          disabled={props.busy}
          placeholder="https://.../model.gguf"
          onChange={(e) => props.onModelUrlChange(e.target.value)}
        />
      </label>

      <button
        type="button"
        className="start-load-button"
        disabled={props.busy || trimmedModelUrl.length === 0}
        onClick={() => props.onLoad(trimmedModelUrl)}
      >
        Load
      </button>

      <div className="status start-status">{props.status}</div>
    </div>
  );
}
