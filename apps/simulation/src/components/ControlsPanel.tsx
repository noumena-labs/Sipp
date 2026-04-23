//////////////////////////////////////////////////////////////////////////////
//
// components/ControlsPanel.tsx
//
// - Setup + transport panel. Mirrors the avatar app pattern: paste a
//   .gguf URL, click Load, then Start / Pause / Step and tune the tick
//   rate. Character config URLs are implicit (scenario-defined).
//
//////////////////////////////////////////////////////////////////////////////

import { useState } from 'react';

export interface ControlsPanelProps {
  readonly modelUrl: string;
  readonly onLoad: (modelUrl: string) => void | Promise<void>;
  readonly onStart: () => void;
  readonly onPause: () => void;
  readonly onStep: () => void;
  readonly onReset: () => void;
  readonly tickHz: number;
  readonly onTickHzChange: (hz: number) => void;
  readonly status: string;
  readonly busy: boolean;
  readonly loaded: boolean;
  readonly running: boolean;
  readonly tick: number;
}

export function ControlsPanel(props: ControlsPanelProps) {
  const [modelUrl, setModelUrl] = useState(props.modelUrl);

  return (
    <div className="controls-panel glass-panel">
      <div className="panel-eyebrow">Simulation</div>
      <div className="panel-title">Banana Dash</div>

      <label className="field">
        <span>Model URL (.gguf)</span>
        <input
          type="text"
          value={modelUrl}
          disabled={props.busy}
          placeholder="https://…/model.gguf"
          onChange={(e) => setModelUrl(e.target.value)}
        />
      </label>

      <div className="row">
        <button
          type="button"
          disabled={props.busy || modelUrl.trim().length === 0}
          onClick={() => props.onLoad(modelUrl.trim())}
        >
          {props.loaded ? 'Reload' : 'Load'}
        </button>
      </div>

      <div className="row">
        <button
          type="button"
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

      <label className="field">
        <span>Tick rate: {props.tickHz.toFixed(2)} Hz</span>
        <input
          type="range"
          min={0.5}
          max={4}
          step={0.1}
          value={props.tickHz}
          onChange={(e) => props.onTickHzChange(parseFloat(e.target.value))}
        />
      </label>

      <div className="status">
        <div>tick #{props.tick}</div>
        <div>{props.status}</div>
      </div>
    </div>
  );
}
