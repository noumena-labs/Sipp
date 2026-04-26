//////////////////////////////////////////////////////////////////////////////
//
// ControlsPanel.tsx
//
// - Lets the user point at a model URL and trigger engine load. Character
//   config is owned internally by the demo.
//
//////////////////////////////////////////////////////////////////////////////

import type { FormEvent } from 'react';
import { useEffect, useState } from 'react';

interface ControlsPanelProps {
  readonly modelUrl: string;
  readonly characterName?: string;
  readonly personaSummary?: string;
  readonly status: string;
  readonly busy: boolean;
  readonly loaded: boolean;
  readonly onLoad: (args: { modelUrl: string }) => void;
  readonly onReset?: () => void;
}

export function ControlsPanel({
  modelUrl,
  characterName = 'Companion',
  personaSummary = 'A warm, playful stage companion.',
  status,
  busy,
  loaded,
  onLoad,
  onReset,
}: ControlsPanelProps) {
  const [model, setModel] = useState(modelUrl);

  useEffect(() => {
    setModel(modelUrl);
  }, [modelUrl]);

  const handleSubmit = (event: FormEvent): void => {
    event.preventDefault();
    if (busy || model.trim().length === 0) {
      return;
    }
    onLoad({ modelUrl: model.trim() });
  };

  return (
    <div className="controls-panel glass-panel">
      <div className="controls-header">
        <div className="controls-copy">
          <span className="panel-eyebrow">Avatar Setup</span>
          <h1>{characterName}</h1>
          <p>{personaSummary}</p>
        </div>
        <div className={`status-pill ${busy ? 'busy' : loaded ? 'ready' : 'idle'}`}>
          {busy ? 'Loading' : loaded ? 'Live' : 'Standby'}
        </div>
      </div>

      <div className="status-line">{status}</div>

      <form className="controls-form" onSubmit={handleSubmit}>
        <div className="controls-fields">
          <label className="field-label">
            <span>Model (.gguf) URL</span>
            <input
              type="url"
              value={model}
              onChange={(event) => setModel(event.target.value)}
              disabled={busy}
              placeholder="https://huggingface.co/.../model.gguf"
            />
          </label>
        </div>

        <div className="controls-toolbar">
          <button type="submit" disabled={busy || model.trim().length === 0}>
            {loaded ? 'Reload Model' : 'Load Model'}
          </button>
          {loaded && onReset ? (
            <button type="button" className="secondary-button" onClick={onReset} disabled={busy}>
              Reset Memory
            </button>
          ) : null}
        </div>
      </form>
    </div>
  );
}
