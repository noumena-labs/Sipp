//////////////////////////////////////////////////////////////////////////////
//
// ControlsPanel.tsx
//
// - Lets the user point at a character.json + a model URL and trigger
//   engine load. Intentionally minimal — this is an example harness, not a
//   production settings page.
//
//////////////////////////////////////////////////////////////////////////////

import type { FormEvent } from 'react';
import { useEffect, useState } from 'react';

interface ControlsPanelProps {
  readonly characterUrl: string;
  readonly modelUrl: string;
  readonly characterName?: string;
  readonly personaSummary?: string;
  readonly status: string;
  readonly busy: boolean;
  readonly loaded: boolean;
  readonly onLoad: (args: { characterUrl: string; modelUrl: string }) => void;
  readonly onReset?: () => void;
}

export function ControlsPanel({
  characterUrl,
  modelUrl,
  characterName = 'Companion',
  personaSummary = 'A warm, playful stage companion.',
  status,
  busy,
  loaded,
  onLoad,
  onReset,
}: ControlsPanelProps) {
  const [cfg, setCfg] = useState(characterUrl);
  const [model, setModel] = useState(modelUrl);
  const [expanded, setExpanded] = useState(modelUrl.trim().length === 0);

  useEffect(() => {
    if (loaded) {
      setExpanded(false);
    }
  }, [loaded]);

  const handleSubmit = (event: FormEvent): void => {
    event.preventDefault();
    if (busy || cfg.trim().length === 0 || model.trim().length === 0) {
      return;
    }
    onLoad({ characterUrl: cfg.trim(), modelUrl: model.trim() });
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
        <div className="controls-toolbar">
          <button type="submit" disabled={busy || cfg.trim().length === 0 || model.trim().length === 0}>
            {loaded ? 'Reload Model' : 'Load Model'}
          </button>
          <button
            type="button"
            className="secondary-button"
            onClick={() => setExpanded((open) => !open)}
          >
            {expanded ? 'Hide Setup' : 'Setup'}
          </button>
          {loaded && onReset ? (
            <button type="button" className="secondary-button" onClick={onReset} disabled={busy}>
              Reset Memory
            </button>
          ) : null}
        </div>

        {expanded ? (
          <div className="controls-fields">
            <label className="field-label">
              <span>character.json URL</span>
              <input
                type="text"
                value={cfg}
                onChange={(event) => setCfg(event.target.value)}
                disabled={busy}
                placeholder="/character.json"
              />
            </label>
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
        ) : null}
      </form>
    </div>
  );
}
