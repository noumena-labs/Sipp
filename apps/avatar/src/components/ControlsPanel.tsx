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
import { useState } from 'react';

interface ControlsPanelProps {
  readonly characterUrl: string;
  readonly modelUrl: string;
  readonly status: string;
  readonly busy: boolean;
  readonly loaded: boolean;
  readonly ttsEnabled: boolean;
  readonly ttsSupported: boolean;
  readonly onLoad: (args: { characterUrl: string; modelUrl: string }) => void;
  readonly onToggleTts: (enabled: boolean) => void;
  readonly onReset?: () => void;
}

export function ControlsPanel({
  characterUrl,
  modelUrl,
  status,
  busy,
  loaded,
  ttsEnabled,
  ttsSupported,
  onLoad,
  onToggleTts,
  onReset,
}: ControlsPanelProps) {
  const [cfg, setCfg] = useState(characterUrl);
  const [model, setModel] = useState(modelUrl);

  const handleSubmit = (event: FormEvent): void => {
    event.preventDefault();
    if (busy || cfg.trim().length === 0 || model.trim().length === 0) {
      return;
    }
    onLoad({ characterUrl: cfg.trim(), modelUrl: model.trim() });
  };

  return (
    <div className="controls-panel">
      <h2>Avatar Setup</h2>
      <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
        <label style={{ fontSize: 12, color: '#8a94a8' }}>
          character.json URL
          <input
            type="text"
            value={cfg}
            onChange={(e) => setCfg(e.target.value)}
            disabled={busy}
            placeholder="/character.json"
          />
        </label>
        <label style={{ fontSize: 12, color: '#8a94a8' }}>
          Model (.gguf) URL
          <input
            type="url"
            value={model}
            onChange={(e) => setModel(e.target.value)}
            disabled={busy}
            placeholder="https://huggingface.co/.../model.gguf"
          />
        </label>
        <div style={{ display: 'flex', gap: 8 }}>
          <button type="submit" disabled={busy}>
            {loaded ? 'Reload' : 'Load'}
          </button>
          {loaded && onReset ? (
            <button type="button" onClick={onReset} disabled={busy}>
              Reset memory
            </button>
          ) : null}
        </div>
        <label
          style={{
            fontSize: 12,
            color: ttsSupported ? '#8a94a8' : '#555',
            display: 'flex',
            alignItems: 'center',
            gap: 6,
          }}
        >
          <input
            type="checkbox"
            checked={ttsEnabled && ttsSupported}
            disabled={!ttsSupported}
            onChange={(e) => onToggleTts(e.target.checked)}
          />
          Speak responses {ttsSupported ? '' : '(unsupported in this browser)'}
        </label>
      </form>
      <div className="status-line">{status}</div>
    </div>
  );
}
