//////////////////////////////////////////////////////////////////////////////
//
// StartScreen.tsx
//
// - Presents the avatar demo as a start menu and owns the initial model URL
//   entry before the interactive experience is unlocked.
//
//////////////////////////////////////////////////////////////////////////////

import type { FormEvent } from 'react';
import { useEffect, useState } from 'react';

interface StartScreenProps {
  readonly modelUrl: string;
  readonly characterName?: string;
  readonly personaSummary?: string;
  readonly status: string;
  readonly busy: boolean;
  readonly onStart: (args: { modelUrl: string }) => void | Promise<void>;
}

export function StartScreen({
  modelUrl,
  characterName = 'Aria',
  personaSummary = 'A warm, playful stage companion.',
  status,
  busy,
  onStart,
}: StartScreenProps) {
  const [model, setModel] = useState(modelUrl);

  useEffect(() => {
    setModel(modelUrl);
  }, [modelUrl]);

  const trimmedModel = model.trim();

  const handleSubmit = (event: FormEvent): void => {
    event.preventDefault();
    if (busy || trimmedModel.length === 0) {
      return;
    }
    void onStart({ modelUrl: trimmedModel });
  };

  return (
    <div className="start-screen">
      <section className="start-card glass-panel" aria-labelledby="start-title">
        <div className="start-hero">
          <span className="panel-eyebrow">Cogent Avatar Demo</span>
          <h1 id="start-title">Enter the Starfall Gate</h1>
          <p className="start-lede">
            Meet {characterName}, a real-time interactive character powered by CogentLM, a high-performance inference engine for local LLMs.
            This tech demo shows how a local model can drive lifelike character interactions, call actions in response to user input, and operate seamlessly in a dynamic real-time environment.
          </p>
        </div>

        <div className="start-character-card">
          <span className="start-character-label">Featured companion</span>
          <strong>{characterName}</strong>
          <p>{personaSummary}</p>
        </div>
       
        {/* 
        <div className="start-feature-list" aria-label="Experience features">
          <span>Local GGUF model</span>
          <span>High-performance runtime</span>
          <span>Dynamic avatar actions</span>
        </div> */}

        <form className="start-model-form" onSubmit={handleSubmit}>
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



          <button
            className="start-button"
            type="submit"
            disabled={busy || trimmedModel.length === 0}
          >
            {busy ? 'Starting' : 'Start'}
          </button>

          <div className="start-status" aria-live="polite">
           {status}
        </div>
        </form>
      </section>
    </div>
  );
}
