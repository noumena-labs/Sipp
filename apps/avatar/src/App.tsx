//////////////////////////////////////////////////////////////////////////////
//
// App.tsx
//
// - Wires a CogentEngine + CharacterAgent together with the three-vrm scene
//   and the chat UI. Config lives in /character.json; the user picks a
//   model URL at runtime. Everything below the app is renderer-agnostic.
//
//////////////////////////////////////////////////////////////////////////////

import { useEffect, useMemo, useRef, useState } from 'react';
import { CogentEngine, getBundledRuntimeUrls } from 'cogent-engine';
import {
  ActionBus,
  CharacterAgent,
  parseCharacterConfig,
  type CharacterConfig,
} from 'cogent-engine/character';
import { AvatarCanvas } from './components/AvatarCanvas';
import { ChatComposer } from './components/ChatComposer';
import { ControlsPanel } from './components/ControlsPanel';
import { TranscriptDrawer } from './components/TranscriptDrawer';
import type { ChatMessage } from './components/chat-types';

const DEFAULT_CHARACTER_URL = '/character.json';

interface LoadedHarness {
  readonly engine: CogentEngine;
  readonly agent: CharacterAgent;
  readonly config: CharacterConfig;
}

export default function App() {
  const [characterUrl, setCharacterUrl] = useState(DEFAULT_CHARACTER_URL);
  const [modelUrl, setModelUrl] = useState('');
  const [status, setStatus] = useState('Idle.');
  const [busy, setBusy] = useState(false);
  const [harness, setHarness] = useState<LoadedHarness | null>(null);
  const [previewConfig, setPreviewConfig] = useState<CharacterConfig | null>(null);
  const [previewResolved, setPreviewResolved] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [drawerOpen, setDrawerOpen] = useState(false);
  // The bus is created once per app lifetime and reused across harness
  // reloads so the scene binding is stable.
  const bus = useMemo(() => new ActionBus(), []);
  const abortRef = useRef<AbortController | null>(null);
  const previewRequestIdRef = useRef(0);

  // Bridge the agent's bus into the current harness when it changes so
  // scene bindings stay attached to whichever agent is live.
  useEffect(() => {
    if (!harness) {
      return;
    }
    const dispose = harness.agent.bus.onAny((event) => bus.emit(event));
    return dispose;
  }, [harness, bus]);

  const fetchCharacterConfig = async (configUrl: string): Promise<CharacterConfig> => {
    const res = await fetch(configUrl);
    if (!res.ok) {
      throw new Error(`character.json HTTP ${res.status}`);
    }
    const raw = await res.json();
    return parseCharacterConfig(raw);
  };

  const loadPreviewConfig = async (
    configUrl: string,
    requestId: number
  ): Promise<CharacterConfig> => {
    try {
      const config = await fetchCharacterConfig(configUrl);
      if (requestId === previewRequestIdRef.current) {
        setPreviewConfig(config);
        setPreviewResolved(true);
      }
      return config;
    } catch (error) {
      if (requestId === previewRequestIdRef.current) {
        setPreviewResolved(true);
      }
      throw error;
    }
  };

  useEffect(() => {
    let cancelled = false;
    const requestId = ++previewRequestIdRef.current;

    void (async () => {
      try {
        await loadPreviewConfig(DEFAULT_CHARACTER_URL, requestId);
      } catch (error) {
        if (cancelled || requestId !== previewRequestIdRef.current) {
          return;
        }
        setStatus(`Character preview failed: ${(error as Error).message}`);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  const handleLoad = async (args: { characterUrl: string; modelUrl: string }): Promise<void> => {
    abortRef.current?.abort();
    abortRef.current = null;
    setCharacterUrl(args.characterUrl);
    setModelUrl(args.modelUrl);
    setBusy(true);
    setStatus('Fetching character.json…');
    const previousHarness = harness;
    const requestId = ++previewRequestIdRef.current;
    let previewUpdated = false;
    try {
      const config = await loadPreviewConfig(args.characterUrl, requestId);
      previewUpdated = true;

      setStatus('Initialising engine…');
      const engine = new CogentEngine({ ...getBundledRuntimeUrls() });
      await engine.initModule();

      setStatus('Downloading model…');
      const modelPath = await engine.loadModelFromUrl(
        args.modelUrl,
        'model.gguf',
        (pct) => setStatus(`Downloading model… ${Math.floor(pct)}%`)
      );

      setStatus('Initialising inference runtime…');
      await engine.initEngine(modelPath, {
        sampling: {
          temperature: 0.6,
          topP: 0.9,
          topK: 40,
          minP: 0.05,
          repeatPenalty: 1.05,
        },
      });

      const agent = new CharacterAgent(engine, config, { bus: new ActionBus() });
      if (previousHarness) {
        previousHarness.engine.close();
      }
      setHarness({ engine, agent, config });
      setMessages([]);
      setDrawerOpen(false);
      setStatus(`Ready. Character: ${config.persona.name}.`);
    } catch (error) {
      console.error(error);
      if (previewUpdated && previousHarness && requestId === previewRequestIdRef.current) {
        setPreviewConfig(previousHarness.config);
      }
      setStatus(`Load failed: ${(error as Error).message}`);
    } finally {
      setBusy(false);
    }
  };

  const handleSend = async (text: string): Promise<void> => {
    if (!harness) {
      return;
    }
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    const userMessage: ChatMessage = {
      id: `u-${Date.now()}`,
      role: 'user',
      text,
      actions: [],
    };
    const assistantId = `a-${Date.now()}`;
    const assistantMessage: ChatMessage = {
      id: assistantId,
      role: 'assistant',
      text: '',
      actions: [],
      pending: true,
    };
    setMessages((prev) => [...prev, userMessage, assistantMessage]);

    try {
      for await (const event of harness.agent.chat(text, { signal: controller.signal })) {
        if (event.kind === 'prose') {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantId ? { ...msg, text: msg.text + event.text } : msg
            )
          );
        } else if (event.kind === 'action') {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantId
                ? {
                    ...msg,
                    actions: [
                      ...msg.actions,
                      {
                        name: event.name,
                        label: event.raw.slice(1, -1),
                      },
                    ],
                  }
                : msg
            )
          );
        } else if (event.kind === 'turn-end') {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantId
                ? {
                    ...msg,
                    text:
                      event.finalText.trim().length === 0 && msg.actions.length === 0
                        ? event.cancelled
                          ? '[Response interrupted.]'
                          : '[No visible response generated.]'
                        : event.finalText,
                    pending: false,
                  }
                : msg
            )
          );
          if (event.errorMessage) {
            setStatus(`Turn error: ${event.errorMessage}`);
          }
        }
      }
    } catch (error) {
      console.error(error);
      setStatus(`Turn failed: ${(error as Error).message}`);
      setMessages((prev) =>
        prev.map((msg) => (msg.id === assistantId ? { ...msg, pending: false } : msg))
      );
    }
  };

  const handleReset = (): void => {
    abortRef.current?.abort();
    abortRef.current = null;
    harness?.agent.clearMemory();
    setMessages([]);
    setStatus('Memory cleared.');
    setDrawerOpen(false);
  };

  const vrmUrl = previewConfig?.assets?.vrm;
  const latestAssistantMessage = [...messages]
    .reverse()
    .find((message) => message.role === 'assistant');
  const speaking =
    latestAssistantMessage?.pending === true && latestAssistantMessage.text.trim().length > 0;
  const characterName =
    previewConfig?.persona.name ?? harness?.config.persona.name ?? 'Companion';
  const personaSummary =
    previewConfig?.persona.summary ??
    harness?.config.persona.summary ??
    'A warm, playful stage companion.';
  const setupStatus = previewResolved ? status : 'Loading character preview…';

  return (
    <div className="avatar-app">
      <div className="stage-shell">
        <AvatarCanvas
          bus={bus}
          vrmUrl={vrmUrl}
          speaking={speaking}
          bubbleText={latestAssistantMessage?.text ?? ''}
          bubblePending={latestAssistantMessage?.pending ?? false}
        />

        <div className="stage-overlay stage-top-left">
          <ControlsPanel
            characterUrl={characterUrl}
            modelUrl={modelUrl}
            characterName={characterName}
            personaSummary={personaSummary}
            status={setupStatus}
            busy={busy}
            loaded={harness != null}
            onLoad={handleLoad}
            onReset={handleReset}
          />
        </div>

        <div className="stage-overlay stage-top-right">
          <button
            type="button"
            className={`history-toggle glass-panel${drawerOpen ? ' active' : ''}`}
            onClick={() => setDrawerOpen((open) => !open)}
            aria-expanded={drawerOpen}
            aria-controls="transcript-drawer"
          >
            <span className="panel-eyebrow">Transcript</span>
            <span className="history-toggle-label">Full chat log</span>
            <span className="history-toggle-count">{String(messages.length).padStart(2, '0')}</span>
          </button>
        </div>

        <div className="stage-overlay stage-bottom">
          <ChatComposer
            onSend={handleSend}
            disabled={!harness || busy}
            characterName={characterName}
          />
        </div>

        <TranscriptDrawer
          open={drawerOpen}
          messages={messages}
          onClose={() => setDrawerOpen(false)}
          characterName={characterName}
        />
      </div>
    </div>
  );
}
