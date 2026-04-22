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
import { ChatPanel, type ChatMessage } from './components/ChatPanel';
import { ControlsPanel } from './components/ControlsPanel';

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
                        args: event.args,
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
                        ? '[No visible response generated.]'
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
    harness?.agent.clearMemory();
    setMessages([]);
    setStatus('Memory cleared.');
  };

  const vrmUrl = previewConfig?.assets?.vrm;

  return (
    <>
      <AvatarCanvas
        bus={bus}
        vrmUrl={vrmUrl}
        status={previewResolved ? undefined : 'Loading character preview…'}
      />
      <aside className="side-panel">
        <ControlsPanel
          characterUrl={characterUrl}
          modelUrl={modelUrl}
          status={status}
          busy={busy}
          loaded={harness != null}
          onLoad={handleLoad}
          onReset={handleReset}
        />
        <ChatPanel messages={messages} onSend={handleSend} disabled={!harness || busy} />
      </aside>
    </>
  );
}
