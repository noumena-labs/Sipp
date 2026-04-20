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
  createLipsyncDriver,
  createWebSpeechTextToSpeech,
  parseCharacterConfig,
  type CharacterConfig,
} from 'cogent-engine/character';
import { AvatarCanvas } from './components/AvatarCanvas';
import { ChatPanel, type ChatMessage } from './components/ChatPanel';
import { ControlsPanel } from './components/ControlsPanel';

interface LoadedHarness {
  readonly engine: CogentEngine;
  readonly agent: CharacterAgent;
  readonly config: CharacterConfig;
}

export default function App() {
  const [characterUrl, setCharacterUrl] = useState('/character.json');
  const [modelUrl, setModelUrl] = useState('');
  const [status, setStatus] = useState('Idle.');
  const [busy, setBusy] = useState(false);
  const [harness, setHarness] = useState<LoadedHarness | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  // The bus is created once per app lifetime and reused across harness
  // reloads so the scene binding is stable.
  const bus = useMemo(() => new ActionBus(), []);
  // Lipsync + TTS are also stable across reloads; the driver emits openness
  // samples that the binding consumes regardless of which agent is live.
  const lipsync = useMemo(() => createLipsyncDriver(), []);
  const tts = useMemo(() => createWebSpeechTextToSpeech(), []);
  const [ttsEnabled, setTtsEnabled] = useState(tts.isSupported);
  const abortRef = useRef<AbortController | null>(null);

  // Dispose the lipsync driver on unmount. Speech + bus have no explicit
  // dispose hooks; stopping TTS on unmount is best-effort.
  useEffect(() => {
    return () => {
      lipsync.dispose();
      tts.stop();
    };
  }, [lipsync, tts]);

  // Bridge the agent's bus into the current harness when it changes so
  // scene bindings stay attached to whichever agent is live.
  useEffect(() => {
    if (!harness) {
      return;
    }
    const dispose = harness.agent.bus.onAny((event) => bus.emit(event));
    return dispose;
  }, [harness, bus]);

  const handleLoad = async (args: { characterUrl: string; modelUrl: string }): Promise<void> => {
    setCharacterUrl(args.characterUrl);
    setModelUrl(args.modelUrl);
    setBusy(true);
    setStatus('Fetching character.json…');
    try {
      const res = await fetch(args.characterUrl);
      if (!res.ok) {
        throw new Error(`character.json HTTP ${res.status}`);
      }
      const raw = await res.json();
      const config = parseCharacterConfig(raw);

      setStatus('Initialising engine…');
      const engine = new CogentEngine({...getBundledRuntimeUrls()});
      await engine.initModule();

      setStatus('Downloading model…');
      const modelPath = await engine.loadModelFromUrl(
        args.modelUrl,
        'model.gguf',
        (pct) => setStatus(`Downloading model… ${Math.floor(pct)}%`)
      );

      setStatus('Initialising inference runtime…');
      await engine.initEngine(modelPath);

      const agent = new CharacterAgent(engine, config, { bus: new ActionBus() });
      if (harness) {
        harness.engine.close();
      }
      setHarness({ engine, agent, config });
      setMessages([]);
      setStatus(`Ready. Character: ${config.persona.name}.`);
    } catch (error) {
      console.error(error);
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
    // Cancel any in-flight TTS from the previous turn so the mouth doesn't
    // linger open while the next response streams in.
    tts.stop();
    lipsync.stop();
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

    let proseBuffer = '';

    try {
      for await (const event of harness.agent.chat(text, { signal: controller.signal })) {
        if (event.kind === 'prose') {
          proseBuffer += event.text;
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
                    actions: [...msg.actions, { name: event.name, args: event.args }],
                  }
                : msg
            )
          );
        } else if (event.kind === 'turn-end') {
          setMessages((prev) =>
            prev.map((msg) => (msg.id === assistantId ? { ...msg, pending: false } : msg))
          );
          if (event.errorMessage) {
            setStatus(`Turn error: ${event.errorMessage}`);
          }
          // Speak the accumulated prose once the turn completes. We don't
          // stream TTS mid-generation because Web Speech has no incremental
          // API and chopping utterances produces awkward prosody.
          const speakable = proseBuffer.trim();
          if (ttsEnabled && tts.isSupported && speakable.length > 0) {
            lipsync.start();
            try {
              await tts.speak(speakable);
            } finally {
              lipsync.stop();
            }
          }
        }
      }
    } catch (error) {
      console.error(error);
      setStatus(`Turn failed: ${(error as Error).message}`);
      setMessages((prev) =>
        prev.map((msg) => (msg.id === assistantId ? { ...msg, pending: false } : msg))
      );
      lipsync.stop();
    }
  };

  const handleReset = (): void => {
    tts.stop();
    lipsync.stop();
    harness?.agent.clearMemory();
    setMessages([]);
    setStatus('Memory cleared.');
  };

  const vrmUrl = harness?.config.assets?.vrm;

  return (
    <>
      <AvatarCanvas
        bus={bus}
        vrmUrl={vrmUrl}
        lipsync={lipsync}
        status={harness ? undefined : 'No character loaded.'}
      />
      <aside className="side-panel">
        <ControlsPanel
          characterUrl={characterUrl}
          modelUrl={modelUrl}
          status={status}
          busy={busy}
          loaded={harness != null}
          ttsEnabled={ttsEnabled}
          ttsSupported={tts.isSupported}
          onLoad={handleLoad}
          onToggleTts={setTtsEnabled}
          onReset={handleReset}
        />
        <ChatPanel messages={messages} onSend={handleSend} disabled={!harness || busy} />
      </aside>
    </>
  );
}
