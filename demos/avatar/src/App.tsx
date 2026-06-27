//////////////////////////////////////////////////////////////////////////////
//
// App.tsx
//
// - Wires a SippClient + CharacterRuntime together with the avatar stage
//   and chat UI. `character.json` remains semantic-only; the avatar app owns
//   render assets and resolves them by character-folder convention.
//
//////////////////////////////////////////////////////////////////////////////

import { useEffect, useMemo, useRef, useState } from 'react';
import { SippClient, type ModelLoadProgress, type NativeRuntimeConfig } from '@noumena-labs/sipp';
import {
  CharacterEventBus,
  createCharacterFromConfigUrl,
  parseCharacterConfig,
  type CharacterConfig,
  type CharacterRuntime,
} from '@noumena-labs/sipp/character';
import { AvatarCanvas } from './components/AvatarCanvas';
import { ChatComposer } from './components/ChatComposer';
import {
  resolveAvatarRenderAssets,
  validateAvatarRenderAssets,
  type AvatarRenderAssets,
} from './characters/render-assets';
import { ControlsPanel } from './components/ControlsPanel';
import { TranscriptDrawer } from './components/TranscriptDrawer';
import { ActionsPanel } from './components/ActionsPanel';
import { StartScreen } from './components/StartScreen';
import type { ChatMessage } from './components/chat-types';

const DEFAULT_CHARACTER_URL = '/characters/aria/character.json';
const DEFAULT_MODEL_URL =
  'https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/main/LFM2.5-1.2B-Instruct-Q4_K_M.gguf';
const AVATAR_RUNTIME: NativeRuntimeConfig = {
  placement: {
    gpu_layers: 'all',
  },
  context: {
    n_ctx: 4096,
    n_parallel: 1,
    n_threads: 2,
    n_threads_batch: 4,
  },
  cache: {
    mode: 'live_slot_and_snapshot',
    retained_prefix_tokens: 256,
    snapshot_interval_tokens: 32,
  },
  sampling: {
    temperature: 0.6,
    top_p: 0.9,
    top_k: 40,
    min_p: 0.05,
    repeat_penalty: 1.05,
  },
};

const SUGGESTED_PROMPTS = [
  'Who are you, Aria?',
  'What danger is near us?',
  'Show me your favorite battle move.',
  'Can you summon your familiar?',
  'Teach me a spell before the quest.',
  'What does this ruin feel like?',
  "I'm nervous before the fight.",
  'Give me a heroic pep talk.',
] as const;

interface LoadedHarness {
  readonly client: SippClient;
  readonly character: CharacterRuntime;
  readonly config: CharacterConfig;
}

interface PreviewCharacter {
  readonly config: CharacterConfig;
  readonly renderAssets: AvatarRenderAssets;
}

export default function App() {
  const [modelUrl, setModelUrl] = useState(DEFAULT_MODEL_URL);
  const [status, setStatus] = useState('Idle.');
  const [busy, setBusy] = useState(false);
  const [started, setStarted] = useState(false);
  const [harness, setHarness] = useState<LoadedHarness | null>(null);
  const [previewCharacter, setPreviewCharacter] = useState<PreviewCharacter | null>(null);
  const [previewResolved, setPreviewResolved] = useState(false);
  const [avatarReady, setAvatarReady] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [drawerOpen, setDrawerOpen] = useState(true);
  // The bus is created once per app lifetime and reused across harness
  // reloads so the scene binding stays stable across character replacement.
  const bus = useMemo(() => new CharacterEventBus(), []);
  const abortRef = useRef<AbortController | null>(null);
  const previewRequestIdRef = useRef(0);

  const loadCharacterPreview = async (configUrl: string): Promise<PreviewCharacter> => {
    const res = await fetch(configUrl);
    if (!res.ok) {
      throw new Error(`character.json HTTP ${res.status}`);
    }
    const config = parseCharacterConfig(await res.json());
    const renderAssets = resolveAvatarRenderAssets(configUrl);
    await validateAvatarRenderAssets(config, renderAssets);
    return { config, renderAssets };
  };

  const loadPreviewConfig = async (
    configUrl: string,
    requestId: number
  ): Promise<PreviewCharacter> => {
    try {
      const preview = await loadCharacterPreview(configUrl);
      if (requestId === previewRequestIdRef.current) {
        setPreviewCharacter(preview);
        setPreviewResolved(true);
      }
      return preview;
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

  useEffect(() => {
    setAvatarReady(false);
  }, [previewCharacter]);

  const handleLoad = async (args: { modelUrl: string }): Promise<void> => {
    abortRef.current?.abort();
    abortRef.current = null;
    setModelUrl(args.modelUrl);
    setBusy(true);
    setStatus('Loading Aria?');
    const previousHarness = harness;
    const previousPreviewCharacter = previewCharacter;
    const requestId = ++previewRequestIdRef.current;
    let previewUpdated = false;
    try {
      const preview = await loadPreviewConfig(DEFAULT_CHARACTER_URL, requestId);
      const config = preview.config;
      previewUpdated = true;

      const client = new SippClient();

      setStatus('Downloading and loading model?');
      await client.add('local', {
        kind: 'local',
        source: args.modelUrl,
        options: {
          onProgress: (progress: ModelLoadProgress) => {
            if (progress.phase === 'download') {
              setStatus(`Downloading model... ${Math.floor(progress.percent ?? 0)}%`);
            } else if (progress.phase === 'load') {
              setStatus('Loading into memory...');
            }
          },
          runtime: AVATAR_RUNTIME,
        },
      });

      const { character } = await createCharacterFromConfigUrl({
        configUrl: DEFAULT_CHARACTER_URL,
        client,
        bus,
      });
      if (previousHarness) {
        previousHarness.client.close();
      }
      setHarness({ client, character, config });
      setMessages([]);
      setDrawerOpen(true);
      setStarted(true);
      setStatus(`Ready. Character: ${config.persona.name}.`);
    } catch (error) {
      console.error(error);
      if (previewUpdated && previousHarness && requestId === previewRequestIdRef.current) {
        setPreviewCharacter(previousPreviewCharacter);
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
      for await (const event of harness.character.chat(text, { signal: controller.signal })) {
        if (event.kind === 'prose') {
          // PROSE OPTIMIZATION: We no longer update the global 'messages' state
          // for every delivered token. Instead, components like TranscriptDrawer
          // and SpeechBubble subscribe to the CharacterEventBus ('bus') to
          // provide real-time visual feedback without triggering full React
          // reconciliation cycles.
        } else if (event.kind === 'action') {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantId
                ? {
                  ...msg,
                  actions: [
                    ...msg.actions,
                    {
                      id: event.id,
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
                      ? event.status === 'aborted'
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
    harness?.character.clearMemory();
    setMessages([]);
    setStatus('Memory cleared.');
    setDrawerOpen(true);
  };

  const handleAvatarError = (message: string): void => {
    setAvatarReady(false);
    setStatus(`Avatar failed: ${message}`);
  };

  const handleManualAction = (actionId: string, cueLabel: string): void => {
    bus.emit({ kind: 'action', id: actionId, raw: `[${cueLabel}]` });
  };

  const latestAssistantMessage = [...messages]
    .reverse()
    .find((message) => message.role === 'assistant');
  const speaking =
    latestAssistantMessage?.pending === true;
  const characterName =
    previewCharacter?.config.persona.name ?? harness?.config.persona.name ?? 'Companion';
  const personaSummary =
    previewCharacter?.config.persona.summary ??
    harness?.config.persona.summary ??
    'A warm, playful stage companion.';
  const actionNames = useMemo(
    () => previewCharacter?.config.actions.map((action: { id: any; }) => action.id) ?? [],
    [previewCharacter]
  );
  const setupStatus = previewResolved ? status : 'Loading character preview?';

  return (
    <div className="avatar-app">
      <div className="stage-shell">
        <AvatarCanvas
          bus={bus}
          renderAssets={previewCharacter?.renderAssets}
          actionNames={actionNames}
          speaking={speaking}
          bubbleText={latestAssistantMessage?.text ?? ''}
          bubblePending={latestAssistantMessage?.pending ?? false}
          bubbleActions={latestAssistantMessage?.actions ?? []}
          characterName={characterName}
          onReady={() => setAvatarReady(true)}
          onError={handleAvatarError}
        />

        {started ? (
          <>
            <div className="stage-overlay stage-top-left">
              <ControlsPanel
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
                <span className="history-toggle-label">
                  {drawerOpen ? 'Hide chat log' : 'Show chat log'}
                </span>
                <span className="history-toggle-count">{String(messages.length).padStart(2, '0')}</span>
              </button>
            </div>

            <div className="stage-overlay stage-actions">
              <ActionsPanel
                actions={previewCharacter?.config.actions ?? []}
                disabled={!avatarReady || busy}
                onTrigger={handleManualAction}
              />
            </div>

            <div className="stage-overlay stage-bottom">
              <ChatComposer
                onSend={handleSend}
                disabled={!harness || busy}
                characterName={characterName}
                suggestions={SUGGESTED_PROMPTS}
              />
            </div>

            <TranscriptDrawer
              open={drawerOpen}
              messages={messages}
              bus={bus}
              onClose={() => setDrawerOpen(false)}
              characterName={characterName}
            />

          </>
        ) : (
          <StartScreen
            modelUrl={modelUrl}
            characterName={characterName}
            personaSummary={personaSummary}
            status={setupStatus}
            busy={busy}
            onStart={handleLoad}
          />
        )}
      </div>
    </div>
  );
}
