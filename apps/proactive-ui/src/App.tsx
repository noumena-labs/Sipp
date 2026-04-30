import { useEffect, useRef, useState } from 'react';
import { flushSync } from 'react-dom';
import { CogentEngine } from '@noumena-labs/cogent-engine';
import { toCanvas } from 'html-to-image';
import {
  applyDomPatches,
  collectPatchTargets,
  DEFAULT_PATCH_DIRECTOR_CONFIG,
  DomPatchDirector,
  loadDomPatchDirectorConfig,
  type AppliedMutation,
  type DomPatch,
  type DomPatchDirectorConfig,
  type RejectedPatch,
} from './dom-patch';
import {
  calculateFieldKitScore,
  categoryLabel,
  CATEGORY_GOALS,
  FIELD_KIT_LIMITS,
  GEAR_ITEMS,
  type GearTab,
} from './game-data';

const DEFAULT_MODEL_URL =
  'https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF/resolve/main/LFM2.5-VL-450M-F16.gguf';
const DEFAULT_PROJECTOR_URL =
  'https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF/resolve/main/mmproj-LFM2.5-VL-450m-F16.gguf';
const DIRECTOR_CONFIG_URL = '/directors/field-kit/dom-patch-director.json';
const INITIAL_COACH_HTML = `
  <h3 class="ai-gen-title">Proactive coach</h3>
  <p class="ai-gen-note">Pick a phase, pack a few items, then run a peek. The model can replace this panel with generated guidance.</p>
`;

type LoadState = 'idle' | 'loading' | 'ready' | 'error';
type VisionState = 'idle' | 'capturing' | 'compressing' | 'thinking' | 'patching' | 'error';
type PeekSource = 'manual' | 'auto';
type CaptureMode = 'fast' | 'detailed';

interface LoadProgressState {
  readonly phase: string;
  readonly assetName?: string;
  readonly percent: number | null;
}

interface TraceState {
  readonly id: number;
  readonly source: PeekSource;
  readonly visionState: VisionState;
  readonly status: string;
  readonly startedAt: string;
  readonly durationMs?: number;
  readonly screenshotUrl?: string;
  readonly screenshotBytes?: number;
  readonly screenshotWidth?: number;
  readonly screenshotHeight?: number;
  readonly captureMode?: CaptureMode;
  readonly targetCount?: number;
  readonly promptPreview?: string;
  readonly rawText?: string;
  readonly observation?: string;
  readonly intent?: string;
  readonly patches?: readonly DomPatch[];
  readonly rejectedPatches?: readonly RejectedPatch[];
  readonly appliedMutations?: readonly AppliedMutation[];
  readonly errorMessage?: string;
}

interface CapturedImage {
  readonly bytes: Uint8Array;
  readonly url: string;
  readonly width: number;
  readonly height: number;
  readonly byteLength: number;
}

const PHASE_COPY: Record<GearTab, { title: string; objective: string; cta: string }> = {
  brief: {
    title: 'Read the mission brief',
    objective: 'Understand the objective, win condition, and constraints before packing.',
    cta: 'Start Packing',
  },
  gear: {
    title: 'Pack expedition gear',
    objective: 'Choose items that cover every required category while staying under weight and budget.',
    cta: 'Review Launch Gate',
  },
  launch: {
    title: 'Pass the launch gate',
    objective: 'Resolve any missing categories or over-limit constraints, then launch the expedition.',
    cta: 'Launch Expedition',
  },
};

const CAPTURE_PRESETS: Record<CaptureMode, { maxWidth: number; quality: number }> = {
  fast: { maxWidth: 768, quality: 0.72 },
  detailed: { maxWidth: 1400, quality: 0.88 },
};

export default function App() {
  const [modelUrl, setModelUrl] = useState(DEFAULT_MODEL_URL);
  const [projectorUrl, setProjectorUrl] = useState(DEFAULT_PROJECTOR_URL);
  const [loadState, setLoadState] = useState<LoadState>('idle');
  const [loadProgress, setLoadProgress] = useState<LoadProgressState | null>(null);
  const [status, setStatus] = useState('Load a vision model to begin.');
  const [visionState, setVisionState] = useState<VisionState>('idle');
  const [activeTab, setActiveTab] = useState<GearTab>('brief');
  const [selectedIds, setSelectedIds] = useState<ReadonlySet<string>>(() => new Set(['paper-map']));
  const [autoPeek, setAutoPeek] = useState(false);
  const [captureMode, setCaptureMode] = useState<CaptureMode>('fast');
  const [trace, setTrace] = useState<TraceState | null>(null);
  const [directorConfig, setDirectorConfig] = useState<DomPatchDirectorConfig>(DEFAULT_PATCH_DIRECTOR_CONFIG);
  const workspaceRef = useRef<HTMLDivElement | null>(null);
  const engineRef = useRef<CogentEngine | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const screenshotUrlRef = useRef<string | null>(null);
  const peekRef = useRef<((source: PeekSource) => Promise<void>) | null>(null);
  const busyRef = useRef(false);
  const traceIdRef = useRef(0);

  const score = calculateFieldKitScore(selectedIds);
  const selectedKey = Array.from(selectedIds).sort().join('|');
  const visibleGear = activeTab === 'gear'
    ? GEAR_ITEMS.filter((item) => item.tab === 'gear')
    : activeTab === 'launch'
      ? GEAR_ITEMS.filter((item) => item.tab === 'launch')
      : [];
  const engineReady = loadState === 'ready' && engineRef.current != null;
  const busy = loadState === 'loading' || visionState === 'capturing' || visionState === 'compressing' || visionState === 'thinking' || visionState === 'patching';
  busyRef.current = busy;

  useEffect(() => {
    let cancelled = false;
    void loadDomPatchDirectorConfig(DIRECTOR_CONFIG_URL)
      .then((config) => {
        if (!cancelled) {
          setDirectorConfig(config);
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setStatus(`Using built-in director config: ${(error as Error).message}`);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      engineRef.current?.close();
      if (screenshotUrlRef.current) {
        URL.revokeObjectURL(screenshotUrlRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (!autoPeek || !engineReady || busyRef.current) {
      return;
    }
    const timeout = window.setTimeout(() => {
      void peekRef.current?.('auto');
    }, 1800);
    return () => window.clearTimeout(timeout);
  }, [activeTab, autoPeek, engineReady, selectedKey]);

  const handleLoad = async (): Promise<void> => {
    const trimmedModel = modelUrl.trim();
    const trimmedProjector = projectorUrl.trim();
    if (!trimmedModel || !trimmedProjector) {
      setStatus('Provide both model and projector URLs for the vision pipeline.');
      setLoadState('error');
      return;
    }

    abortRef.current?.abort();
    abortRef.current = null;
    setLoadState('loading');
    setLoadProgress({ phase: 'create', percent: null });
    setStatus('Creating Cogent Engine instance...');

    try {
      const nextEngine = await CogentEngine.create();
      setStatus('Downloading vision model and projector...');
      await nextEngine.models.load(
        { model: trimmedModel, projector: trimmedProjector },
        {
          onProgress: (progress) => {
            setLoadProgress({
              phase: progress.phase,
              ...(progress.assetName ? { assetName: progress.assetName } : {}),
              percent: progress.percent,
            });
            if (progress.phase === 'download') {
              const asset = progress.assetName ? ` ${progress.assetName}` : '';
              setStatus(`Downloading${asset}... ${Math.floor(progress.percent ?? 0)}%`);
            } else if (progress.phase === 'load') {
              setStatus('Loading model into memory...');
            } else if (progress.phase === 'metadata') {
              setStatus('Resolving model metadata...');
            } else if (progress.phase === 'store') {
              setStatus('Storing model assets...');
            }
          },
          runtime: {
            imageMinTokens: 64,
            imageMaxTokens: 768,
            sampling: {
              temperature: 0.15,
              topP: 0.9,
              topK: 30,
              minP: 0.05,
              repeatPenalty: 1.04,
            },
          },
        }
      );
      engineRef.current?.close();
      engineRef.current = nextEngine;
      setLoadProgress({ phase: 'ready', percent: 100 });
      setLoadState('ready');
      setStatus('Vision model ready. Demo started automatically.');
    } catch (error) {
      setLoadState('error');
      setStatus(`Load failed: ${(error as Error).message}`);
    }
  };

  const handleChangeModel = (): void => {
    abortRef.current?.abort();
    engineRef.current?.close();
    engineRef.current = null;
    setAutoPeek(false);
    setLoadProgress(null);
    setLoadState('idle');
    setVisionState('idle');
    setStatus('Load a vision model to begin.');
  };

  const handleVisionPeek = async (source: PeekSource): Promise<void> => {
    const root = workspaceRef.current;
    const engine = engineRef.current;
    if (!root || !engine) {
      setStatus('Load the vision model before peeking.');
      return;
    }

    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    clearTransientAiClasses(root, directorConfig.patchPolicy.allowedClasses);

    const id = ++traceIdRef.current;
    const started = performance.now();
    const startedAt = new Date().toLocaleTimeString();
    setVisionState('capturing');
    setTrace({ id, source, startedAt, visionState: 'capturing', captureMode, status: 'Capturing main UI...' });

    try {
      const canvas = await toCanvas(root, {
        cacheBust: true,
        pixelRatio: 1,
        backgroundColor: '#150f0a',
        filter: (node) => !(node instanceof HTMLElement && node.dataset.captureExclude === 'true'),
      });

      setVisionState('compressing');
      setTrace((previous) => previous?.id === id ? {
        ...previous,
        visionState: 'compressing',
        status: `${captureMode === 'fast' ? 'Fast' : 'Detailed'} capture: downscaling and JPEG encoding...`,
      } : previous);

      const captured = await encodeCapture(canvas, captureMode);
      if (screenshotUrlRef.current) {
        URL.revokeObjectURL(screenshotUrlRef.current);
      }
      screenshotUrlRef.current = captured.url;
      const targets = collectPatchTargets(root);

      setVisionState('thinking');
      setTrace({
        id,
        source,
        startedAt,
        screenshotUrl: captured.url,
        screenshotBytes: captured.byteLength,
        screenshotWidth: captured.width,
        screenshotHeight: captured.height,
        captureMode,
        targetCount: targets.length,
        visionState: 'thinking',
        status: `Sending ${formatBytes(captured.byteLength)} image and ${targets.length} DOM targets to the model...`,
      });

      const director = new DomPatchDirector(engine, directorConfig);
      const result = await director.run({
        screenshot: captured.bytes,
        targets,
        gameState: { activeTab, score, selectedItems: score.selectedItems },
        signal: controller.signal,
      });
      const appliedMutations = summarizePatches(result.patches);
      const durationMs = Math.round(performance.now() - started);
      const finalTrace: TraceState = {
        id,
        source,
        startedAt,
        durationMs,
        screenshotUrl: captured.url,
        screenshotBytes: captured.byteLength,
        screenshotWidth: captured.width,
        screenshotHeight: captured.height,
        captureMode,
        targetCount: result.targetCount,
        promptPreview: result.promptPreview,
        rawText: result.rawText,
        observation: result.observation,
        intent: result.intent,
        patches: result.patches,
        rejectedPatches: result.rejectedPatches,
        appliedMutations,
        visionState: 'patching',
        status: `Parsed JSON and validated ${result.patches.length} patch(es); applying DOM mutations...`,
      };

      flushSync(() => {
        setVisionState('idle');
        setStatus(`Peek complete in ${(durationMs / 1000).toFixed(1)}s. DOM patched from model JSON.`);
        setTrace({ ...finalTrace, visionState: 'idle', status: 'DOM patches applied.' });
      });
      applyDomPatches(root, result.patches);
    } catch (error) {
      if (controller.signal.aborted) {
        return;
      }
      const durationMs = Math.round(performance.now() - started);
      setVisionState('error');
      setStatus(`Peek failed: ${(error as Error).message}`);
      setTrace((previous) => ({
        id,
        source,
        startedAt,
        durationMs,
        screenshotUrl: previous?.id === id ? previous.screenshotUrl : undefined,
        screenshotBytes: previous?.id === id ? previous.screenshotBytes : undefined,
        screenshotWidth: previous?.id === id ? previous.screenshotWidth : undefined,
        screenshotHeight: previous?.id === id ? previous.screenshotHeight : undefined,
        captureMode,
        targetCount: previous?.id === id ? previous.targetCount : undefined,
        visionState: 'error',
        status: 'Vision-to-DOM loop failed.',
        errorMessage: (error as Error).message,
      }));
    }
  };

  peekRef.current = handleVisionPeek;

  const toggleGear = (itemId: string): void => {
    setSelectedIds((previous) => {
      const next = new Set(previous);
      if (next.has(itemId)) {
        next.delete(itemId);
      } else {
        next.add(itemId);
      }
      return next;
    });
  };

  const resetRun = (): void => {
    abortRef.current?.abort();
    setSelectedIds(new Set(['paper-map']));
    setActiveTab('brief');
    setVisionState('idle');
    setStatus('Run reset. Pick gear and peek again.');
    if (workspaceRef.current) {
      clearTransientAiClasses(workspaceRef.current, directorConfig.patchPolicy.allowedClasses);
      deleteModifiedFlags(workspaceRef.current);
    }
  };

  if (!engineReady) {
    return (
      <StartScreen
        modelUrl={modelUrl}
        projectorUrl={projectorUrl}
        loadState={loadState}
        status={status}
        progress={loadProgress}
        onModelUrlChange={setModelUrl}
        onProjectorUrlChange={setProjectorUrl}
        onLoad={handleLoad}
      />
    );
  }

  return (
    <div className="proactive-app demo-mode">
      <main className="demo-grid">
        <section className="workspace-shell" aria-label="Dust Ridge Field Kit game">
          <div className={`capture-stage ${visionState}`} ref={workspaceRef} data-ai-zone="dust-ridge-field-kit" data-ai-goal="Help the user finish a safe desert expedition kit.">
            <div className="stage-topline">
              <div>
                <p className="eyebrow">Dust Ridge Field Kit</p>
                <h1>Launch before the sand wall hits</h1>
                <p className="stage-subtitle">Choose items to cover every survival category while staying under carry weight and budget.</p>
              </div>
              <div className="storm-clock" data-ai-id="storm-clock" data-ai-label="Storm arrival countdown" data-ai-ops="replaceText,addClass,removeClass,setAttribute">
                <span>{FIELD_KIT_LIMITS.stormMinutes}</span>
                <small>min to storm</small>
              </div>
            </div>

            {visionState !== 'idle' && visionState !== 'error' ? (
              <div className="vision-overlay">
                <div className="scanner" />
                <strong>{describeVisionState(visionState, captureMode)}</strong>
              </div>
            ) : null}

            <nav className="tab-row" aria-label="Field kit steps">
              {(['brief', 'gear', 'launch'] as const).map((tab) => (
                <button
                  key={tab}
                  type="button"
                  className={activeTab === tab ? 'active' : ''}
                  onClick={() => setActiveTab(tab)}
                  data-ai-id={`tab-${tab}`}
                  data-ai-label={`${tab} navigation tab`}
                  data-ai-ops="addClass,removeClass,setAttribute,scrollIntoView"
                >
                  <span>{tab === 'brief' ? '1' : tab === 'gear' ? '2' : '3'}</span>
                  {tab === 'brief' ? 'Brief' : tab === 'gear' ? 'Pack Gear' : 'Launch'}
                </button>
              ))}
            </nav>

            <div className="objective-banner" data-ai-id="phase-objective" data-ai-label="Current phase objective" data-ai-ops="replaceHtml,appendHtml,addClass,removeClass,setAttribute">
              <p className="eyebrow">Current Objective</p>
              <h2>{PHASE_COPY[activeTab].title}</h2>
              <p>{PHASE_COPY[activeTab].objective}</p>
            </div>

            <div className="score-strip">
              <Meter id="readiness" label="Readiness" value={score.readiness} max={100} suffix="%" state={score.readyToLaunch ? 'good' : score.readiness > 65 ? 'warn' : 'low'} />
              <Meter id="weight" label="Weight" value={score.totalWeight} max={FIELD_KIT_LIMITS.maxWeight} suffix="kg" state={score.weightOk ? 'good' : 'bad'} />
              <Meter id="budget" label="Budget" value={score.totalCost} max={FIELD_KIT_LIMITS.maxBudget} suffix="$" state={score.budgetOk ? 'good' : 'bad'} />
            </div>

            <div className="mission-layout">
              <aside className="mission-panel glass-card" data-ai-id="mission-checklist" data-ai-label="Mission checklist" data-ai-ops="addClass,removeClass,setAttribute,scrollIntoView">
                <p className="eyebrow">Required Coverage</p>
                <div className="goal-list">
                  {CATEGORY_GOALS.map((goal) => {
                    const covered = score.coveredCategories.includes(goal.id);
                    return (
                      <div
                        key={goal.id}
                        className={`goal-row ${covered ? 'covered' : 'missing'}`}
                        data-ai-id={`goal-${goal.id}`}
                        data-ai-label={`${goal.label} requirement`}
                        data-ai-ops="addClass,removeClass,setAttribute,scrollIntoView"
                      >
                        <span>{covered ? '✓' : '!'}</span>
                        <div>
                          <strong>{goal.label}</strong>
                          <small>{goal.prompt}</small>
                        </div>
                      </div>
                    );
                  })}
                </div>

                <div className="selected-manifest" data-ai-id="selected-manifest" data-ai-label="Selected gear manifest" data-ai-ops="replaceHtml,appendHtml,addClass,removeClass,setAttribute,scrollIntoView">
                  <p className="eyebrow">Packed Items</p>
                  {score.selectedItems.length === 0 ? (
                    <small>No items packed.</small>
                  ) : (
                    <ul>
                      {score.selectedItems.map((item) => <li key={item.id}>{item.name}</li>)}
                    </ul>
                  )}
                </div>
              </aside>

              <section className="phase-zone">
                {activeTab === 'brief' ? (
                  <BriefPhase onStartPacking={() => setActiveTab('gear')} />
                ) : activeTab === 'gear' ? (
                  <GearPhase visibleGear={visibleGear} selectedIds={selectedIds} onToggleGear={toggleGear} onReview={() => setActiveTab('launch')} />
                ) : (
                  <LaunchPhase visibleGear={visibleGear} selectedIds={selectedIds} score={score} onToggleGear={toggleGear} />
                )}
              </section>

              <aside className="coach-column">
                <div
                  className="coach-panel glass-card"
                  data-ai-id="coach-panel"
                  data-ai-label="Generated proactive coach panel"
                  data-ai-ops="replaceHtml,appendHtml,addClass,removeClass,setAttribute,scrollIntoView"
                  dangerouslySetInnerHTML={{ __html: INITIAL_COACH_HTML }}
                />
              </aside>
            </div>
          </div>
        </section>

        <AIConsole
          status={status}
          loadState={loadState}
          visionState={visionState}
          engineReady={engineReady}
          busy={busy}
          autoPeek={autoPeek}
          captureMode={captureMode}
          trace={trace}
          onPeek={() => void handleVisionPeek('manual')}
          onAutoPeekChange={setAutoPeek}
          onCaptureModeChange={setCaptureMode}
          onReset={resetRun}
          onChangeModel={handleChangeModel}
        />
      </main>
    </div>
  );
}

function StartScreen(props: {
  readonly modelUrl: string;
  readonly projectorUrl: string;
  readonly loadState: LoadState;
  readonly status: string;
  readonly progress: LoadProgressState | null;
  readonly onModelUrlChange: (value: string) => void;
  readonly onProjectorUrlChange: (value: string) => void;
  readonly onLoad: () => void | Promise<void>;
}) {
  const percent = props.progress?.percent ?? null;
  return (
    <div className="start-screen">
      <section className="start-hero glass-card">
        <div className="start-copy">
          <p className="eyebrow">Cogent Engine Vision Pipeline</p>
          <h1>Proactive UI that sees, reasons, and modifies the DOM in real-time.</h1>
          <p>
            This demo loads a local vision model, captures the web app as an image, asks the model
            what the user appears to be doing, and applies validated JSON DOM patches to help them.
          </p>
          <div className="start-steps">
            <span>1. Load model</span>
            <span>2. Pack the field kit</span>
            <span>3. Peek at UI</span>
            <span>4. Watch DOM patches land</span>
          </div>
        </div>

        <div className="loader-panel">
          <p className="eyebrow">Start Demo</p>
          <label>
            Vision model URL
            <input value={props.modelUrl} onChange={(event) => props.onModelUrlChange(event.target.value)} disabled={props.loadState === 'loading'} />
          </label>
          <label>
            Projector URL
            <input value={props.projectorUrl} onChange={(event) => props.onProjectorUrlChange(event.target.value)} disabled={props.loadState === 'loading'} />
          </label>
          <button className="primary-button" type="button" onClick={() => void props.onLoad()} disabled={props.loadState === 'loading'}>
            {props.loadState === 'loading' ? 'Loading vision model...' : 'Load Model and Start'}
          </button>

          <div className="load-observability">
            <div className="status-line">
              <span className={`state-dot ${props.loadState}`} />
              <strong>{props.status}</strong>
            </div>
            <div className="load-progress-track">
              <span style={{ width: `${percent ?? (props.loadState === 'loading' ? 12 : 0)}%` }} />
            </div>
            <dl>
              <div>
                <dt>Phase</dt>
                <dd>{props.progress?.phase ?? props.loadState}</dd>
              </div>
              <div>
                <dt>Asset</dt>
                <dd>{props.progress?.assetName ?? 'waiting'}</dd>
              </div>
              <div>
                <dt>Progress</dt>
                <dd>{percent == null ? 'streaming' : `${Math.floor(percent)}%`}</dd>
              </div>
            </dl>
          </div>
        </div>
      </section>
    </div>
  );
}

function BriefPhase(props: { readonly onStartPacking: () => void }) {
  return (
    <div className="phase-card glass-card" data-ai-id="brief-card" data-ai-label="Mission brief" data-ai-ops="replaceHtml,appendHtml,addClass,removeClass,setAttribute">
      <p className="eyebrow">Mission Rules</p>
      <h3>Cross Dust Ridge and return before visibility collapses.</h3>
      <div className="rules-grid">
        <div>
          <strong>Win condition</strong>
          <p>Cover all six survival categories and keep the launch gate unlocked.</p>
        </div>
        <div>
          <strong>Carry limit</strong>
          <p>Stay at or below {FIELD_KIT_LIMITS.maxWeight}kg total pack weight.</p>
        </div>
        <div>
          <strong>Budget cap</strong>
          <p>Stay at or below ${FIELD_KIT_LIMITS.maxBudget} so the supply officer signs off.</p>
        </div>
      </div>
      <button className="phase-cta" type="button" onClick={props.onStartPacking}>Start Packing</button>
    </div>
  );
}

function GearPhase(props: {
  readonly visibleGear: readonly typeof GEAR_ITEMS[number][];
  readonly selectedIds: ReadonlySet<string>;
  readonly onToggleGear: (itemId: string) => void;
  readonly onReview: () => void;
}) {
  return (
    <>
      <div className="instruction-card glass-card" data-ai-id="pack-instructions" data-ai-label="Pack gear instructions" data-ai-ops="replaceHtml,appendHtml,addClass,removeClass,setAttribute">
        <h3>Choose the core kit</h3>
        <p>Select cards that satisfy hydration, navigation, sun cover, and power without overpacking. The model should notice missing categories from the visible checklist and cards.</p>
      </div>
      <GearGrid items={props.visibleGear} selectedIds={props.selectedIds} onToggleGear={props.onToggleGear} />
      <button className="phase-cta" type="button" onClick={props.onReview}>Review Launch Gate</button>
    </>
  );
}

function LaunchPhase(props: {
  readonly visibleGear: readonly typeof GEAR_ITEMS[number][];
  readonly selectedIds: ReadonlySet<string>;
  readonly score: ReturnType<typeof calculateFieldKitScore>;
  readonly onToggleGear: (itemId: string) => void;
}) {
  return (
    <>
      <div className="launch-panel glass-card" data-ai-id="launch-panel" data-ai-label="Launch readiness panel" data-ai-ops="replaceHtml,appendHtml,addClass,removeClass,setAttribute,scrollIntoView">
        <p className="eyebrow">Expedition Gate</p>
        <h3>{props.score.readyToLaunch ? 'Ready to launch' : 'Not safe yet'}</h3>
        <p>
          {props.score.readyToLaunch
            ? 'All required categories are covered within constraints.'
            : props.score.missingCategories.length > 0
              ? `Missing: ${props.score.missingCategories.map(categoryLabel).join(', ')}.`
              : 'Coverage is complete, but weight or budget needs attention.'}
        </p>
        <button
          className="launch-button"
          type="button"
          disabled={!props.score.readyToLaunch}
          data-ai-id="launch-button"
          data-ai-label="Launch expedition button"
          data-ai-ops="replaceText,addClass,removeClass,setAttribute,scrollIntoView"
        >
          {props.score.readyToLaunch ? 'Launch Expedition' : 'Locked'}
        </button>
      </div>
      <div className="instruction-card glass-card" data-ai-id="final-safety-instructions" data-ai-label="Final safety instructions" data-ai-ops="replaceHtml,appendHtml,addClass,removeClass,setAttribute">
        <h3>Final safety shelf</h3>
        <p>Add signal and first-aid items here. The launch button unlocks only when the full field kit is safe.</p>
      </div>
      <GearGrid items={props.visibleGear} selectedIds={props.selectedIds} onToggleGear={props.onToggleGear} />
    </>
  );
}

function GearGrid(props: {
  readonly items: readonly typeof GEAR_ITEMS[number][];
  readonly selectedIds: ReadonlySet<string>;
  readonly onToggleGear: (itemId: string) => void;
}) {
  return (
    <div className="gear-grid" aria-label="Gear cards">
      {props.items.map((item) => {
        const selected = props.selectedIds.has(item.id);
        return (
          <button
            key={item.id}
            type="button"
            className={`gear-card ${selected ? 'selected' : ''}`}
            onClick={() => props.onToggleGear(item.id)}
            data-ai-id={`gear-${item.id}`}
            data-ai-label={`${item.name} gear card`}
            data-ai-ops="addClass,removeClass,setAttribute,scrollIntoView"
          >
            <span className="gear-category">{categoryLabel(item.category)}</span>
            <strong>{item.name}</strong>
            <p>{item.summary}</p>
            <small>{item.tradeoff}</small>
            <span className="gear-stats">{item.weight}kg · ${item.cost}</span>
          </button>
        );
      })}
    </div>
  );
}

function Meter(props: {
  readonly id: 'readiness' | 'weight' | 'budget';
  readonly label: string;
  readonly value: number;
  readonly max: number;
  readonly suffix: string;
  readonly state: 'good' | 'warn' | 'low' | 'bad';
}) {
  const percent = Math.max(0, Math.min(100, Math.round((props.value / props.max) * 100)));
  const displayValue = props.id === 'budget' ? `$${props.value}` : `${props.value}${props.suffix}`;
  const displayMax = props.id === 'budget' ? `$${props.max}` : `${props.max}${props.suffix}`;
  return (
    <div className={`meter-card ${props.state}`} data-ai-id={`meter-${props.id}`} data-ai-label={`${props.label} meter`} data-ai-ops="replaceText,addClass,removeClass,setAttribute,scrollIntoView">
      <div>
        <span>{props.label}</span>
        <strong>{displayValue}</strong>
      </div>
      <div className="meter-track">
        <span style={{ width: `${percent}%` }} />
      </div>
      <small>limit {displayMax}</small>
    </div>
  );
}

function AIConsole(props: {
  readonly status: string;
  readonly loadState: LoadState;
  readonly visionState: VisionState;
  readonly engineReady: boolean;
  readonly busy: boolean;
  readonly autoPeek: boolean;
  readonly captureMode: CaptureMode;
  readonly trace: TraceState | null;
  readonly onPeek: () => void;
  readonly onAutoPeekChange: (value: boolean) => void;
  readonly onCaptureModeChange: (value: CaptureMode) => void;
  readonly onReset: () => void;
  readonly onChangeModel: () => void;
}) {
  return (
    <aside className="ai-console" data-capture-exclude="true" aria-label="AI controls and observability">
      <div className="glass-card console-card">
        <p className="eyebrow">AI Console</p>
        <h2>Vision Loop</h2>
        <div className="status-line console-status">
          <span className={`state-dot ${props.loadState}`} />
          <strong>{props.status}</strong>
        </div>

        <button className="primary-button" type="button" onClick={props.onPeek} disabled={!props.engineReady || props.busy}>
          {props.visionState === 'thinking' ? 'Model inspecting...' : 'Peek at UI'}
        </button>

        <div className="capture-mode-control" role="radiogroup" aria-label="Capture quality">
          {(['fast', 'detailed'] as const).map((mode) => (
            <button
              key={mode}
              type="button"
              className={props.captureMode === mode ? 'active' : ''}
              onClick={() => props.onCaptureModeChange(mode)}
            >
              {mode === 'fast' ? 'Fast capture' : 'Detailed capture'}
            </button>
          ))}
        </div>

        <label className="toggle-row">
          <input type="checkbox" checked={props.autoPeek} onChange={(event) => props.onAutoPeekChange(event.target.checked)} disabled={!props.engineReady} />
          Auto-peek after user actions
        </label>

        <div className="console-actions">
          <button className="secondary-button" type="button" onClick={props.onReset}>Reset</button>
          <button className="secondary-button" type="button" onClick={props.onChangeModel}>Change model</button>
        </div>
      </div>

      <TracePanel trace={props.trace} visionState={props.visionState} />
    </aside>
  );
}

function TracePanel(props: { readonly trace: TraceState | null; readonly visionState: VisionState }) {
  const trace = props.trace;
  const stages: readonly { id: VisionState | 'done'; label: string }[] = [
    { id: 'capturing', label: 'Capture UI' },
    { id: 'compressing', label: 'Compress' },
    { id: 'thinking', label: 'Vision inspect' },
    { id: 'patching', label: 'Patch DOM' },
  ];
  return (
    <div className="glass-card trace-card">
      <p className="eyebrow">Behind the Scenes</p>
      <h2>Model Trace</h2>
      <div className="pipeline">
        {stages.map((stage) => (
          <span key={stage.id} className={stage.id === props.visionState || (trace?.visionState === 'idle' && stage.id === 'patching') ? 'active' : ''}>
            {stage.label}
          </span>
        ))}
      </div>
      {!trace ? (
        <p className="empty-trace">No peek yet. The first run will show the screenshot, model JSON, validation, and applied DOM mutations here.</p>
      ) : (
        <div className="trace-stack">
          <div className="trace-meta">
            <span>#{trace.id}</span>
            <span>{trace.source}</span>
            <span>{trace.captureMode ?? 'fast'}</span>
            <span>{trace.startedAt}</span>
            {trace.durationMs ? <span>{(trace.durationMs / 1000).toFixed(1)}s</span> : null}
          </div>
          <p className="trace-status">{trace.status}</p>
          {trace.screenshotUrl ? (
            <figure className="screenshot-preview">
              <img src={trace.screenshotUrl} alt="Captured UI sent to vision model" />
              <figcaption>
                {trace.screenshotWidth ?? '?'}×{trace.screenshotHeight ?? '?'} · {formatBytes(trace.screenshotBytes ?? 0)} · {trace.targetCount ?? 0} targets
              </figcaption>
            </figure>
          ) : null}
          {trace.observation ? <TraceBlock title="Observation" content={trace.observation} /> : null}
          {trace.intent ? <TraceBlock title="Intent" content={trace.intent} /> : null}
          {trace.patches && trace.patches.length > 0 ? (
            <TraceList title="Accepted Patches" items={trace.patches.map((patch) => `${patch.op} → ${patch.targetId}`)} />
          ) : null}
          {trace.appliedMutations && trace.appliedMutations.length > 0 ? (
            <TraceList title="DOM Mutations" items={trace.appliedMutations.map((mutation) => mutation.summary)} />
          ) : null}
          {trace.rejectedPatches && trace.rejectedPatches.length > 0 ? (
            <TraceList title="Rejected Patches" items={trace.rejectedPatches.map((patch) => `#${patch.index}: ${patch.reason}`)} tone="warning" />
          ) : null}
          {trace.errorMessage ? <TraceBlock title="Error" content={trace.errorMessage} tone="warning" /> : null}
          {trace.rawText ? <TraceCode title="Raw JSON" content={trace.rawText} /> : null}
          {trace.promptPreview ? <TraceCode title="Prompt Contract Preview" content={trace.promptPreview} /> : null}
        </div>
      )}
    </div>
  );
}

function TraceBlock(props: { readonly title: string; readonly content: string; readonly tone?: 'warning' }) {
  return (
    <section className={`trace-block ${props.tone ?? ''}`}>
      <h3>{props.title}</h3>
      <p>{props.content}</p>
    </section>
  );
}

function TraceList(props: { readonly title: string; readonly items: readonly string[]; readonly tone?: 'warning' }) {
  return (
    <section className={`trace-block ${props.tone ?? ''}`}>
      <h3>{props.title}</h3>
      <ul>
        {props.items.map((item, index) => <li key={`${item}-${index}`}>{item}</li>)}
      </ul>
    </section>
  );
}

function TraceCode(props: { readonly title: string; readonly content: string }) {
  return (
    <details className="trace-code">
      <summary>{props.title}</summary>
      <pre>{props.content}</pre>
    </details>
  );
}

async function encodeCapture(canvas: HTMLCanvasElement, mode: CaptureMode): Promise<CapturedImage> {
  const preset = CAPTURE_PRESETS[mode];
  const scale = Math.min(1, preset.maxWidth / canvas.width);
  const width = Math.max(1, Math.round(canvas.width * scale));
  const height = Math.max(1, Math.round(canvas.height * scale));
  const outputCanvas = document.createElement('canvas');
  outputCanvas.width = width;
  outputCanvas.height = height;
  const context = outputCanvas.getContext('2d');
  if (!context) {
    throw new Error('Could not create capture canvas context.');
  }
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = mode === 'fast' ? 'medium' : 'high';
  context.drawImage(canvas, 0, 0, width, height);
  const blob = await canvasToBlob(outputCanvas, 'image/jpeg', preset.quality);
  const bytes = new Uint8Array(await blob.arrayBuffer());
  return {
    bytes,
    url: URL.createObjectURL(blob),
    width,
    height,
    byteLength: bytes.byteLength,
  };
}

function canvasToBlob(canvas: HTMLCanvasElement, mimeType: string, quality: number): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) {
        resolve(blob);
        return;
      }
      reject(new Error('Failed to encode capture image.'));
    }, mimeType, quality);
  });
}

function summarizePatches(patches: readonly DomPatch[]): readonly AppliedMutation[] {
  return patches.map((patch) => ({
    targetId: patch.targetId,
    summary: `${patch.op} applied to ${patch.targetId}`,
  }));
}

function clearTransientAiClasses(root: HTMLElement, classNames: readonly string[]): void {
  for (const element of Array.from(root.querySelectorAll<HTMLElement>('[data-ai-id]'))) {
    for (const className of classNames) {
      element.classList.remove(className);
    }
  }
  deleteModifiedFlags(root);
}

function deleteModifiedFlags(root: HTMLElement): void {
  for (const element of Array.from(root.querySelectorAll<HTMLElement>('[data-ai-modified]'))) {
    delete element.dataset.aiModified;
  }
}

function describeVisionState(state: VisionState, mode: CaptureMode): string {
  switch (state) {
    case 'capturing':
      return 'Capturing main UI as image';
    case 'compressing':
      return `${mode === 'fast' ? 'Fast' : 'Detailed'} image compression`;
    case 'thinking':
      return 'Vision model inspecting screenshot';
    case 'patching':
      return 'Applying JSON DOM patches';
    case 'idle':
    case 'error':
      return '';
  }
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}
