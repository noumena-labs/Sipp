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
  type GearItem,
} from './game-data';

const DEFAULT_MODEL_URL =
  'https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF/resolve/main/LFM2.5-VL-450M-F16.gguf';
const DEFAULT_PROJECTOR_URL =
  'https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF/resolve/main/mmproj-LFM2.5-VL-450m-F16.gguf';
const DIRECTOR_CONFIG_URL = '/directors/field-kit/dom-patch-director.json';
const INITIAL_COACH_HTML = `
  <h3 class="ai-gen-title">Proactive coach</h3>
  <p class="ai-gen-note">Click gear cards to pack the kit, then run a peek. Cogent Engine can highlight what matters and attach a helpful note.</p>
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

const CAPTURE_PRESETS: Record<CaptureMode, { maxWidth: number; maxHeight: number; quality: number }> = {
  fast: { maxWidth: 560, maxHeight: 760, quality: 0.62 },
  detailed: { maxWidth: 1120, maxHeight: 1400, quality: 0.86 },
};

export default function App() {
  const [modelUrl, setModelUrl] = useState(DEFAULT_MODEL_URL);
  const [projectorUrl, setProjectorUrl] = useState(DEFAULT_PROJECTOR_URL);
  const [loadState, setLoadState] = useState<LoadState>('idle');
  const [loadProgress, setLoadProgress] = useState<LoadProgressState | null>(null);
  const [status, setStatus] = useState('Load a vision model to begin.');
  const [visionState, setVisionState] = useState<VisionState>('idle');
  const [selectedIds, setSelectedIds] = useState<ReadonlySet<string>>(() => new Set(['paper-map']));
  const [autoPeek, setAutoPeek] = useState(false);
  const [captureMode, setCaptureMode] = useState<CaptureMode>('fast');
  const [drawerOpen, setDrawerOpen] = useState(true);
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
    }, 1600);
    return () => window.clearTimeout(timeout);
  }, [autoPeek, engineReady, selectedKey]);

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
            imageMinTokens: 48,
            imageMaxTokens: 256,
            sampling: {
              temperature: 0.12,
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
      setDrawerOpen(true);
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
    clearTransientAiArtifacts(root, directorConfig.patchPolicy.allowedClasses);

    const id = ++traceIdRef.current;
    const started = performance.now();
    const startedAt = new Date().toLocaleTimeString();
    setVisionState('capturing');
    setTrace({ id, source, startedAt, visionState: 'capturing', captureMode, status: 'Capturing field-kit board...' });

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
        status: `Sending ${formatBytes(captured.byteLength)} image and ${targets.length} target ids to the model...`,
      });

      const director = new DomPatchDirector(engine, directorConfig);
      const result = await director.run({
        screenshot: captured.bytes,
        targets,
        gameState: { score, selectedItems: score.selectedItems },
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
    setVisionState('idle');
    setStatus('Run reset. Pick gear and peek again.');
    if (workspaceRef.current) {
      clearTransientAiArtifacts(workspaceRef.current, directorConfig.patchPolicy.allowedClasses);
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
    <div className={`proactive-app demo-mode ${drawerOpen ? 'drawer-open' : 'drawer-closed'}`}>
      <main className="workspace-shell" aria-label="Dust Ridge Field Kit game">
        <div className={`capture-stage ${visionState}`} ref={workspaceRef} data-ai-zone="dust-ridge-field-kit" data-ai-goal="Help the user finish a safe desert expedition kit.">
          <div className="stage-topline">
            <div>
              <p className="eyebrow">Dust Ridge Field Kit</p>
              <h1>Pack the kit before the sand wall hits</h1>
              <p className="stage-subtitle">Click gear cards to pack or unpack them. Cover all six survival needs while staying under {FIELD_KIT_LIMITS.maxWeight}kg and ${FIELD_KIT_LIMITS.maxBudget}.</p>
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

          <section className="mission-instructions" data-ai-id="mission-instructions" data-ai-label="Main mission instructions" data-ai-ops="replaceHtml,appendHtml,addClass,removeClass,setAttribute">
            <p className="eyebrow">Current Objective</p>
            <h2>Choose a balanced desert kit.</h2>
            <p>Goal: pack one or more items covering Hydration, Navigation, Sun cover, Signal, Power, and First aid. Then ask the vision model to peek and help patch the UI.</p>
          </section>

          <div className="score-strip">
            <Meter id="readiness" label="Readiness" value={score.readiness} max={100} suffix="%" state={score.readyToLaunch ? 'good' : score.readiness > 65 ? 'warn' : 'low'} />
            <Meter id="weight" label="Weight" value={score.totalWeight} max={FIELD_KIT_LIMITS.maxWeight} suffix="kg" state={score.weightOk ? 'good' : 'bad'} />
            <Meter id="budget" label="Budget" value={score.totalCost} max={FIELD_KIT_LIMITS.maxBudget} suffix="$" state={score.budgetOk ? 'good' : 'bad'} />
          </div>

          <div className="board-layout">
            <section className="gear-zone" data-ai-id="gear-board" data-ai-label="All selectable field kit gear" data-ai-ops="addClass,removeClass,setAttribute,scrollIntoView">
              <div className="instruction-card glass-card" data-ai-id="pack-instructions" data-ai-label="Pack gear instructions" data-ai-ops="replaceHtml,appendHtml,addClass,removeClass,setAttribute">
                <h3>Field shelf</h3>
                <p>Every card is visible at once so the vision model can reason directly from the UI. Green cards are already packed.</p>
              </div>
              <div className="category-grid">
                {CATEGORY_GOALS.map((goal) => (
                  <GearCategorySection
                    key={goal.id}
                    goal={goal}
                    items={GEAR_ITEMS.filter((item) => item.category === goal.id)}
                    selectedIds={selectedIds}
                    onToggleGear={toggleGear}
                  />
                ))}
              </div>
            </section>

            <aside className="mission-rail">
              <section className="mission-panel glass-card" data-ai-id="mission-checklist" data-ai-label="Mission checklist" data-ai-ops="addClass,removeClass,setAttribute,scrollIntoView">
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
              </section>

              <section className="selected-manifest glass-card" data-ai-id="selected-manifest" data-ai-label="Selected gear manifest" data-ai-ops="replaceHtml,appendHtml,addClass,removeClass,setAttribute,scrollIntoView">
                <p className="eyebrow">Packed Items</p>
                {score.selectedItems.length === 0 ? (
                  <small>No items packed.</small>
                ) : (
                  <ul>
                    {score.selectedItems.map((item) => <li key={item.id}>{item.name}</li>)}
                  </ul>
                )}
              </section>

              <section className="launch-panel glass-card" data-ai-id="launch-panel" data-ai-label="Launch readiness panel" data-ai-ops="replaceHtml,appendHtml,addClass,removeClass,setAttribute,scrollIntoView">
                <p className="eyebrow">Expedition Gate</p>
                <h3>{score.readyToLaunch ? 'Ready to launch' : 'Not safe yet'}</h3>
                <p>
                  {score.readyToLaunch
                    ? 'All required categories are covered within constraints.'
                    : score.missingCategories.length > 0
                      ? `Missing: ${score.missingCategories.map(categoryLabel).join(', ')}.`
                      : 'Coverage is complete, but weight or budget needs attention.'}
                </p>
                <button
                  className="launch-button"
                  type="button"
                  disabled={!score.readyToLaunch}
                  data-ai-id="launch-button"
                  data-ai-label="Launch expedition button"
                  data-ai-ops="replaceText,addClass,removeClass,setAttribute,scrollIntoView"
                >
                  {score.readyToLaunch ? 'Launch Expedition' : 'Locked'}
                </button>
              </section>

              <section
                className="coach-panel glass-card"
                data-ai-id="coach-panel"
                data-ai-label="Generated proactive coach panel"
                data-ai-ops="replaceHtml,appendHtml,addClass,removeClass,setAttribute,scrollIntoView"
                dangerouslySetInnerHTML={{ __html: INITIAL_COACH_HTML }}
              />
            </aside>
          </div>
        </div>
      </main>

      <DeveloperDrawer
        open={drawerOpen}
        status={status}
        loadState={loadState}
        visionState={visionState}
        engineReady={engineReady}
        busy={busy}
        autoPeek={autoPeek}
        captureMode={captureMode}
        trace={trace}
        onOpenChange={setDrawerOpen}
        onPeek={() => void handleVisionPeek('manual')}
        onAutoPeekChange={setAutoPeek}
        onCaptureModeChange={setCaptureMode}
        onReset={resetRun}
        onChangeModel={handleChangeModel}
      />
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
          <h1>Proactive UI that sees, reasons, and patches the DOM.</h1>
          <p>
            This demo loads a local vision model, captures a web app as an image, asks what the user
            appears to be doing, then applies validated JSON DOM patches with visible notes.
          </p>
          <div className="start-steps">
            <span>1. Load model</span>
            <span>2. Pack the field kit</span>
            <span>3. Peek at UI</span>
            <span>4. Watch DOM notes land</span>
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

function GearCategorySection(props: {
  readonly goal: typeof CATEGORY_GOALS[number];
  readonly items: readonly GearItem[];
  readonly selectedIds: ReadonlySet<string>;
  readonly onToggleGear: (itemId: string) => void;
}) {
  const covered = props.items.some((item) => props.selectedIds.has(item.id));
  return (
    <section className={`category-section ${covered ? 'covered' : 'missing'}`} data-ai-id={`category-${props.goal.id}`} data-ai-label={`${props.goal.label} gear section`} data-ai-ops="addClass,removeClass,setAttribute,scrollIntoView">
      <header>
        <div>
          <span>{covered ? 'covered' : 'missing'}</span>
          <h3>{props.goal.label}</h3>
        </div>
        <small>{props.goal.prompt}</small>
      </header>
      <div className="gear-grid">
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
    </section>
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

function DeveloperDrawer(props: {
  readonly open: boolean;
  readonly status: string;
  readonly loadState: LoadState;
  readonly visionState: VisionState;
  readonly engineReady: boolean;
  readonly busy: boolean;
  readonly autoPeek: boolean;
  readonly captureMode: CaptureMode;
  readonly trace: TraceState | null;
  readonly onOpenChange: (open: boolean) => void;
  readonly onPeek: () => void;
  readonly onAutoPeekChange: (value: boolean) => void;
  readonly onCaptureModeChange: (value: CaptureMode) => void;
  readonly onReset: () => void;
  readonly onChangeModel: () => void;
}) {
  if (!props.open) {
    return (
      <div className="dev-pill" data-capture-exclude="true">
        <button type="button" className="dev-peek" onClick={props.onPeek} disabled={!props.engineReady || props.busy}>Peek</button>
        <button type="button" className="dev-pill-status" onClick={() => props.onOpenChange(true)}>
          <span className={`console-led ${props.visionState === 'error' ? 'error' : props.busy ? 'busy' : 'ready'}`} />
          <span>{props.trace?.durationMs ? `${(props.trace.durationMs / 1000).toFixed(1)}s` : 'trace'}</span>
        </button>
      </div>
    );
  }

  return (
    <aside className="dev-drawer" data-capture-exclude="true" aria-label="AI developer console">
      <div className="dev-drawer-titlebar">
        <div>
          <span className="console-led ready" />
          <strong>AI_TRACE</strong>
        </div>
        <button type="button" onClick={() => props.onOpenChange(false)}>minimize</button>
      </div>

      <section className="dev-section controls-section">
        <div className="dev-section-header">
          <span>controls</span>
          <code>{props.captureMode}</code>
        </div>
        <p className="dev-status-line">
          <span className={`console-led ${props.visionState === 'error' ? 'error' : props.busy ? 'busy' : 'ready'}`} />
          {props.status}
        </p>
        <button className="terminal-button primary" type="button" onClick={props.onPeek} disabled={!props.engineReady || props.busy}>
          {props.visionState === 'thinking' ? 'model.inspecting()' : 'peek.ui()'}
        </button>
        <div className="capture-mode-control" role="radiogroup" aria-label="Capture quality">
          {(['fast', 'detailed'] as const).map((mode) => (
            <button key={mode} type="button" className={props.captureMode === mode ? 'active' : ''} onClick={() => props.onCaptureModeChange(mode)}>
              {mode}
            </button>
          ))}
        </div>
        <label className="terminal-toggle">
          <input type="checkbox" checked={props.autoPeek} onChange={(event) => props.onAutoPeekChange(event.target.checked)} disabled={!props.engineReady} />
          autoPeek.onUserAction
        </label>
        <div className="drawer-actions">
          <button className="terminal-button" type="button" onClick={props.onReset}>reset()</button>
          <button className="terminal-button" type="button" onClick={props.onChangeModel}>model.swap()</button>
        </div>
      </section>

      <TracePanel trace={props.trace} visionState={props.visionState} />
    </aside>
  );
}

function TracePanel(props: { readonly trace: TraceState | null; readonly visionState: VisionState }) {
  const trace = props.trace;
  const stages: readonly { id: VisionState; label: string }[] = [
    { id: 'capturing', label: 'capture' },
    { id: 'compressing', label: 'compress' },
    { id: 'thinking', label: 'infer' },
    { id: 'patching', label: 'patch' },
  ];
  return (
    <section className="dev-section trace-section">
      <div className="dev-section-header">
        <span>model_trace</span>
        <code>{trace ? `#${trace.id}` : 'idle'}</code>
      </div>
      <div className="pipeline">
        {stages.map((stage) => (
          <span key={stage.id} className={stage.id === props.visionState || (trace?.visionState === 'idle' && stage.id === 'patching') ? 'active' : ''}>
            {stage.label}
          </span>
        ))}
      </div>
      {!trace ? (
        <p className="empty-trace">No run yet. Click <code>peek.ui()</code> to capture the board, run vision inference, and inspect validated DOM patches.</p>
      ) : (
        <div className="trace-stack">
          <div className="trace-meta">
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
          {trace.observation ? <TraceBlock title="observation" content={trace.observation} /> : null}
          {trace.intent ? <TraceBlock title="intent" content={trace.intent} /> : null}
          {trace.patches && trace.patches.length > 0 ? (
            <TraceList title="accepted_patches" items={trace.patches.map((patch) => `${patch.op} -> ${patch.targetId}${patch.note ? ` // ${patch.note}` : ''}`)} />
          ) : null}
          {trace.appliedMutations && trace.appliedMutations.length > 0 ? (
            <TraceList title="dom_mutations" items={trace.appliedMutations.map((mutation) => mutation.summary)} />
          ) : null}
          {trace.rejectedPatches && trace.rejectedPatches.length > 0 ? (
            <TraceList title="rejected_patches" items={trace.rejectedPatches.map((patch) => `#${patch.index}: ${patch.reason}`)} tone="warning" />
          ) : null}
          {trace.errorMessage ? <TraceBlock title="error" content={trace.errorMessage} tone="warning" /> : null}
          {trace.rawText ? <TraceCode title="raw_json" content={trace.rawText} /> : null}
          {trace.promptPreview ? <TraceCode title="prompt_contract" content={trace.promptPreview} /> : null}
        </div>
      )}
    </section>
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
  const scale = Math.min(1, preset.maxWidth / canvas.width, preset.maxHeight / canvas.height);
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
    summary: `${patch.op} applied to ${patch.targetId}${patch.note ? ' with note' : ''}`,
  }));
}

function clearTransientAiArtifacts(root: HTMLElement, classNames: readonly string[]): void {
  for (const element of Array.from(root.querySelectorAll<HTMLElement>('[data-ai-id]'))) {
    for (const className of classNames) {
      element.classList.remove(className);
    }
  }
  for (const note of Array.from(root.querySelectorAll('[data-ai-patch-note="true"]'))) {
    note.remove();
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
      return 'Capturing field-kit board';
    case 'compressing':
      return `${mode === 'fast' ? 'Fast' : 'Detailed'} JPEG compression`;
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
