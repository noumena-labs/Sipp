import { useEffect, useRef, useState } from 'react';
import { flushSync } from 'react-dom';
import { CogentEngine } from '@noumena-labs/cogent-engine';
import { toBlob } from 'html-to-image';
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
  <h3 class="ai-gen-title">Vision coach standby</h3>
  <p class="ai-gen-note">Load a vision model, pick a few items, then ask Cogent Engine to peek at this UI and patch the DOM.</p>
`;

type LoadState = 'idle' | 'loading' | 'ready' | 'error';
type VisionState = 'idle' | 'capturing' | 'thinking' | 'patching' | 'error';
type PeekSource = 'manual' | 'auto';

interface TraceState {
  readonly id: number;
  readonly source: PeekSource;
  readonly visionState: VisionState;
  readonly status: string;
  readonly startedAt: string;
  readonly durationMs?: number;
  readonly screenshotUrl?: string;
  readonly screenshotBytes?: number;
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

export default function App() {
  const [modelUrl, setModelUrl] = useState(DEFAULT_MODEL_URL);
  const [projectorUrl, setProjectorUrl] = useState(DEFAULT_PROJECTOR_URL);
  const [loadState, setLoadState] = useState<LoadState>('idle');
  const [status, setStatus] = useState('Load a vision model to begin.');
  const [visionState, setVisionState] = useState<VisionState>('idle');
  const [activeTab, setActiveTab] = useState<GearTab>('brief');
  const [selectedIds, setSelectedIds] = useState<ReadonlySet<string>>(() => new Set(['paper-map']));
  const [autoPeek, setAutoPeek] = useState(false);
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
  const visibleGear = GEAR_ITEMS.filter((item) => activeTab === 'brief' || item.tab === activeTab);
  const engineReady = loadState === 'ready' && engineRef.current != null;
  const busy = loadState === 'loading' || visionState === 'capturing' || visionState === 'thinking' || visionState === 'patching';
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
    setStatus('Creating Cogent Engine instance...');

    try {
      const nextEngine = await CogentEngine.create();
      setStatus('Downloading vision model and projector...');
      await nextEngine.models.load(
        { model: trimmedModel, projector: trimmedProjector },
        {
          onProgress: (progress) => {
            if (progress.phase === 'download') {
              const asset = progress.assetName ? ` ${progress.assetName}` : '';
              setStatus(`Downloading${asset}... ${Math.floor(progress.percent ?? 0)}%`);
            } else if (progress.phase === 'load') {
              setStatus('Loading model into memory...');
            }
          },
          runtime: {
            imageMinTokens: 64,
            imageMaxTokens: 1024,
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
      setLoadState('ready');
      setStatus('Vision model ready. Select gear, then run a peek.');
    } catch (error) {
      setLoadState('error');
      setStatus(`Load failed: ${(error as Error).message}`);
    }
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
    setTrace({ id, source, startedAt, visionState: 'capturing', status: 'Capturing annotated UI...' });

    try {
      const blob = await toBlob(root, {
        cacheBust: true,
        pixelRatio: Math.min(window.devicePixelRatio || 1, 1.5),
        backgroundColor: '#150f0a',
        filter: (node) => !(node instanceof HTMLElement && node.dataset.captureExclude === 'true'),
      });
      if (!blob) {
        throw new Error('html-to-image returned an empty capture.');
      }

      const screenshotUrl = URL.createObjectURL(blob);
      if (screenshotUrlRef.current) {
        URL.revokeObjectURL(screenshotUrlRef.current);
      }
      screenshotUrlRef.current = screenshotUrl;
      const screenshot = new Uint8Array(await blob.arrayBuffer());
      const targets = collectPatchTargets(root);

      setVisionState('thinking');
      setTrace({
        id,
        source,
        startedAt,
        screenshotUrl,
        screenshotBytes: screenshot.byteLength,
        targetCount: targets.length,
        visionState: 'thinking',
        status: `Sending screenshot and ${targets.length} DOM targets to the model...`,
      });

      const director = new DomPatchDirector(engine, directorConfig);
      const result = await director.run({
        screenshot,
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
        screenshotUrl,
        screenshotBytes: screenshot.byteLength,
        targetCount: result.targetCount,
        promptPreview: result.promptPreview,
        rawText: result.rawText,
        observation: result.observation,
        intent: result.intent,
        patches: result.patches,
        rejectedPatches: result.rejectedPatches,
        appliedMutations,
        visionState: 'patching',
        status: `Validated ${result.patches.length} patch(es); applying DOM mutations...`,
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
    setStatus(engineReady ? 'Run reset. Pick gear and peek again.' : 'Run reset. Load a vision model to begin.');
    if (workspaceRef.current) {
      clearTransientAiClasses(workspaceRef.current, directorConfig.patchPolicy.allowedClasses);
      deleteModifiedFlags(workspaceRef.current);
    }
  };

  return (
    <div className="proactive-app">
      <header className="hero-panel">
        <div>
          <p className="eyebrow">Cogent Engine Vision Pipeline</p>
          <h1>Proactive UI: Dust Ridge Field Kit</h1>
          <p className="hero-copy">
            A vision model peeks at the UI, explains what the user is doing, emits strict JSON DOM
            patches, and the app validates and applies those patches live.
          </p>
        </div>
        <div className="status-card" data-ai-id="hud-status" data-ai-label="Top-level app status" data-ai-ops="replaceText,addClass,removeClass,setAttribute">
          <span className={`state-dot ${loadState}`} />
          <strong>{status}</strong>
        </div>
      </header>

      <main className="app-grid">
        <section className="control-column" data-capture-exclude="true">
          <div className="glass-card setup-card">
            <p className="eyebrow">Setup</p>
            <label>
              Model URL
              <input value={modelUrl} onChange={(event) => setModelUrl(event.target.value)} disabled={loadState === 'loading'} />
            </label>
            <label>
              Projector URL
              <input value={projectorUrl} onChange={(event) => setProjectorUrl(event.target.value)} disabled={loadState === 'loading'} />
            </label>
            <button className="primary-button" type="button" onClick={handleLoad} disabled={loadState === 'loading'}>
              {loadState === 'loading' ? 'Loading vision model...' : 'Load Vision Model'}
            </button>
          </div>

          <div className="glass-card action-card">
            <p className="eyebrow">Vision Loop</p>
            <button className="primary-button" type="button" onClick={() => void handleVisionPeek('manual')} disabled={!engineReady || busy}>
              {visionState === 'thinking' ? 'Model inspecting...' : 'Peek at UI'}
            </button>
            <label className="toggle-row">
              <input type="checkbox" checked={autoPeek} onChange={(event) => setAutoPeek(event.target.checked)} disabled={!engineReady} />
              Auto-peek after user actions
            </label>
            <button className="secondary-button" type="button" onClick={resetRun}>Reset toy run</button>
          </div>
        </section>

        <section className="workspace-shell">
          <div className={`capture-stage ${visionState}`} ref={workspaceRef} data-ai-zone="dust-ridge-field-kit" data-ai-goal="Help the user finish a safe desert expedition kit.">
            <div className="stage-topline">
              <div>
                <p className="eyebrow">Mission Board</p>
                <h2>Launch before the sand wall hits</h2>
              </div>
              <div className="storm-clock" data-ai-id="storm-clock" data-ai-label="Storm arrival countdown" data-ai-ops="replaceText,addClass,removeClass,setAttribute">
                <span>{FIELD_KIT_LIMITS.stormMinutes}</span>
                <small>min to storm</small>
              </div>
            </div>

            {visionState !== 'idle' && visionState !== 'error' ? (
              <div className="vision-overlay">
                <div className="scanner" />
                <strong>{visionState === 'capturing' ? 'Capturing UI as image' : visionState === 'thinking' ? 'Vision model inspecting screenshot' : 'Applying JSON DOM patches'}</strong>
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
                  {tab === 'brief' ? '1. Brief' : tab === 'gear' ? '2. Pack Gear' : '3. Launch'}
                </button>
              ))}
            </nav>

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
              </aside>

              <section className="gear-zone">
                <div className="score-strip">
                  <Meter id="readiness" label="Readiness" value={score.readiness} max={100} suffix="%" state={score.readyToLaunch ? 'good' : score.readiness > 65 ? 'warn' : 'low'} />
                  <Meter id="weight" label="Weight" value={score.totalWeight} max={FIELD_KIT_LIMITS.maxWeight} suffix="kg" state={score.weightOk ? 'good' : 'bad'} />
                  <Meter id="budget" label="Budget" value={score.totalCost} max={FIELD_KIT_LIMITS.maxBudget} suffix="$" state={score.budgetOk ? 'good' : 'bad'} />
                </div>

                <div className="brief-card glass-card" data-ai-id="brief-card" data-ai-label="Current mission brief" data-ai-ops="replaceHtml,appendHtml,addClass,removeClass,setAttribute">
                  {activeTab === 'brief' ? (
                    <>
                      <h3>Field objective</h3>
                      <p>
                        Cross Dust Ridge, mark the old relay tower, and return before visibility collapses.
                        The UI is intentionally game-like so the vision model can infer progress from visible state.
                      </p>
                    </>
                  ) : activeTab === 'gear' ? (
                    <>
                      <h3>Pack phase</h3>
                      <p>Select gear that covers every mission need without blowing the carry weight or budget.</p>
                    </>
                  ) : (
                    <>
                      <h3>Launch gate</h3>
                      <p>Final safety checks live here. If something is missing, the model should patch the UI to make it obvious.</p>
                    </>
                  )}
                </div>

                <div className="gear-grid" aria-label="Gear cards">
                  {visibleGear.map((item) => {
                    const selected = selectedIds.has(item.id);
                    return (
                      <button
                        key={item.id}
                        type="button"
                        className={`gear-card ${selected ? 'selected' : ''}`}
                        onClick={() => toggleGear(item.id)}
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

              <aside className="coach-column">
                <div
                  className="coach-panel glass-card"
                  data-ai-id="coach-panel"
                  data-ai-label="Generated proactive coach panel"
                  data-ai-ops="replaceHtml,appendHtml,addClass,removeClass,setAttribute,scrollIntoView"
                  dangerouslySetInnerHTML={{ __html: INITIAL_COACH_HTML }}
                />

                <div className="launch-panel glass-card" data-ai-id="launch-panel" data-ai-label="Launch readiness panel" data-ai-ops="replaceHtml,appendHtml,addClass,removeClass,setAttribute,scrollIntoView">
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
                </div>
              </aside>
            </div>
          </div>
        </section>

        <TracePanel trace={trace} visionState={visionState} />
      </main>
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

function TracePanel(props: { readonly trace: TraceState | null; readonly visionState: VisionState }) {
  const trace = props.trace;
  const stages = [
    { id: 'capturing', label: 'Capture UI' },
    { id: 'thinking', label: 'Vision inspect' },
    { id: 'patching', label: 'Validate JSON' },
    { id: 'idle', label: 'Patch DOM' },
  ];
  return (
    <aside className="trace-panel" data-capture-exclude="true">
      <div className="glass-card trace-card">
        <p className="eyebrow">Behind the Scenes</p>
        <h2>Vision-to-DOM Trace</h2>
        <div className="pipeline">
          {stages.map((stage) => (
            <span key={stage.id} className={stage.id === props.visionState || (trace?.visionState === 'idle' && stage.id === 'idle') ? 'active' : ''}>
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
              <span>{trace.startedAt}</span>
              {trace.durationMs ? <span>{(trace.durationMs / 1000).toFixed(1)}s</span> : null}
            </div>
            <p className="trace-status">{trace.status}</p>
            {trace.screenshotUrl ? (
              <figure className="screenshot-preview">
                <img src={trace.screenshotUrl} alt="Captured UI sent to vision model" />
                <figcaption>{formatBytes(trace.screenshotBytes ?? 0)} · {trace.targetCount ?? 0} targets</figcaption>
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
    </aside>
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

function formatBytes(bytes: number): string {
  if (bytes < 1024) {
    return `${bytes} B`;
  }
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}
