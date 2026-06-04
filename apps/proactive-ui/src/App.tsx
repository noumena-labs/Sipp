import { useEffect, useRef, useState, type PointerEvent as ReactPointerEvent } from 'react';
import { CogentClient, type RuntimeObservation } from '@noumena-labs/cogentlm';
import {
  DEFAULT_DRAWING_DIRECTOR_CONFIG,
  DRAWING_COLORS,
  DrawingDirector,
  HECKLE_VOICES,
  loadDrawingDirectorConfig,
  type CapturedDrawing,
  type CapturePresetId,
  type DrawingColor,
  type DrawingDirectorConfig,
  type HeckleVoice,
} from './drawing-director';

const DEFAULT_MODEL_URL =
  'https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF/resolve/main/LFM2.5-VL-450M-F16.gguf';
const DEFAULT_PROJECTOR_URL =
  'https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF/resolve/main/mmproj-LFM2.5-VL-450m-F16.gguf';
const DIRECTOR_CONFIG_URL = '/directors/sketch-lab/drawing-director.json';
const DRAWING_WIDTH = 800;
const DRAWING_HEIGHT = 520;
const DRAWING_BACKGROUND = '#fbf7ef';
const DRAWING_BACKGROUND_RGB = { r: 251, g: 247, b: 239 };
const INITIAL_COMMENT = 'Draw something. I am warming up my tiny art critic goggles.';
const INITIAL_GUESS = 'waiting for ink';

type LoadState = 'idle' | 'loading' | 'ready' | 'error';
type VisionState = 'idle' | 'capturing' | 'encoding' | 'thinking' | 'error';
type InferenceSource = 'manual' | 'auto';

interface LoadProgressState {
  readonly phase: string;
  readonly assetName?: string;
  readonly percent: number | null;
  readonly overallPercent: number;
}

interface ProgressAggregator {
  assetOrder: string[];
  perAssetDownloadPercent: Map<string, number>;
  perAssetStorePercent: Map<string, number>;
  loadPhaseStarted: boolean;
  loadPhasePercent: number;
}

const PROGRESS_WEIGHTS = {
  modelDownload: 55,
  modelStore: 5,
  projectorDownload: 30,
  projectorStore: 3,
  load: 7,
} as const;

function createProgressAggregator(): ProgressAggregator {
  return {
    assetOrder: [],
    perAssetDownloadPercent: new Map(),
    perAssetStorePercent: new Map(),
    loadPhaseStarted: false,
    loadPhasePercent: 0,
  };
}

function ingestProgress(agg: ProgressAggregator, phase: string, assetName: string | undefined, percent: number | null): number {
  const name = assetName ?? '__unknown__';
  if (phase === 'metadata' || phase === 'download' || phase === 'store') {
    if (!agg.assetOrder.includes(name)) {
      agg.assetOrder.push(name);
    }
  }
  const safePct = percent == null ? null : Math.max(0, Math.min(100, percent));
  if (phase === 'download' && safePct != null) {
    agg.perAssetDownloadPercent.set(name, safePct);
  } else if (phase === 'store' && safePct != null) {
    agg.perAssetStorePercent.set(name, safePct);
  } else if (phase === 'load') {
    agg.loadPhaseStarted = true;
    agg.loadPhasePercent = safePct ?? Math.max(agg.loadPhasePercent, 10);
    // When load phase begins, any earlier asset that never reported download/store
    // was served from cache. Mark them as fully complete so the bar advances.
    for (const asset of agg.assetOrder) {
      if (!agg.perAssetDownloadPercent.has(asset)) {
        agg.perAssetDownloadPercent.set(asset, 100);
      }
      if (!agg.perAssetStorePercent.has(asset)) {
        agg.perAssetStorePercent.set(asset, 100);
      }
    }
    // If we never saw a projector asset by the time load starts, assume it shared
    // the model file (single-file bundle) and credit its weights as complete.
    if (agg.assetOrder.length < 2) {
      agg.assetOrder.push('__projector_assumed__');
      agg.perAssetDownloadPercent.set('__projector_assumed__', 100);
      agg.perAssetStorePercent.set('__projector_assumed__', 100);
    }
  }

  // Resolve which asset is model (first seen) vs projector (second seen).
  const modelName = agg.assetOrder[0];
  const projectorName = agg.assetOrder[1];
  const modelDl = modelName ? agg.perAssetDownloadPercent.get(modelName) ?? 0 : 0;
  const modelStore = modelName ? agg.perAssetStorePercent.get(modelName) ?? 0 : 0;
  const projDl = projectorName ? agg.perAssetDownloadPercent.get(projectorName) ?? 0 : 0;
  const projStore = projectorName ? agg.perAssetStorePercent.get(projectorName) ?? 0 : 0;

  const overall =
    (modelDl / 100) * PROGRESS_WEIGHTS.modelDownload +
    (modelStore / 100) * PROGRESS_WEIGHTS.modelStore +
    (projDl / 100) * PROGRESS_WEIGHTS.projectorDownload +
    (projStore / 100) * PROGRESS_WEIGHTS.projectorStore +
    (agg.loadPhasePercent / 100) * PROGRESS_WEIGHTS.load;

  return Math.max(0, Math.min(100, Math.round(overall)));
}

interface CapturePreset {
  readonly maxWidth: number;
  readonly maxHeight: number;
  readonly quality: number;
}

interface CapturedDrawingAsset extends CapturedDrawing {
  readonly url: string;
  readonly composeMs: number;
  readonly encodeMs: number;
}

interface TraceState {
  readonly id: number;
  readonly source: InferenceSource;
  readonly preset: CapturePresetId;
  readonly visionState: VisionState;
  readonly status: string;
  readonly startedAt: string;
  readonly captureUrl?: string;
  readonly captureBytes?: number;
  readonly captureWidth?: number;
  readonly captureHeight?: number;
  readonly cropX?: number;
  readonly cropY?: number;
  readonly cropWidth?: number;
  readonly cropHeight?: number;
  readonly composeMs?: number;
  readonly encodeMs?: number;
  readonly inferenceMs?: number;
  readonly perceptionMs?: number;
  readonly heckleMs?: number;
  readonly totalMs?: number;
  readonly subject?: string;
  readonly features?: readonly string[];
  readonly weirdDetail?: string;
  readonly lineQuality?: string;
  readonly parseStatus?: 'parsed' | 'fallback';
  readonly parseNote?: string;
  readonly heckleParseStatus?: 'parsed' | 'fallback';
  readonly heckleParseNote?: string;
  readonly comment?: string;
  readonly guess?: string;
  readonly perceptionPromptPreview?: string;
  readonly hecklePromptPreview?: string;
  readonly perceptionRawText?: string;
  readonly heckleRawText?: string;
  readonly runtime?: RuntimeObservation | null;
  readonly errorMessage?: string;
}

interface DrawingPoint {
  readonly x: number;
  readonly y: number;
}

interface DrawingSessionRef {
  readonly pointerId: number;
  readonly points: DrawingPoint[];
  lastPoint: DrawingPoint;
}

const CAPTURE_PRESETS: Record<CapturePresetId, CapturePreset> = {
  turbo: { maxWidth: 224, maxHeight: 224, quality: 0.48 },
  trace: { maxWidth: 336, maxHeight: 336, quality: 0.68 },
};

export default function App() {
  const [modelUrl, setModelUrl] = useState(DEFAULT_MODEL_URL);
  const [projectorUrl, setProjectorUrl] = useState(DEFAULT_PROJECTOR_URL);
  const [loadState, setLoadState] = useState<LoadState>('idle');
  const [loadProgress, setLoadProgress] = useState<LoadProgressState | null>(null);
  const [status, setStatus] = useState('Load a vision model to begin.');
  const [visionState, setVisionState] = useState<VisionState>('idle');
  const [selectedColor, setSelectedColor] = useState<DrawingColor>('#111827');
  const [penSize, setPenSize] = useState(12);
  const [autoInfer, setAutoInfer] = useState(true);
  const [capturePreset, setCapturePreset] = useState<CapturePresetId>('turbo');
  const [drawerOpen, setDrawerOpen] = useState(true);
  const [trace, setTrace] = useState<TraceState | null>(null);
  const [comment, setComment] = useState(INITIAL_COMMENT);
  const [guess, setGuess] = useState(INITIAL_GUESS);
  const [heckleVoice, setHeckleVoice] = useState<HeckleVoice>(HECKLE_VOICES[0]);
  const [strokeCount, setStrokeCount] = useState(0);
  const [historyDepth, setHistoryDepth] = useState(0);
  const [runtimeObservation, setRuntimeObservation] = useState<RuntimeObservation | null>(null);
  const [directorConfig, setDirectorConfig] = useState<DrawingDirectorConfig>(DEFAULT_DRAWING_DIRECTOR_CONFIG);

  const userCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const clientRef = useRef<CogentClient | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const unsubscribeObservabilityRef = useRef<(() => void) | null>(null);
  const captureUrlRef = useRef<string | null>(null);
  const activeDrawingRef = useRef<DrawingSessionRef | null>(null);
  const historyRef = useRef<ImageData[]>([]);
  const autoTimerRef = useRef<number | null>(null);
  const traceIdRef = useRef(0);
  const strokeCountRef = useRef(0);
  const autoInferRef = useRef(autoInfer);
  const runtimeObservationRef = useRef<RuntimeObservation | null>(null);
  const runInferenceRef = useRef<((source: InferenceSource, preset?: CapturePresetId) => Promise<void>) | null>(null);

  const clientReady = loadState === 'ready' && clientRef.current != null;
  const busy = loadState === 'loading' || (visionState !== 'idle' && visionState !== 'error');
  autoInferRef.current = autoInfer;
  runtimeObservationRef.current = runtimeObservation;

  useEffect(() => {
    let cancelled = false;
    void loadDrawingDirectorConfig(DIRECTOR_CONFIG_URL)
      .then((config) => {
        if (!cancelled) {
          setDirectorConfig(config);
        }
      })
      .catch((error) => {
        if (!cancelled) {
          setStatus(`Using built-in drawing director: ${(error as Error).message}`);
        }
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    return () => {
      abortRef.current?.abort();
      unsubscribeObservabilityRef.current?.();
      void clientRef.current?.close();
      if (captureUrlRef.current) {
        URL.revokeObjectURL(captureUrlRef.current);
      }
      if (autoTimerRef.current != null) {
        window.clearTimeout(autoTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (!clientReady) {
      return;
    }
    resetUserCanvas();
    setStatus('Canvas ready. Draw and the model will heckle after each stroke.');
  }, [clientReady]);

  const handleLoad = async (): Promise<void> => {
    const trimmedModel = modelUrl.trim();
    const trimmedProjector = projectorUrl.trim();
    if (!trimmedModel || !trimmedProjector) {
      setStatus('Provide both model and projector URLs for the vision pipeline.');
      setLoadState('error');
      return;
    }

    abortRef.current?.abort();
    unsubscribeObservabilityRef.current?.();
    unsubscribeObservabilityRef.current = null;
    clearScheduledInference();
    setRuntimeObservation(null);
    runtimeObservationRef.current = null;
    setLoadState('loading');
    const progressAgg = createProgressAggregator();
    setLoadProgress({ phase: 'create', percent: null, overallPercent: 1 });
    setStatus('Creating CogentClient instance...');

    let nextClient: CogentClient | null = null;
    try {
      nextClient = new CogentClient();
      unsubscribeObservabilityRef.current = nextClient.observability.subscribe((event) => {
        setRuntimeObservation(event.snapshot.runtime ?? null);
        runtimeObservationRef.current = event.snapshot.runtime ?? null;
      });

      setStatus('Downloading vision model and projector...');
      await nextClient.addLocal(
        { model: trimmedModel, projector: trimmedProjector },
        {
          observability: 'runtime',
          onProgress: (progress) => {
            const overallPercent = ingestProgress(progressAgg, progress.phase, progress.assetName, progress.percent);
            setLoadProgress({
              phase: progress.phase,
              ...(progress.assetName ? { assetName: progress.assetName } : {}),
              percent: progress.percent,
              overallPercent,
            });
            if (progress.phase === 'download') {
              const asset = progress.assetName ? ` ${progress.assetName}` : '';
              setStatus(`Downloading${asset}... ${Math.floor(progress.percent ?? 0)}% (${overallPercent}% overall)`);
            } else if (progress.phase === 'load') {
              setStatus(`Loading fast vision runtime... (${overallPercent}% overall)`);
            } else if (progress.phase === 'metadata') {
              setStatus(`Resolving model metadata... (${overallPercent}% overall)`);
            } else if (progress.phase === 'store') {
              setStatus(`Storing model assets... (${overallPercent}% overall)`);
            }
          },
          runtime: {
            context: {
              n_ctx: 1024,
              n_batch: 256,
              n_ubatch: 128,
            },
            multimodal: {
              image_min_tokens: 24,
              image_max_tokens: 96,
            },
            scheduler: {
              policy: {
                mode: 'latency_first' as const,
                decode_token_reserve: 128,
              },
              prefill_chunk_size: 256,
            },
            cache: {
              retained_prefix_tokens: 256,
              snapshot_interval_tokens: 32,
            },
            sampling: {
              temperature: 0.45,
              top_p: 0.85,
              top_k: 32,
              min_p: 0.04,
              repeat_penalty: 1.05,
            },
          },
        }
      );

      void clientRef.current?.close();
      clientRef.current = nextClient;
      nextClient = null;
      setLoadProgress({ phase: 'ready', percent: 100, overallPercent: 100 });
      setLoadState('ready');
      setDrawerOpen(true);
      setStatus('Vision model ready. Fast sketch loop armed.');
    } catch (error) {
      setLoadState('error');
      setStatus(`Load failed: ${(error as Error).message}`);
      unsubscribeObservabilityRef.current?.();
      unsubscribeObservabilityRef.current = null;
      void nextClient?.close();
    }
  };

  const handleChangeModel = (): void => {
    abortRef.current?.abort();
    unsubscribeObservabilityRef.current?.();
    unsubscribeObservabilityRef.current = null;
    clearScheduledInference();
    void clientRef.current?.close();
    clientRef.current = null;
    setAutoInfer(false);
    setLoadProgress(null);
    setLoadState('idle');
    setVisionState('idle');
    setRuntimeObservation(null);
    setStatus('Load a vision model to begin.');
  };

  const runInference = async (source: InferenceSource, preset: CapturePresetId = capturePreset): Promise<void> => {
    const userCanvas = userCanvasRef.current;
    const client = clientRef.current;
    if (!userCanvas || !client) {
      setStatus('Load the vision model before asking for a roast.');
      return;
    }

    clearScheduledInference();
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    const id = ++traceIdRef.current;
    const started = performance.now();
    const startedAt = new Date().toLocaleTimeString();
    const voice = HECKLE_VOICES[id % HECKLE_VOICES.length];
    setHeckleVoice(voice);
    setVisionState('capturing');
    setTrace({ id, source, preset, startedAt, visionState: 'capturing', status: 'Capturing canvas...' });

    try {
      const captured = await captureDrawing(userCanvas, preset);
      if (controller.signal.aborted) {
        return;
      }
      if (captureUrlRef.current) {
        URL.revokeObjectURL(captureUrlRef.current);
      }
      captureUrlRef.current = captured.url;

      setVisionState('encoding');
      setTrace({
        id,
        source,
        preset,
        startedAt,
        captureUrl: captured.url,
        captureBytes: captured.byteLength,
        captureWidth: captured.width,
        captureHeight: captured.height,
        cropX: captured.cropX,
        cropY: captured.cropY,
        cropWidth: captured.cropWidth,
        cropHeight: captured.cropHeight,
        composeMs: captured.composeMs,
        encodeMs: captured.encodeMs,
        visionState: 'encoding',
        status: `${preset} capture encoded at ${captured.width}x${captured.height} (${formatBytes(captured.byteLength)}).`,
      });

      setVisionState('thinking');
      setTrace((previous) => previous?.id === id ? {
        ...previous,
        visionState: 'thinking',
        status: 'Model is describing the sketch, then writing a heckle...',
      } : previous);

      const inferenceStarted = performance.now();
      const director = new DrawingDirector(client, directorConfig);
      const result = await director.run({
        capture: captured,
        state: {
          strokeCount: strokeCountRef.current,
          selectedColor,
          selectedSize: penSize,
          canvasWidth: DRAWING_WIDTH,
          canvasHeight: DRAWING_HEIGHT,
          voice,
        },
        signal: controller.signal,
      });
      if (controller.signal.aborted) {
        return;
      }

      const inferenceMs = Math.round(performance.now() - inferenceStarted);
      setComment(result.heckle.comment);
      setGuess(result.perception.subject);
      const totalMs = Math.round(performance.now() - started);
      setTrace({
        id,
        source,
        preset,
        startedAt,
        captureUrl: captured.url,
        captureBytes: captured.byteLength,
        captureWidth: captured.width,
        captureHeight: captured.height,
        cropX: captured.cropX,
        cropY: captured.cropY,
        cropWidth: captured.cropWidth,
        cropHeight: captured.cropHeight,
        composeMs: captured.composeMs,
        encodeMs: captured.encodeMs,
        inferenceMs,
        perceptionMs: result.perceptionMs,
        heckleMs: result.heckleMs,
        totalMs,
        subject: result.perception.subject,
        features: result.perception.features,
        weirdDetail: result.perception.weirdDetail,
        lineQuality: result.perception.lineQuality,
        parseStatus: result.perception.parseStatus,
        parseNote: result.perception.parseNote,
        heckleParseStatus: result.heckle.parseStatus,
        heckleParseNote: result.heckle.parseNote,
        comment: result.heckle.comment,
        guess: result.perception.subject,
        perceptionPromptPreview: result.perceptionPromptPreview,
        hecklePromptPreview: result.hecklePromptPreview,
        perceptionRawText: result.perceptionRawText,
        heckleRawText: result.heckleRawText,
        runtime: runtimeObservationRef.current,
        visionState: 'idle',
        status: `Done in ${(totalMs / 1000).toFixed(2)}s. Perception ${result.perception.parseStatus}; heckle ${result.heckle.parseStatus}.`,
      });
      setVisionState('idle');
      setStatus(`Model heckled in ${(totalMs / 1000).toFixed(2)}s as ${voice}.`);
    } catch (error) {
      if (controller.signal.aborted) {
        return;
      }
      const totalMs = Math.round(performance.now() - started);
      setVisionState('error');
      setStatus(`Inference failed: ${(error as Error).message}`);
      setTrace((previous) => ({
        id,
        source,
        preset,
        startedAt,
        captureUrl: previous?.id === id ? previous.captureUrl : undefined,
        captureBytes: previous?.id === id ? previous.captureBytes : undefined,
        captureWidth: previous?.id === id ? previous.captureWidth : undefined,
        captureHeight: previous?.id === id ? previous.captureHeight : undefined,
        cropX: previous?.id === id ? previous.cropX : undefined,
        cropY: previous?.id === id ? previous.cropY : undefined,
        cropWidth: previous?.id === id ? previous.cropWidth : undefined,
        cropHeight: previous?.id === id ? previous.cropHeight : undefined,
        composeMs: previous?.id === id ? previous.composeMs : undefined,
        encodeMs: previous?.id === id ? previous.encodeMs : undefined,
        totalMs,
        runtime: runtimeObservationRef.current,
        visionState: 'error',
        status: 'Vision sketch loop failed.',
        errorMessage: (error as Error).message,
      }));
    }
  };

  runInferenceRef.current = runInference;

  function resetUserCanvas(): void {
    const canvas = userCanvasRef.current;
    const context = canvas?.getContext('2d');
    if (!canvas || !context) {
      return;
    }
    context.save();
    context.globalCompositeOperation = 'source-over';
    context.fillStyle = DRAWING_BACKGROUND;
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.restore();
    const snapshot = context.getImageData(0, 0, canvas.width, canvas.height);
    historyRef.current = [snapshot];
    setHistoryDepth(1);
  }

  function saveHistorySnapshot(): void {
    const canvas = userCanvasRef.current;
    const context = canvas?.getContext('2d');
    if (!canvas || !context) {
      return;
    }
    historyRef.current = [...historyRef.current.slice(-23), context.getImageData(0, 0, canvas.width, canvas.height)];
    setHistoryDepth(historyRef.current.length);
  }

  function drawDot(point: DrawingPoint, color: DrawingColor, size: number): void {
    const context = userCanvasRef.current?.getContext('2d');
    if (!context) {
      return;
    }
    context.save();
    context.fillStyle = color;
    context.beginPath();
    context.arc(point.x, point.y, size / 2, 0, Math.PI * 2);
    context.fill();
    context.restore();
  }

  function drawSegment(from: DrawingPoint, to: DrawingPoint, color: DrawingColor, size: number): void {
    const context = userCanvasRef.current?.getContext('2d');
    if (!context) {
      return;
    }
    context.save();
    context.strokeStyle = color;
    context.lineWidth = size;
    context.lineCap = 'round';
    context.lineJoin = 'round';
    context.beginPath();
    context.moveTo(from.x, from.y);
    context.lineTo(to.x, to.y);
    context.stroke();
    context.restore();
  }

  function scheduleAutoInference(): void {
    if (!autoInferRef.current) {
      return;
    }
    clearScheduledInference();
    autoTimerRef.current = window.setTimeout(() => {
      void runInferenceRef.current?.('auto', 'turbo');
    }, 260);
  }

  function clearScheduledInference(): void {
    if (autoTimerRef.current == null) {
      return;
    }
    window.clearTimeout(autoTimerRef.current);
    autoTimerRef.current = null;
  }

  const handlePointerDown = (event: ReactPointerEvent<HTMLCanvasElement>): void => {
    if (event.button !== 0 && event.pointerType !== 'touch' && event.pointerType !== 'pen') {
      return;
    }
    const point = getCanvasPoint(event.currentTarget, event.clientX, event.clientY);
    clearScheduledInference();
    abortRef.current?.abort();
    setVisionState('idle');
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    activeDrawingRef.current = { pointerId: event.pointerId, points: [point], lastPoint: point };
    drawDot(point, selectedColor, penSize);
  };

  const handlePointerMove = (event: ReactPointerEvent<HTMLCanvasElement>): void => {
    const session = activeDrawingRef.current;
    if (!session || session.pointerId !== event.pointerId) {
      return;
    }
    const point = getCanvasPoint(event.currentTarget, event.clientX, event.clientY);
    event.preventDefault();
    drawSegment(session.lastPoint, point, selectedColor, penSize);
    session.points.push(point);
    session.lastPoint = point;
  };

  const handlePointerUp = (event: ReactPointerEvent<HTMLCanvasElement>): void => {
    const session = activeDrawingRef.current;
    if (!session || session.pointerId !== event.pointerId) {
      return;
    }
    event.preventDefault();
    event.currentTarget.releasePointerCapture(event.pointerId);
    activeDrawingRef.current = null;
    saveHistorySnapshot();
    strokeCountRef.current += 1;
    setStrokeCount(strokeCountRef.current);
    setStatus('Stroke captured. Scheduling fast vision peek...');
    scheduleAutoInference();
  };

  const handlePointerCancel = (event: ReactPointerEvent<HTMLCanvasElement>): void => {
    const session = activeDrawingRef.current;
    if (!session || session.pointerId !== event.pointerId) {
      return;
    }
    activeDrawingRef.current = null;
  };

  const handleClearUser = (): void => {
    clearScheduledInference();
    abortRef.current?.abort();
    resetUserCanvas();
    strokeCountRef.current = 0;
    setStrokeCount(0);
    setComment(INITIAL_COMMENT);
    setGuess(INITIAL_GUESS);
    setVisionState('idle');
    setStatus('Drawing cleared. The critic is pretending it forgot everything.');
  };

  const handleResetAll = (): void => {
    clearScheduledInference();
    abortRef.current?.abort();
    resetUserCanvas();
    strokeCountRef.current = 0;
    setStrokeCount(0);
    setComment(INITIAL_COMMENT);
    setGuess(INITIAL_GUESS);
    setVisionState('idle');
    setStatus('Canvas reset. Fresh chaos available.');
  };

  const handleUndo = (): void => {
    const canvas = userCanvasRef.current;
    const context = canvas?.getContext('2d');
    if (!canvas || !context || historyRef.current.length <= 1) {
      return;
    }
    clearScheduledInference();
    abortRef.current?.abort();
    setVisionState('idle');
    historyRef.current.pop();
    context.putImageData(historyRef.current[historyRef.current.length - 1], 0, 0);
    setHistoryDepth(historyRef.current.length);
    strokeCountRef.current = Math.max(0, strokeCountRef.current - 1);
    setStrokeCount(strokeCountRef.current);
    setStatus('Last stroke undone. The model pretends it did not see that.');
    scheduleAutoInference();
  };

  if (!clientReady) {
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
    <div className={`proactive-app ${drawerOpen ? 'drawer-open' : 'drawer-closed'}`}>
      <main className="studio-shell" aria-label="AI drawing studio">
        <section className="studio-stage">
          <header className="studio-header">
            <div>
              <p className="eyebrow">Speed Sketch Lab</p>
              <h1>Draw. AI heckles.</h1>
              <p>Tiny canvas frames go straight to vision. The model guesses what it sees and heckles the visible marks.</p>
            </div>
            <div className={`loop-badge ${busy ? 'busy' : 'ready'}`}>
              <span />
              {busy ? describeVisionState(visionState) : 'fast loop ready'}
            </div>
          </header>

          <section className="toolbelt" aria-label="Drawing tools">
            <div className="palette" aria-label="Fixed colors">
              {DRAWING_COLORS.map((color) => (
                <button
                  key={color}
                  type="button"
                  className={selectedColor === color ? 'active' : ''}
                  style={{ background: color }}
                  onClick={() => setSelectedColor(color)}
                  aria-label={`Select ${color}`}
                />
              ))}
            </div>
            <label className="size-control">
              <span>pen {penSize}px</span>
              <input min="3" max="34" step="1" type="range" value={penSize} onChange={(event) => setPenSize(Number(event.target.value))} />
            </label>
            <div className="tool-actions">
              <button type="button" onClick={handleUndo} disabled={historyDepth <= 1}>undo</button>
              <button type="button" onClick={handleClearUser}>clear ink</button>
              <button type="button" onClick={() => void runInference('manual', 'turbo')} disabled={busy}>roast now</button>
            </div>
          </section>

          <div className="canvas-shell">
            <canvas
              ref={userCanvasRef}
              width={DRAWING_WIDTH}
              height={DRAWING_HEIGHT}
              className="drawing-canvas user-canvas"
              onPointerDown={handlePointerDown}
              onPointerMove={handlePointerMove}
              onPointerUp={handlePointerUp}
              onPointerCancel={handlePointerCancel}
              aria-label="Drawing canvas"
            />
            {busy ? (
              <div className="vision-overlay">
                <span className="scanner" />
                <strong>{describeVisionState(visionState)}</strong>
              </div>
            ) : null}
          </div>

          <section className="reaction-strip">
            <div className="reaction-card primary">
              <span>heckle</span>
              <strong>{comment}</strong>
            </div>
            <div className="reaction-card">
              <span>guess</span>
              <strong>{guess}</strong>
            </div>
            <div className="reaction-card stats">
              <span>canvas</span>
              <strong>{strokeCount} strokes / {heckleVoice}</strong>
            </div>
          </section>
        </section>
      </main>

      <DeveloperDrawer
        open={drawerOpen}
        status={status}
        loadState={loadState}
        visionState={visionState}
        clientReady={clientReady}
        busy={busy}
        autoInfer={autoInfer}
        capturePreset={capturePreset}
        trace={trace}
        runtime={runtimeObservation}
        strokeCount={strokeCount}
        heckleVoice={heckleVoice}
        onOpenChange={setDrawerOpen}
        onInfer={() => void runInference('manual', capturePreset)}
        onAutoInferChange={setAutoInfer}
        onCapturePresetChange={setCapturePreset}
        onReset={handleResetAll}
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
  const overallPercent = props.progress?.overallPercent ?? (props.loadState === 'loading' ? 1 : 0);
  const phasePercent = props.progress?.percent ?? null;
  return (
    <div className="start-screen">
      <section className="start-hero glass-card">
        <div className="start-copy">
          <p className="eyebrow">CogentClient Vision Pipeline</p>
          <h1>Vision Pipline Demo</h1>
          <p>
            This demo loads a local vision model, captures a tiny image of the drawing canvas, then asks the model to guess and heckle the sketch.
          </p>
          <div className="start-steps">
            <span>1. Load vision model</span>
            <span>2. Draw with fixed tools</span>
            <span>3. Send low-res canvas</span>
            <span>4. Get heckled</span>
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
              <span style={{ width: `${overallPercent}%` }} />
            </div>
            <div className="load-progress-numbers">
              <span>Overall: {overallPercent}%</span>
              {phasePercent != null ? <span>Phase: {Math.floor(phasePercent)}%</span> : null}
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
                <dd>{phasePercent == null ? 'loading' : `${Math.floor(phasePercent)}%`}</dd>
              </div>
            </dl>
          </div>
        </div>
      </section>
    </div>
  );
}

function DeveloperDrawer(props: {
  readonly open: boolean;
  readonly status: string;
  readonly loadState: LoadState;
  readonly visionState: VisionState;
  readonly clientReady: boolean;
  readonly busy: boolean;
  readonly autoInfer: boolean;
  readonly capturePreset: CapturePresetId;
  readonly trace: TraceState | null;
  readonly runtime: RuntimeObservation | null;
  readonly strokeCount: number;
  readonly heckleVoice: HeckleVoice;
  readonly onOpenChange: (open: boolean) => void;
  readonly onInfer: () => void;
  readonly onAutoInferChange: (value: boolean) => void;
  readonly onCapturePresetChange: (value: CapturePresetId) => void;
  readonly onReset: () => void;
  readonly onChangeModel: () => void;
}) {
  if (!props.open) {
    return (
      <div className="dev-pill">
        <button type="button" className="dev-peek" onClick={props.onInfer} disabled={!props.clientReady || props.busy}>trace</button>
        <button type="button" className="dev-pill-status" onClick={() => props.onOpenChange(true)}>
          <span className={`console-led ${props.visionState === 'error' ? 'error' : props.busy ? 'busy' : 'ready'}`} />
          <span>{props.trace?.totalMs ? `${(props.trace.totalMs / 1000).toFixed(2)}s` : 'AI_TRACE'}</span>
        </button>
      </div>
    );
  }

  return (
    <aside className="dev-drawer" aria-label="AI Trace developer console">
      <div className="dev-drawer-titlebar">
        <div>
          <span className={`console-led ${props.visionState === 'error' ? 'error' : props.busy ? 'busy' : 'ready'}`} />
          <strong>AI_TRACE</strong>
        </div>
        <button type="button" onClick={() => props.onOpenChange(false)}>minimize</button>
      </div>

      <section className="dev-section controls-section">
        <div className="dev-section-header">
          <span>controls</span>
          <code>{props.capturePreset}</code>
        </div>
        <p className="dev-status-line">
          <span className={`console-led ${props.visionState === 'error' ? 'error' : props.busy ? 'busy' : 'ready'}`} />
          {props.status}
        </p>
        <button className="terminal-button primary" type="button" onClick={props.onInfer} disabled={!props.clientReady || props.busy}>
          {props.visionState === 'thinking' ? 'model.inspecting()' : 'trace.infer()'}
        </button>
        <div className="capture-mode-control" role="radiogroup" aria-label="Capture preset">
          {(['turbo', 'trace'] as const).map((preset) => (
            <button key={preset} type="button" className={props.capturePreset === preset ? 'active' : ''} onClick={() => props.onCapturePresetChange(preset)}>
              {preset}
            </button>
          ))}
        </div>
        <label className="terminal-toggle">
          <input type="checkbox" checked={props.autoInfer} onChange={(event) => props.onAutoInferChange(event.target.checked)} disabled={!props.clientReady} />
          auto.inferAfterStroke
        </label>
        <dl className="mini-stats">
          <div>
            <dt>strokes</dt>
            <dd>{props.strokeCount}</dd>
          </div>
          <div>
            <dt>voice</dt>
            <dd>{props.heckleVoice}</dd>
          </div>
          <div>
            <dt>load</dt>
            <dd>{props.loadState}</dd>
          </div>
        </dl>
        <div className="drawer-actions">
          <button className="terminal-button" type="button" onClick={props.onReset}>reset()</button>
          <button className="terminal-button" type="button" onClick={props.onChangeModel}>model.swap()</button>
        </div>
      </section>

      <TracePanel trace={props.trace} visionState={props.visionState} runtime={props.runtime} />
    </aside>
  );
}

function TracePanel(props: { readonly trace: TraceState | null; readonly visionState: VisionState; readonly runtime: RuntimeObservation | null }) {
  const trace = props.trace;
  const stages: readonly { id: VisionState; label: string }[] = [
    { id: 'capturing', label: 'capture' },
    { id: 'encoding', label: 'encode' },
    { id: 'thinking', label: 'infer' },
  ];
  const runtime = trace?.runtime ?? props.runtime;
  return (
    <section className="dev-section trace-section">
      <div className="dev-section-header">
        <span>model_trace</span>
        <code>{trace ? `#${trace.id}` : 'idle'}</code>
      </div>
      <div className="pipeline">
        {stages.map((stage) => (
          <span key={stage.id} className={stage.id === props.visionState || (trace?.visionState === 'idle' && stage.id === 'thinking') ? 'active' : ''}>
            {stage.label}
          </span>
        ))}
      </div>
      {!trace ? (
        <p className="empty-trace">No run yet. Draw a stroke or click <code>trace.infer()</code> to inspect capture, two-pass inference, raw output, metrics, and heckle parsing.</p>
      ) : (
        <div className="trace-stack">
          <div className="trace-meta">
            <span>{trace.source}</span>
            <span>{trace.preset}</span>
            <span>{trace.startedAt}</span>
            {trace.totalMs ? <span>{(trace.totalMs / 1000).toFixed(2)}s</span> : null}
          </div>
          <p className="trace-status">{trace.status}</p>
          {trace.captureUrl ? (
            <figure className="screenshot-preview">
              <img src={trace.captureUrl} alt="Low-res canvas sent to the vision model" />
              <figcaption>
                {trace.captureWidth ?? '?'}x{trace.captureHeight ?? '?'} / {formatBytes(trace.captureBytes ?? 0)} / crop {trace.cropWidth ?? '?'}x{trace.cropHeight ?? '?'} @ {trace.cropX ?? '?'},{trace.cropY ?? '?'}
              </figcaption>
            </figure>
          ) : null}
          <TimingGrid trace={trace} runtime={runtime} />
          {trace.subject || trace.features ? (
            <TraceBlock
              title="perception"
              content={[
                `subject: ${trace.subject ?? 'unknown'}`,
                `features: ${trace.features?.join(', ') ?? 'none'}`,
                `weird: ${trace.weirdDetail ?? 'unknown'}`,
                `quality: ${trace.lineQuality ?? 'unknown'}`,
                `parse: ${trace.parseStatus ?? 'unknown'}${trace.parseNote ? ` (${trace.parseNote})` : ''}`,
                `heckle_parse: ${trace.heckleParseStatus ?? 'unknown'}${trace.heckleParseNote ? ` (${trace.heckleParseNote})` : ''}`,
              ].join('\n')}
            />
          ) : null}
          {trace.comment ? <TraceBlock title="heckle" content={trace.comment} /> : null}
          {trace.guess ? <TraceBlock title="guess" content={trace.guess} /> : null}
          {trace.errorMessage ? <TraceBlock title="error" content={trace.errorMessage} tone="warning" /> : null}
          {trace.perceptionRawText ? <TraceCode title="perception_raw_output" content={trace.perceptionRawText} /> : null}
          {trace.heckleRawText ? <TraceCode title="heckle_raw_output" content={trace.heckleRawText} /> : null}
          {trace.perceptionPromptPreview ? <TraceCode title="perception_prompt" content={trace.perceptionPromptPreview} /> : null}
          {trace.hecklePromptPreview ? <TraceCode title="heckle_prompt" content={trace.hecklePromptPreview} /> : null}
        </div>
      )}
    </section>
  );
}

function TimingGrid(props: { readonly trace: TraceState; readonly runtime: RuntimeObservation | null }) {
  const rows = [
    ['compose', formatMs(props.trace.composeMs)],
    ['encode', formatMs(props.trace.encodeMs)],
    ['infer', formatMs(props.trace.inferenceMs)],
    ['describe', formatMs(props.trace.perceptionMs)],
    ['heckle', formatMs(props.trace.heckleMs)],
    ['total', formatMs(props.trace.totalMs)],
    ['ttft', formatMs(props.runtime?.ttftMs)],
    ['tok/s', props.runtime?.decodeTokensPerSecond == null ? 'n/a' : props.runtime.decodeTokensPerSecond.toFixed(1)],
    ['input', formatCount(props.runtime?.inputTokens)],
    ['output', formatCount(props.runtime?.outputTokens)],
  ] as const;
  return (
    <dl className="timing-grid">
      {rows.map(([label, value]) => (
        <div key={label}>
          <dt>{label}</dt>
          <dd>{value}</dd>
        </div>
      ))}
    </dl>
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

function TraceCode(props: { readonly title: string; readonly content: string }) {
  return (
    <details className="trace-code">
      <summary>{props.title}</summary>
      <pre>{props.content}</pre>
    </details>
  );
}

async function captureDrawing(userCanvas: HTMLCanvasElement, presetId: CapturePresetId): Promise<CapturedDrawingAsset> {
  const preset = CAPTURE_PRESETS[presetId];
  const composeStarted = performance.now();
  const crop = findInkCrop(userCanvas);
  const width = Math.min(preset.maxWidth, preset.maxHeight);
  const height = width;
  const outputCanvas = document.createElement('canvas');
  outputCanvas.width = width;
  outputCanvas.height = height;
  const context = outputCanvas.getContext('2d');
  if (!context) {
    throw new Error('Could not create capture canvas context.');
  }
  context.imageSmoothingEnabled = true;
  context.imageSmoothingQuality = presetId === 'turbo' ? 'low' : 'medium';
  context.fillStyle = DRAWING_BACKGROUND;
  context.fillRect(0, 0, width, height);
  const cropScale = Math.min(width / crop.width, height / crop.height);
  const drawnWidth = Math.max(1, Math.round(crop.width * cropScale));
  const drawnHeight = Math.max(1, Math.round(crop.height * cropScale));
  context.drawImage(
    userCanvas,
    crop.x,
    crop.y,
    crop.width,
    crop.height,
    Math.round((width - drawnWidth) / 2),
    Math.round((height - drawnHeight) / 2),
    drawnWidth,
    drawnHeight
  );
  const composeMs = Math.round(performance.now() - composeStarted);
  const encodeStarted = performance.now();
  const blob = await canvasToBlob(outputCanvas, 'image/jpeg', preset.quality);
  const encodeMs = Math.round(performance.now() - encodeStarted);
  const bytes = new Uint8Array(await blob.arrayBuffer());
  return {
    bytes,
    url: URL.createObjectURL(blob),
    width,
    height,
    byteLength: bytes.byteLength,
    preset: presetId,
    cropX: crop.x,
    cropY: crop.y,
    cropWidth: crop.width,
    cropHeight: crop.height,
    composeMs,
    encodeMs,
  };
}

function findInkCrop(canvas: HTMLCanvasElement): { readonly x: number; readonly y: number; readonly width: number; readonly height: number } {
  const context = canvas.getContext('2d');
  if (!context) {
    return { x: 0, y: 0, width: canvas.width, height: canvas.height };
  }
  const pixels = context.getImageData(0, 0, canvas.width, canvas.height).data;
  let minX = canvas.width;
  let minY = canvas.height;
  let maxX = -1;
  let maxY = -1;

  for (let y = 0; y < canvas.height; y += 1) {
    for (let x = 0; x < canvas.width; x += 1) {
      const index = (y * canvas.width + x) * 4;
      const distance = Math.abs(pixels[index] - DRAWING_BACKGROUND_RGB.r)
        + Math.abs(pixels[index + 1] - DRAWING_BACKGROUND_RGB.g)
        + Math.abs(pixels[index + 2] - DRAWING_BACKGROUND_RGB.b);
      if (pixels[index + 3] > 0 && distance > 26) {
        minX = Math.min(minX, x);
        minY = Math.min(minY, y);
        maxX = Math.max(maxX, x);
        maxY = Math.max(maxY, y);
      }
    }
  }

  if (maxX < minX || maxY < minY) {
    return { x: 0, y: 0, width: canvas.width, height: canvas.height };
  }

  const inkWidth = maxX - minX + 1;
  const inkHeight = maxY - minY + 1;
  const padding = Math.max(36, Math.round(Math.max(inkWidth, inkHeight) * 0.24));
  const x = Math.max(0, minX - padding);
  const y = Math.max(0, minY - padding);
  const right = Math.min(canvas.width, maxX + padding + 1);
  const bottom = Math.min(canvas.height, maxY + padding + 1);
  return { x, y, width: right - x, height: bottom - y };
}

function getCanvasPoint(canvas: HTMLCanvasElement, clientX: number, clientY: number): DrawingPoint {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((clientX - rect.left) / rect.width) * canvas.width,
    y: ((clientY - rect.top) / rect.height) * canvas.height,
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

function describeVisionState(state: VisionState): string {
  switch (state) {
    case 'capturing':
      return 'compositing canvas';
    case 'encoding':
      return 'encoding tiny JPEG';
    case 'thinking':
      return 'two-pass model heckling';
    case 'idle':
      return 'fast loop ready';
    case 'error':
      return 'vision loop error';
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

function formatMs(value: number | undefined): string {
  return value == null ? 'n/a' : `${Math.round(value)}ms`;
}

function formatCount(value: number | undefined): string {
  return value == null ? 'n/a' : String(value);
}
