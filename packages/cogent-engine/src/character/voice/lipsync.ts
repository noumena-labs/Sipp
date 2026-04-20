//////////////////////////////////////////////////////////////////////////////
//
// voice/lipsync.ts
//
// - Renderer-agnostic lipsync driver. Produces a normalised "mouth openness"
//   signal in [0, 1] that renderer bindings (three-vrm, DOM, etc.) can
//   subscribe to and translate into blendshape weights.
//
// - v1 is intentionally simple: when speech is active, the driver emits a
//   band-limited pseudo-phoneme signal based on a running sine + noise
//   model. For Web Speech TTS we can't get phonemes, but this is enough to
//   give the avatar plausible mouth motion that starts/stops with speech.
//
// - A future v2 can accept real phoneme timings (from e.g. Piper/espeak-ng
//   with Web Audio routing) and emit viseme weights directly.
//
//////////////////////////////////////////////////////////////////////////////

export interface LipsyncDriverOptions {
  /** Target sample rate of the openness signal, in Hz. Defaults to 30. */
  readonly sampleRateHz?: number;
  /** Base oscillation frequency (Hz). Defaults to 5 (≈ syllable rate). */
  readonly oscillationHz?: number;
}

export interface LipsyncDriver {
  /** Start generating openness samples. Idempotent. */
  start(): void;
  /** Stop generating samples. Subscribers receive a final 0 to close the mouth. */
  stop(): void;
  /** Whether `start()` is currently active. */
  readonly isActive: boolean;
  /** Subscribe to openness samples in [0, 1]. Returns a disposer. */
  onOpenness(listener: (openness: number) => void): () => void;
  /** Cleanup all timers and listeners. */
  dispose(): void;
}

export function createLipsyncDriver(options: LipsyncDriverOptions = {}): LipsyncDriver {
  const sampleRate = Math.max(5, options.sampleRateHz ?? 30);
  const oscillation = Math.max(0.1, options.oscillationHz ?? 5);
  const periodMs = 1000 / sampleRate;
  const listeners = new Set<(openness: number) => void>();
  let timer: ReturnType<typeof setInterval> | null = null;
  let startedAt = 0;

  const emit = (value: number): void => {
    for (const listener of listeners) {
      try {
        listener(value);
      } catch (error) {
        console.error('[lipsync] listener threw:', error);
      }
    }
  };

  return {
    get isActive() {
      return timer !== null;
    },
    start() {
      if (timer !== null) {
        return;
      }
      startedAt = performance.now();
      timer = setInterval(() => {
        const elapsedSeconds = (performance.now() - startedAt) / 1000;
        // Base oscillation (syllabic rhythm).
        const base = 0.5 + 0.45 * Math.sin(elapsedSeconds * oscillation * Math.PI * 2);
        // Small amount of higher-frequency noise so the motion doesn't feel
        // perfectly periodic.
        const jitter = 0.1 * Math.sin(elapsedSeconds * oscillation * Math.PI * 7.3);
        const openness = Math.max(0, Math.min(1, base + jitter));
        emit(openness);
      }, periodMs);
    },
    stop() {
      if (timer !== null) {
        clearInterval(timer);
        timer = null;
      }
      // Emit a trailing zero so subscribers visibly close the mouth.
      emit(0);
    },
    onOpenness(listener) {
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
    dispose() {
      if (timer !== null) {
        clearInterval(timer);
        timer = null;
      }
      listeners.clear();
    },
  };
}
