import type { RequestObservabilityMetrics } from '@noumena-labs/cogent-engine';

const QUERIES_PER_SECOND_WINDOW_MS = 10_000;
const LIVE_UPDATE_INTERVAL_MS = 120;
const ROLLING_LATENCY_SAMPLE_COUNT = 10;

export interface BrainDefinition {
  readonly id: string;
  readonly label: string;
  readonly kind: 'agent' | 'director';
  readonly accentColor: string;
}

export type BrainQueryType = 'decision' | 'referee' | 'narration';
export type BrainQueryStatus =
  | 'idle'
  | 'running'
  | 'completed'
  | 'cancelled'
  | 'timed_out'
  | 'failed';

export interface BeginBrainQueryArgs {
  readonly brainId: string;
  readonly queryType: BrainQueryType;
  readonly queryName?: string | null;
  readonly contextKey: string;
  readonly systemPrompt?: string | null;
  readonly userPrompt?: string | null;
  readonly renderedPrompt: string;
  readonly grammar?: string | null;
}

export interface BrainActivityEntry {
  readonly brainId: string;
  readonly label: string;
  readonly kind: BrainDefinition['kind'];
  readonly accentColor: string;
  readonly queryId: string | null;
  readonly queryType: BrainQueryType | null;
  readonly queryName: string | null;
  readonly contextKey: string | null;
  readonly tick: number | null;
  readonly status: BrainQueryStatus;
  readonly startedAtMs: number | null;
  readonly elapsedMs: number | null;
  readonly systemPrompt: string;
  readonly userPrompt: string;
  readonly renderedPrompt: string;
  readonly responseText: string;
  readonly grammar: string;
  readonly requestId: number | null;
  readonly ttftMs: number | null;
  readonly inputTokenCount: number | null;
  readonly outputTokenCount: number | null;
  readonly errorMessage: string | null;
}

export interface BrainActivityStoreSnapshot {
  readonly brains: readonly BrainActivityEntry[];
  readonly totalQueries: number;
  readonly queriesPerSecond: number;
  readonly activeBrainId: string | null;
  readonly activeBrainLabel: string | null;
  readonly activeQueryCount: number;
  readonly averageLatencyMs: number | null;
  readonly totalFailures: number;
  readonly totalCancelled: number;
}

interface BrainActivityRecord {
  brainId: string;
  label: string;
  kind: BrainDefinition['kind'];
  accentColor: string;
  queryId: string | null;
  queryType: BrainQueryType | null;
  queryName: string | null;
  contextKey: string | null;
  tick: number | null;
  status: BrainQueryStatus;
  startedAtMs: number | null;
  completedAtMs: number | null;
  systemPrompt: string;
  userPrompt: string;
  renderedPrompt: string;
  responseText: string;
  grammar: string;
  requestId: number | null;
  ttftMs: number | null;
  inputTokenCount: number | null;
  outputTokenCount: number | null;
  errorMessage: string | null;
}

export class BrainActivityStore {
  private readonly definitions: readonly BrainDefinition[];
  private readonly definitionsById: Map<string, BrainDefinition>;
  private readonly listeners = new Set<() => void>();
  private readonly recordsByBrainId = new Map<string, BrainActivityRecord>();
  private readonly queryIdsByRequestId = new Map<number, string>();
  private readonly activeQueryIds = new Set<string>();
  private readonly recentQueryStartsMs: number[] = [];
  private readonly recentLatenciesMs: number[] = [];

  private liveUpdateTimer: ReturnType<typeof setInterval> | null = null;
  private cachedSnapshot: BrainActivityStoreSnapshot | null = null;
  private currentTick = 0;
  private nextQueryId = 1;
  private totalQueries = 0;
  private totalFailures = 0;
  private totalCancelled = 0;

  public constructor(definitions: readonly BrainDefinition[]) {
    this.definitions = definitions.slice();
    this.definitionsById = new Map(definitions.map((definition) => [definition.id, definition]));
  }

  public subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => {
      this.listeners.delete(listener);
    };
  };

  public getSnapshot = (): BrainActivityStoreSnapshot => {
    if (this.cachedSnapshot) {
      return this.cachedSnapshot;
    }

    const now = performance.now();
    const recentQueryCount = countRecentQueries(this.recentQueryStartsMs, now);

    const brains = this.definitions.map((definition) =>
      this.toPublicEntry(this.recordsByBrainId.get(definition.id) ?? this.createEmptyRecord(definition), now)
    );
    const activeBrain = brains.find((brain) => brain.status === 'running') ?? null;

    this.cachedSnapshot = {
      brains,
      totalQueries: this.totalQueries,
      queriesPerSecond: recentQueryCount / (QUERIES_PER_SECOND_WINDOW_MS / 1000),
      activeBrainId: activeBrain?.brainId ?? null,
      activeBrainLabel: activeBrain?.label ?? null,
      activeQueryCount: this.activeQueryIds.size,
      averageLatencyMs: average(this.recentLatenciesMs),
      totalFailures: this.totalFailures,
      totalCancelled: this.totalCancelled,
    };

    return this.cachedSnapshot;
  };

  public setCurrentTick = (tick: number): void => {
    this.currentTick = tick;
  };

  public reset = (): void => {
    this.recordsByBrainId.clear();
    this.queryIdsByRequestId.clear();
    this.activeQueryIds.clear();
    this.recentQueryStartsMs.length = 0;
    this.currentTick = 0;
    this.nextQueryId = 1;
    this.totalQueries = 0;
    this.totalFailures = 0;
    this.totalCancelled = 0;
    this.recentLatenciesMs.length = 0;
    this.stopLiveUpdates();
    this.invalidateSnapshot();
    this.emit();
  };

  public beginQuery(args: BeginBrainQueryArgs): string {
    const definition = this.definitionsById.get(args.brainId);
    if (!definition) {
      throw new Error(`Unknown brain ${JSON.stringify(args.brainId)}.`);
    }

    const now = performance.now();
    this.pruneRecentQueries(now);
    this.recentQueryStartsMs.push(now);
    this.totalQueries += 1;

    const queryId = `brain-query-${this.nextQueryId++}`;
    this.recordsByBrainId.set(args.brainId, {
      brainId: definition.id,
      label: definition.label,
      kind: definition.kind,
      accentColor: definition.accentColor,
      queryId,
      queryType: args.queryType,
      queryName: args.queryName ?? null,
      contextKey: args.contextKey,
      tick: this.currentTick,
      status: 'running',
      startedAtMs: now,
      completedAtMs: null,
      systemPrompt: args.systemPrompt?.trim() ?? '',
      userPrompt: args.userPrompt?.trim() ?? '',
      renderedPrompt: args.renderedPrompt,
      responseText: '',
      grammar: args.grammar?.trim() ?? '',
      requestId: null,
      ttftMs: null,
      inputTokenCount: null,
      outputTokenCount: null,
      errorMessage: null,
    });
    this.activeQueryIds.add(queryId);
    this.ensureLiveUpdates();
    this.invalidateSnapshot(now);
    this.emit();
    return queryId;
  }

  public attachRequestId(queryId: string, requestId: number): void {
    const record = this.findRecordByQueryId(queryId);
    if (!record) {
      return;
    }
    record.requestId = requestId;
    this.queryIdsByRequestId.set(requestId, queryId);
    this.invalidateSnapshot();
    this.emit();
  }

  public getQueryIdForRequest(requestId: number): string | null {
    return this.queryIdsByRequestId.get(requestId) ?? null;
  }

  public appendResponse(queryId: string, chunk: string): void {
    if (chunk.length === 0) {
      return;
    }
    const record = this.findRecordByQueryId(queryId);
    if (!record) {
      return;
    }
    record.responseText += chunk;
    this.invalidateSnapshot();
    this.emit();
  }

  public finishQuery(
    queryId: string,
    args: {
      readonly status: Exclude<BrainQueryStatus, 'idle' | 'running'>;
      readonly responseText?: string | null;
      readonly errorMessage?: string | null;
      readonly requestObservability?: RequestObservabilityMetrics | null;
    }
  ): void {
    const record = this.findRecordByQueryId(queryId);
    this.activeQueryIds.delete(queryId);

    if (!record) {
      this.queryIdsByRequestId.forEach((value, requestId) => {
        if (value === queryId) {
          this.queryIdsByRequestId.delete(requestId);
        }
      });
      this.stopLiveUpdatesIfIdle();
      this.emit();
      return;
    }

    const now = performance.now();
    this.adjustStatusCounters(record.status, args.status);
    record.status = args.status;
    record.completedAtMs = now;
    if (args.responseText != null) {
      record.responseText = args.responseText;
    }
    record.errorMessage = args.errorMessage?.trim() || null;
    record.ttftMs = args.requestObservability?.ttftMs ?? null;
    record.inputTokenCount = args.requestObservability?.inputTokenCount ?? null;
    record.outputTokenCount = args.requestObservability?.outputTokenCount ?? null;
    if (record.startedAtMs != null) {
      this.recentLatenciesMs.push(now - record.startedAtMs);
      if (this.recentLatenciesMs.length > ROLLING_LATENCY_SAMPLE_COUNT) {
        this.recentLatenciesMs.shift();
      }
    }
    if (record.requestId != null) {
      this.queryIdsByRequestId.delete(record.requestId);
    }

    this.stopLiveUpdatesIfIdle();
    this.invalidateSnapshot(now);
    this.emit();
  }

  public reviseLatestQuery(
    brainId: string,
    args: {
      readonly status: Exclude<BrainQueryStatus, 'idle' | 'running'>;
      readonly errorMessage?: string | null;
    }
  ): void {
    const record = this.recordsByBrainId.get(brainId);
    if (!record || record.queryId == null) {
      return;
    }

    const nextErrorMessage = args.errorMessage?.trim() || null;
    if (record.status !== args.status) {
      this.adjustStatusCounters(record.status, args.status);
      record.status = args.status;
    }
    if (record.completedAtMs == null) {
      record.completedAtMs = performance.now();
    }
    if (nextErrorMessage) {
      record.errorMessage = nextErrorMessage;
    }
    this.stopLiveUpdatesIfIdle();
    this.invalidateSnapshot(record.completedAtMs ?? performance.now());
    this.emit();
  }

  private findRecordByQueryId(queryId: string): BrainActivityRecord | null {
    for (const record of this.recordsByBrainId.values()) {
      if (record.queryId === queryId) {
        return record;
      }
    }
    return null;
  }

  private createEmptyRecord(definition: BrainDefinition): BrainActivityRecord {
    return {
      brainId: definition.id,
      label: definition.label,
      kind: definition.kind,
      accentColor: definition.accentColor,
      queryId: null,
      queryType: null,
      queryName: null,
      contextKey: null,
      tick: null,
      status: 'idle',
      startedAtMs: null,
      completedAtMs: null,
      systemPrompt: '',
      userPrompt: '',
      renderedPrompt: '',
      responseText: '',
      grammar: '',
      requestId: null,
      ttftMs: null,
      inputTokenCount: null,
      outputTokenCount: null,
      errorMessage: null,
    };
  }

  private toPublicEntry(record: BrainActivityRecord, now: number): BrainActivityEntry {
    let elapsedMs: number | null = null;
    if (record.startedAtMs != null) {
      if (record.status === 'running') {
        elapsedMs = now - record.startedAtMs;
      } else if (record.completedAtMs != null) {
        elapsedMs = record.completedAtMs - record.startedAtMs;
      }
    }

    return {
      brainId: record.brainId,
      label: record.label,
      kind: record.kind,
      accentColor: record.accentColor,
      queryId: record.queryId,
      queryType: record.queryType,
      queryName: record.queryName,
      contextKey: record.contextKey,
      tick: record.tick,
      status: record.status,
      startedAtMs: record.startedAtMs,
      elapsedMs,
      systemPrompt: record.systemPrompt,
      userPrompt: record.userPrompt,
      renderedPrompt: record.renderedPrompt,
      responseText: record.responseText,
      grammar: record.grammar,
      requestId: record.requestId,
      ttftMs: record.ttftMs,
      inputTokenCount: record.inputTokenCount,
      outputTokenCount: record.outputTokenCount,
      errorMessage: record.errorMessage,
    };
  }

  private pruneRecentQueries(now: number): void {
    while (this.recentQueryStartsMs.length > 0 && now - this.recentQueryStartsMs[0]! > QUERIES_PER_SECOND_WINDOW_MS) {
      this.recentQueryStartsMs.shift();
    }
  }

  private ensureLiveUpdates(): void {
    if (this.liveUpdateTimer != null || !this.shouldKeepLiveUpdates()) {
      return;
    }
    this.liveUpdateTimer = globalThis.setInterval(() => {
      const now = performance.now();
      this.invalidateSnapshot(now);
      this.emit();
      if (!this.shouldKeepLiveUpdates()) {
        this.stopLiveUpdates();
      }
    }, LIVE_UPDATE_INTERVAL_MS);
  }

  private stopLiveUpdatesIfIdle(): void {
    this.invalidateSnapshot();
    if (!this.shouldKeepLiveUpdates()) {
      this.stopLiveUpdates();
    }
  }

  private stopLiveUpdates(): void {
    if (this.liveUpdateTimer == null) {
      return;
    }
    clearInterval(this.liveUpdateTimer);
    this.liveUpdateTimer = null;
  }

  private emit(): void {
    for (const listener of this.listeners) {
      listener();
    }
  }

  private invalidateSnapshot(now = performance.now()): void {
    this.pruneRecentQueries(now);
    this.cachedSnapshot = null;
  }

  private shouldKeepLiveUpdates(): boolean {
    return this.activeQueryIds.size > 0 || this.recentQueryStartsMs.length > 0;
  }

  private adjustStatusCounters(previous: BrainQueryStatus, next: BrainQueryStatus): void {
    if (previous === next) {
      return;
    }

    if (countsAsFailure(previous)) {
      this.totalFailures = Math.max(0, this.totalFailures - 1);
    }
    if (previous === 'cancelled') {
      this.totalCancelled = Math.max(0, this.totalCancelled - 1);
    }

    if (countsAsFailure(next)) {
      this.totalFailures += 1;
    }
    if (next === 'cancelled') {
      this.totalCancelled += 1;
    }
  }
}

function countsAsFailure(status: BrainQueryStatus): boolean {
  return status === 'failed' || status === 'timed_out';
}

function countRecentQueries(values: readonly number[], now: number): number {
  let count = 0;
  for (const value of values) {
    if (now - value <= QUERIES_PER_SECOND_WINDOW_MS) {
      count += 1;
    }
  }
  return count;
}

function average(values: readonly number[]): number | null {
  if (values.length === 0) {
    return null;
  }
  let sum = 0;
  for (const value of values) {
    sum += value;
  }
  return sum / values.length;
}
