import type {
  AgentGoal,
  AgentIntent,
  DirectorDecision,
  SimulationGameEvent,
  SimulationAgentState,
  WorldConflict,
  WorldSnapshot,
} from './types.js';

export interface TickStartEvent {
  readonly kind: 'tick-start';
  readonly tick: number;
  readonly timeSeconds: number;
}

export interface TickEndEvent {
  readonly kind: 'tick-end';
  readonly tick: number;
  readonly snapshot: WorldSnapshot;
}

export interface WorldSyncEvent {
  readonly kind: 'world-sync';
  readonly tick: number;
  readonly snapshot: WorldSnapshot;
}

export interface AgentQueryStartEvent {
  readonly kind: 'agent-query-start';
  readonly tick: number;
  readonly agentId: string;
}

export interface AgentQueryEndEvent {
  readonly kind: 'agent-query-end';
  readonly tick: number;
  readonly agentId: string;
  readonly goal: AgentGoal | null;
  readonly intent: AgentIntent | null;
  readonly status: string;
  readonly emotion: string | null;
  readonly queryStatus: 'ok' | 'aborted' | 'timed_out' | 'failed' | 'invalid_request' | 'invalid_response';
  readonly errorMessage?: string;
}

export interface AgentIntentEvent {
  readonly kind: 'agent-intent';
  readonly tick: number;
  readonly agentId: string;
  readonly goal: AgentGoal;
  readonly intent: AgentIntent;
  readonly status: string;
}

export interface AgentExpressionEvent {
  readonly kind: 'agent-expression';
  readonly tick: number;
  readonly agentId: string;
  readonly emotion: string;
}

export interface AgentStateChangeEvent {
  readonly kind: 'agent-state';
  readonly tick: number;
  readonly agent: SimulationAgentState;
}

export interface DirectorConflictEvent {
  readonly kind: 'director-conflict';
  readonly tick: number;
  readonly conflicts: readonly WorldConflict[];
}

export interface DirectorDecisionEvent {
  readonly kind: 'director-decision';
  readonly tick: number;
  readonly decision: DirectorDecision;
}

export interface WorldNoteEvent {
  readonly kind: 'world-note';
  readonly tick: number;
  readonly note: string;
}

export interface DirectorNarrationTraceEvent {
  readonly kind: 'director-narration-trace';
  readonly tick: number;
  readonly rawText: string;
  readonly parsedText: string;
  readonly accepted: boolean;
  readonly reason?: string;
}

export interface GameEventBusEvent {
  readonly kind: 'game-event';
  readonly tick: number;
  readonly event: SimulationGameEvent;
}

export interface AgentTokenEvent {
  readonly kind: 'agent-token';
  readonly tick: number;
  readonly agentId: string;
  readonly queryId: string;
  readonly tokens: string[];
}

export interface RuntimeErrorEvent {
  readonly kind: 'runtime-error';
  readonly tick: number;
  readonly severity: 'critical' | 'warning';
  readonly source: 'agent' | 'referee' | 'narration';
  readonly message: string;
  readonly agentId?: string;
  readonly conflictId?: string;
  readonly taskName?: string;
}

export type SimulationEvent =
  | TickStartEvent
  | TickEndEvent
  | WorldSyncEvent
  | AgentQueryStartEvent
  | AgentQueryEndEvent
  | AgentTokenEvent
  | AgentIntentEvent
  | AgentExpressionEvent
  | AgentStateChangeEvent
  | DirectorConflictEvent
  | DirectorDecisionEvent
  | WorldNoteEvent
  | DirectorNarrationTraceEvent
  | GameEventBusEvent
  | RuntimeErrorEvent;

export type SimulationEventKind = SimulationEvent['kind'];
export type SimulationEventListener<K extends SimulationEventKind = SimulationEventKind> = (
  event: Extract<SimulationEvent, { kind: K }>
) => void;

export class SimulationBus {
  private readonly listenersByKind: Map<SimulationEventKind, Set<SimulationEventListener<any>>> =
    new Map();
  private readonly wildcardListeners: Set<(event: SimulationEvent) => void> = new Set();

  public on<K extends SimulationEventKind>(
    kind: K,
    listener: SimulationEventListener<K>
  ): () => void {
    let bucket = this.listenersByKind.get(kind);
    if (!bucket) {
      bucket = new Set();
      this.listenersByKind.set(kind, bucket);
    }
    bucket.add(listener as SimulationEventListener<any>);
    return () => {
      bucket?.delete(listener as SimulationEventListener<any>);
    };
  }

  public onAny(listener: (event: SimulationEvent) => void): () => void {
    this.wildcardListeners.add(listener);
    return () => {
      this.wildcardListeners.delete(listener);
    };
  }

  public emit(event: SimulationEvent): void {
    const bucket = this.listenersByKind.get(event.kind);
    if (bucket) {
      for (const listener of bucket) {
        try {
          listener(event as never);
        } catch (error) {
          this.logListenerError(error, event.kind);
        }
      }
    }
    for (const listener of this.wildcardListeners) {
      try {
        listener(event);
      } catch (error) {
        this.logListenerError(error, event.kind);
      }
    }
  }

  public clear(): void {
    this.listenersByKind.clear();
    this.wildcardListeners.clear();
  }

  private logListenerError(error: unknown, kind: SimulationEventKind): void {
    const message = error instanceof Error ? error.message : String(error);
    console.error(`[SimulationBus] listener for "${kind}" threw: ${message}`);
  }
}
