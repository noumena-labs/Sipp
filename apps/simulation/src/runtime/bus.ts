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
  readonly cancelled: boolean;
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

export interface AgentActionEvent {
  readonly kind: 'agent-action';
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

export interface GameEventBusEvent {
  readonly kind: 'game-event';
  readonly tick: number;
  readonly event: SimulationGameEvent;
}

export type SimulationEvent =
  | TickStartEvent
  | TickEndEvent
  | AgentQueryStartEvent
  | AgentQueryEndEvent
  | AgentIntentEvent
  | AgentActionEvent
  | AgentStateChangeEvent
  | DirectorConflictEvent
  | DirectorDecisionEvent
  | WorldNoteEvent
  | GameEventBusEvent;

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
