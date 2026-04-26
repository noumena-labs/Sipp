//////////////////////////////////////////////////////////////////////////////
//
// action-bus.ts
//
// - Minimal publish/subscribe surface for character events.
// - Lets imperative consumers (three.js bindings, DOM panels, loggers, etc.)
//   react to actions without having to drive the async iterator themselves.
//
//////////////////////////////////////////////////////////////////////////////

import type { ActionEvent, ProseEvent } from './action-parser.js';
import type { RunStatus } from '../core/run-status.js';

export interface ChatTurnStartEvent {
  readonly kind: 'turn-start';
  readonly userMessage: string;
}

export interface ChatTurnEndEvent {
  readonly kind: 'turn-end';
  readonly finalText: string;
  readonly status: RunStatus;
  readonly errorMessage?: string;
}

export type CharacterEvent =
  | ProseEvent
  | ActionEvent
  | ChatTurnStartEvent
  | ChatTurnEndEvent;

export type CharacterEventKind = CharacterEvent['kind'];
export type CharacterEventListener<K extends CharacterEventKind = CharacterEventKind> = (
  event: Extract<CharacterEvent, { kind: K }>
) => void;

/**
 * Tiny typed pub/sub. Not using `EventTarget` because the `CustomEvent`
 * wrapper adds ceremony for each listener and it does not exist in every
 * runtime we target (e.g. Bun workers historically).
 */
export class CharacterEventBus {
  private readonly listenersByKind: Map<CharacterEventKind, Set<CharacterEventListener<any>>> =
    new Map();
  private readonly wildcardListeners: Set<(event: CharacterEvent) => void> = new Set();

  /**
   * Registers a listener for a specific event kind. Returns a disposer that,
   * when invoked, removes the registration.
   */
  public on<K extends CharacterEventKind>(
    kind: K,
    listener: CharacterEventListener<K>
  ): () => void {
    let bucket = this.listenersByKind.get(kind);
    if (!bucket) {
      bucket = new Set();
      this.listenersByKind.set(kind, bucket);
    }
    bucket.add(listener as CharacterEventListener<any>);
    return () => {
      bucket?.delete(listener as CharacterEventListener<any>);
    };
  }

  /**
   * Registers a listener that receives every event. Useful for tracing.
   */
  public onAny(listener: (event: CharacterEvent) => void): () => void {
    this.wildcardListeners.add(listener);
    return () => {
      this.wildcardListeners.delete(listener);
    };
  }

  /**
   * Dispatches an event to registered listeners. Listener errors are caught
   * and logged so a single faulty subscriber cannot break the turn.
   */
  public emit(event: CharacterEvent): void {
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

  /** Clears every registered listener. Useful when tearing down an agent. */
  public clear(): void {
    this.listenersByKind.clear();
    this.wildcardListeners.clear();
  }

  private logListenerError(error: unknown, kind: CharacterEventKind): void {
    const message = error instanceof Error ? error.message : String(error);
    // eslint-disable-next-line no-console -- surfaced for developer debugging.
    console.error(`[CharacterEventBus] listener for "${kind}" threw: ${message}`);
  }
}
