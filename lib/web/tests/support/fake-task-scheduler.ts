import type { ScheduledTask, TaskScheduler } from '../../src/utils/task-scheduler.js';

interface PendingDelay {
  readonly dueAt: number;
  readonly task: () => void;
  cancelled: boolean;
}

/**
 * Deterministic {@link TaskScheduler} driven by an explicit virtual clock.
 *
 * Tests advance time in exact milliseconds, so they can assert against the
 * production policy constants without waiting for them.
 */
export class FakeTaskScheduler implements TaskScheduler {
  private nowMs = 0;
  private readonly delays = new Set<PendingDelay>();
  private readonly frames: Array<{ task: () => void; cancelled: boolean }> = [];

  public delay(task: () => void, delayMs: number): ScheduledTask {
    const pending: PendingDelay = { dueAt: this.nowMs + delayMs, task, cancelled: false };
    this.delays.add(pending);
    return {
      cancel: () => {
        pending.cancelled = true;
        this.delays.delete(pending);
      },
    };
  }

  public frame(task: () => void): ScheduledTask {
    const pending = { task, cancelled: false };
    this.frames.push(pending);
    return {
      cancel: () => {
        pending.cancelled = true;
      },
    };
  }

  /** Number of frame callbacks still queued and not cancelled. */
  public get pendingFrameCount(): number {
    return this.frames.filter((frame) => !frame.cancelled).length;
  }

  /** Advances the virtual clock, running every delay that comes due. */
  public advance(deltaMs: number): void {
    this.nowMs += deltaMs;
    for (const pending of [...this.delays]) {
      if (pending.cancelled || pending.dueAt > this.nowMs) {
        continue;
      }
      this.delays.delete(pending);
      pending.task();
    }
  }

  /** Runs every frame callback queued so far. */
  public runFrames(): void {
    const queued = this.frames.splice(0, this.frames.length);
    for (const frame of queued) {
      if (!frame.cancelled) {
        frame.task();
      }
    }
  }
}
