/** A scheduled callback that has not run yet. */
export interface ScheduledTask {
  /** Prevents the task from running. Safe to call after it has already run. */
  cancel(): void;
}

/**
 * Host timing used by long-lived browser components.
 *
 * Injecting the scheduler—rather than the delays themselves—lets tests assert
 * the real policy constants without waiting for them.
 */
export interface TaskScheduler {
  /** Runs `task` after `delayMs` milliseconds. */
  delay(task: () => void, delayMs: number): ScheduledTask;
  /** Runs `task` on the next paint, or on a timer where frames are absent. */
  frame(task: () => void): ScheduledTask;
}

/** Frame cadence used when the host has no `requestAnimationFrame`. */
const FRAME_FALLBACK_MS = 16;

export const hostTaskScheduler: TaskScheduler = {
  delay(task, delayMs) {
    const handle = setTimeout(task, delayMs);
    return {
      cancel: () => {
        clearTimeout(handle);
      },
    };
  },
  frame(task) {
    if (typeof requestAnimationFrame !== 'function') {
      const handle = setTimeout(task, FRAME_FALLBACK_MS);
      return {
        cancel: () => {
          clearTimeout(handle);
        },
      };
    }
    const handle = requestAnimationFrame(() => task());
    return {
      cancel: () => {
        cancelAnimationFrame(handle);
      },
    };
  },
};
