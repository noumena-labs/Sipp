interface ErrorWithCleanupFailures extends Error {
  readonly cleanupFailures?: AggregateError;
}

export interface CleanupStep {
  readonly label: string;
  readonly release: () => void;
}

export interface AsyncCleanupStep {
  readonly label: string;
  readonly release: () => void | Promise<void>;
}

function labeledCleanupFailure(label: string, error: unknown): Error {
  return new Error(`${label}: ${error instanceof Error ? error.message : String(error)}`, {
    cause: error,
  });
}

function cleanupAggregate(message: string, failures: readonly Error[]): AggregateError | null {
  return failures.length === 0 ? null : new AggregateError(failures, message);
}

/** Attempts every synchronous release step and aggregates any failures. */
export function releaseAll(message: string, steps: readonly CleanupStep[]): void {
  const failures: Error[] = [];
  for (const step of steps) {
    try {
      step.release();
    } catch (error) {
      failures.push(labeledCleanupFailure(step.label, error));
    }
  }
  const aggregate = cleanupAggregate(message, failures);
  if (aggregate != null) {
    throw aggregate;
  }
}

/** Attempts every asynchronous release step and aggregates any failures. */
export async function releaseAllAsync(
  message: string,
  steps: readonly AsyncCleanupStep[]
): Promise<void> {
  const failures: Error[] = [];
  for (const step of steps) {
    try {
      await step.release();
    } catch (error) {
      failures.push(labeledCleanupFailure(step.label, error));
    }
  }
  const aggregate = cleanupAggregate(message, failures);
  if (aggregate != null) {
    throw aggregate;
  }
}

/** Attaches cleanup failures without replacing the operation's primary error. */
export function attachCleanupFailures(primary: unknown, cleanup: unknown): unknown {
  const primaryError: ErrorWithCleanupFailures = primary instanceof Error
    ? primary
    : new Error(String(primary), { cause: primary });
  const failures = cleanup instanceof AggregateError
    ? cleanup.errors
    : [cleanup];
  const priorFailures = primaryError.cleanupFailures?.errors ?? [];
  Object.defineProperty(primaryError, 'cleanupFailures', {
    configurable: true,
    enumerable: false,
    value: new AggregateError(
      [...priorFailures, ...failures],
      'Cleanup failed after the primary operation error.'
    ),
  });
  return primaryError;
}
