export interface QueuedRequestPumpStepResult {
  hasActiveRequests: boolean;
  stepResult: number | null;
  settledAny: boolean;
  shouldYieldAfterStep?: boolean;
}

export const DEFAULT_QUEUED_REQUEST_PUMP_SYNC_BURST_LIMIT = 128;
export const DEFAULT_QUEUED_REQUEST_PUMP_IDLE_STREAK_BEFORE_YIELD = 4;

export async function runQueuedRequestPumpLoop(options: {
  isCurrentGeneration: () => boolean;
  runStep: () => Promise<QueuedRequestPumpStepResult>;
  waitForNextSchedulerStep: () => Promise<void>;
  waitingStepResult: number;
  syncBurstLimit?: number;
  idleStreakBeforeYield?: number;
  shouldYieldForResponsiveness?: (burstTickCount: number) => boolean;
}): Promise<void> {
  const syncBurstLimit =
    options.syncBurstLimit ?? DEFAULT_QUEUED_REQUEST_PUMP_SYNC_BURST_LIMIT;
  const idleStreakBeforeYield =
    options.idleStreakBeforeYield ?? DEFAULT_QUEUED_REQUEST_PUMP_IDLE_STREAK_BEFORE_YIELD;
  let burstTickCount = 0;
  let waitingStreak = 0;

  while (options.isCurrentGeneration()) {
    const pumpStep = await options.runStep();
    if (!options.isCurrentGeneration()) {
      return;
    }
    if (!pumpStep.hasActiveRequests) {
      return;
    }

    burstTickCount += 1;
    if (
      pumpStep.stepResult === options.waitingStepResult &&
      !pumpStep.settledAny
    ) {
      waitingStreak += 1;
    } else {
      waitingStreak = 0;
    }

    const shouldYieldForResponsiveness =
      options.shouldYieldForResponsiveness?.(burstTickCount) ?? false;
    if (
      shouldYieldForResponsiveness ||
      pumpStep.shouldYieldAfterStep === true ||
      burstTickCount >= syncBurstLimit ||
      waitingStreak >= idleStreakBeforeYield
    ) {
      burstTickCount = 0;
      waitingStreak = 0;
      await options.waitForNextSchedulerStep();
    }
  }
}
