import assert from 'node:assert/strict';
import test from 'node:test';

import { runQueuedRequestPumpLoop } from './queued-request-pump.js';

test('runQueuedRequestPumpLoop yields immediately when a step requests post-step yielding', async () => {
  let waitCount = 0;
  let stepCount = 0;

  await runQueuedRequestPumpLoop({
    isCurrentGeneration: () => stepCount < 2,
    waitingStepResult: 0,
    runStep: async () => {
      stepCount += 1;
      if (stepCount === 1) {
        return {
          hasActiveRequests: true,
          stepResult: 1,
          settledAny: false,
          shouldYieldAfterStep: true,
        };
      }

      return {
        hasActiveRequests: false,
        stepResult: null,
        settledAny: false,
      };
    },
    waitForNextSchedulerStep: async () => {
      waitCount += 1;
    },
  });

  assert.equal(waitCount, 1);
  assert.equal(stepCount, 2);
});
