import { describe, expect, test } from 'bun:test';

import { QueryError } from '../models/types.js';
import { unwrapLifecycleResponse } from './lifecycle-bridge.js';

describe('unwrapLifecycleResponse', () => {
  test('preserves unsupported operation errors', () => {
    let thrown: unknown;
    try {
      unwrapLifecycleResponse(
        {
          ok: false,
          error: {
            code: 'UNSUPPORTED_OPERATION',
            message: 'unsupported operation chat: model has no chat template',
          },
        },
        'chat'
      );
    } catch (error) {
      thrown = error;
    }

    expect(thrown).toBeInstanceOf(QueryError);
    expect((thrown as QueryError).code).toBe('UNSUPPORTED_OPERATION');
  });
});
