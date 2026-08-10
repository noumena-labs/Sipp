/**
 * Runs operations one at a time, in submission order.
 *
 * Submission order is fixed synchronously by the `run(...)` call, so callers
 * that queue work before awaiting still execute in the order they queued. A
 * rejected operation settles only its own caller: the queue stays usable and
 * the operations behind it still run.
 */
export class AsyncSerialQueue {
  private tail: Promise<void> = Promise.resolve();

  /** Queues `operation` behind every operation submitted so far. */
  public run<T>(operation: () => T | Promise<T>): Promise<T> {
    const result = this.tail.then(operation);
    this.tail = result.then(
      () => undefined,
      () => undefined
    );
    return result;
  }

  /** Resolves once every operation submitted so far has settled. */
  public idle(): Promise<void> {
    return this.tail;
  }
}
