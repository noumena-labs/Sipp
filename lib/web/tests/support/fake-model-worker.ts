import type {
  WorkerRequestMessage,
  WorkerResponseMessage,
} from '../../src/worker/model-service-protocol.js';

export class FakeModelWorker {
  public static readonly instances: FakeModelWorker[] = [];

  public onmessage: ((event: MessageEvent<WorkerResponseMessage>) => void) | null = null;
  public onerror: ((event: ErrorEvent) => void) | null = null;
  public onmessageerror: (() => void) | null = null;
  public readonly posted: WorkerRequestMessage[] = [];
  public terminated = false;

  public constructor() {
    FakeModelWorker.instances.push(this);
  }

  public postMessage(message: WorkerRequestMessage): void {
    this.posted.push(message);
  }

  public terminate(): void {
    this.terminated = true;
  }

  public respond(message: WorkerResponseMessage): void {
    this.onmessage?.({ data: message } as MessageEvent<WorkerResponseMessage>);
  }
}

export async function waitForModelWorker(index: number): Promise<FakeModelWorker> {
  for (let attempt = 0; attempt < 20; attempt += 1) {
    const worker = FakeModelWorker.instances[index];
    if (worker != null) {
      return worker;
    }
    await Promise.resolve();
  }
  throw new Error(`Worker ${index} was not created.`);
}

export async function waitForWorkerMessage<K extends WorkerRequestMessage['kind']>(
  worker: FakeModelWorker,
  kind: K
): Promise<Extract<WorkerRequestMessage, { kind: K }>> {
  for (let attempt = 0; attempt < 20; attempt += 1) {
    const message = worker.posted.find((candidate) => candidate.kind === kind);
    if (message != null) {
      return message as Extract<WorkerRequestMessage, { kind: K }>;
    }
    await Promise.resolve();
  }
  throw new Error(`Worker did not receive ${kind}.`);
}

export async function withFakeModelWorker<T>(operation: () => Promise<T>): Promise<T> {
  const descriptor = Object.getOwnPropertyDescriptor(globalThis, 'Worker');
  FakeModelWorker.instances.length = 0;
  Object.defineProperty(globalThis, 'Worker', {
    configurable: true,
    writable: true,
    value: FakeModelWorker,
  });
  try {
    return await operation();
  } finally {
    if (descriptor == null) {
      Reflect.deleteProperty(globalThis, 'Worker');
    } else {
      Object.defineProperty(globalThis, 'Worker', descriptor);
    }
    FakeModelWorker.instances.length = 0;
  }
}
