import type { GgufSplitRuntime, RemoteAssetMetadata } from './asset-store.js';
import { AssetStore } from './asset-store.js';
import type { BrowserAcquisitionJournal } from './acquisition-journal.js';
import {
  QueryError,
  type AssetRecord,
  type ClassifiedAsset,
  type ModelLoadOptions,
  type RegistryManifest,
} from './types.js';
import type {
  RustRemoteAction,
  RustRemoteEvent,
  RustRemoteFailure,
} from '../wasm/wasm-bridge.js';

export interface RemoteHostResult {
  readonly event: RustRemoteEvent;
  readonly assets?: readonly AssetRecord[];
  readonly classified?: readonly ClassifiedAsset[];
}

type ClassifyAsset = (
  assetId: string,
  file: File,
  signal?: AbortSignal
) => Promise<ClassifiedAsset>;

/** Executes Rust-selected HTTP and OPFS operations without owning acquisition policy. */
export class RemoteAcquisitionHost {
  private readonly downloaded = new Map<string, AssetRecord>();
  private journal: BrowserAcquisitionJournal | null = null;

  public constructor(
    private readonly assetStore: AssetStore,
    private readonly runtime: GgufSplitRuntime,
    private readonly manifest: RegistryManifest,
    private readonly classify: ClassifyAsset,
    private readonly options: ModelLoadOptions
  ) {}

  public async execute(action: RustRemoteAction): Promise<RemoteHostResult> {
    switch (action.kind) {
      case 'fetch_metadata':
        return await this.fetchMetadata(action);
      case 'wait':
        await waitForRetry(action.delayMs, this.options.signal);
        return {
          event: operationEvent(action, 'wait_completed'),
        };
      case 'validate_cache':
        return await this.validateCache(action);
      case 'download':
        return await this.download(action);
      case 'cleanup':
        return await this.cleanup(action);
    }
  }

  private async fetchMetadata(
    action: Extract<RustRemoteAction, { kind: 'fetch_metadata' }>
  ): Promise<RemoteHostResult> {
    let response: Response;
    try {
      response = await fetch(action.url, { method: 'HEAD', signal: this.options.signal });
    } catch (error) {
      if (this.options.signal?.aborted === true) {
        throw error;
      }
      return failed(action, {
        phase: 'metadata',
        kind: 'transport',
        reason: 'request transport failed',
      });
    }
    if (!response.ok) {
      return failed(action, httpFailure('metadata', response));
    }
    return {
      event: {
        ...operationIdentity(action),
        kind: 'metadata_succeeded',
        headers: {
          ...headerInteger(response.headers, 'Content-Length', 'contentLength'),
          ...headerInteger(response.headers, 'X-Linked-Size', 'linkedSize'),
          ...headerText(response.headers, 'ETag', 'etag'),
          ...headerText(response.headers, 'X-Linked-Etag', 'linkedEtag'),
          ...headerText(response.headers, 'Last-Modified', 'lastModified'),
        },
      },
    };
  }

  private async validateCache(
    action: Extract<RustRemoteAction, { kind: 'validate_cache' }>
  ): Promise<RemoteHostResult> {
    const assets: AssetRecord[] = [];
    const classified: ClassifiedAsset[] = [];
    try {
      for (const assetId of action.candidate.assetIds) {
        const record = this.manifest.assets[assetId];
        if (record == null) {
          return failed(action, {
            phase: 'cache_validation',
            kind: 'integrity',
            reason: 'selected cache asset is absent from the registry',
          });
        }
        const file = await this.assetStore.getFile(record);
        assets.push(record);
        classified.push(await this.classify(assetId, file, this.options.signal));
      }
    } catch (error) {
      if (this.options.signal?.aborted === true) {
        throw error;
      }
      return failed(action, hostFailure('cache_validation', error));
    }
    return {
      event: {
        ...operationIdentity(action),
        kind: 'cache_validated',
        assetIds: action.candidate.assetIds,
      },
      assets,
      classified,
    };
  }

  private async download(
    action: Extract<RustRemoteAction, { kind: 'download' }>
  ): Promise<RemoteHostResult> {
    let response: Response;
    try {
      response = await fetch(action.metadata.url, { signal: this.options.signal });
    } catch (error) {
      if (this.options.signal?.aborted === true) {
        throw error;
      }
      return failed(action, {
        phase: 'download',
        kind: 'transport',
        reason: 'request transport failed',
      });
    }
    if (!response.ok) {
      return failed(action, httpFailure('download', response));
    }

    const metadata: RemoteAssetMetadata = {
      url: action.metadata.url,
      canonicalUrl: action.metadata.url,
      name: action.metadata.name,
      bytes: action.metadata.bytes,
      ...(action.metadata.etag == null ? {} : { etag: action.metadata.etag }),
      ...(action.metadata.lastModified == null
        ? {}
        : { lastModified: action.metadata.lastModified }),
    };
    let createdAssetIds: readonly string[] = [];
    try {
      const journal = this.openJournal(action.acquisitionId);
      const receipt =
        action.role === 'model'
          ? await this.assetStore.downloadRemoteGguf(
              metadata,
              this.runtime,
              response,
              this.options.signal,
              this.options.onProgress,
              journal
            )
          : await this.assetStore.downloadRemote(
              metadata,
              action.role,
              response,
              this.options.signal,
              this.options.onProgress,
              journal
            );
      createdAssetIds = receipt.createdAssetIds;
      const classified: ClassifiedAsset[] = [];
      for (const record of receipt.records) {
        this.downloaded.set(record.id, record);
      }
      for (const record of receipt.records) {
        const file = await this.assetStore.getFile(record);
        classified.push(await this.classify(record.id, file, this.options.signal));
      }
      return {
        event: {
          ...operationIdentity(action),
          kind: 'download_succeeded',
          assetIds: receipt.records.map((record) => record.id),
          createdAssetIds: receipt.createdAssetIds,
        },
        assets: receipt.records,
        classified,
      };
    } catch (error) {
      if (this.options.signal?.aborted === true) {
        await this.rollbackCreatedAssets(createdAssetIds);
        throw error;
      }
      return failed(action, hostFailure('download', error), createdAssetIds);
    }
  }

  private async cleanup(
    action: Extract<RustRemoteAction, { kind: 'cleanup' }>
  ): Promise<RemoteHostResult> {
    try {
      for (const assetId of action.assetIds) {
        const record = this.downloaded.get(assetId);
        if (record == null) {
          return failed(action, {
            phase: 'cleanup',
            kind: 'storage',
            reason: 'cleanup asset is absent from the acquisition',
          });
        }
        await this.assetStore.delete(record);
        this.downloaded.delete(assetId);
      }
    } catch (error) {
      return failed(action, hostFailure('cleanup', error));
    }
    return {
      event: operationEvent(action, 'cleanup_succeeded'),
    };
  }

  private async rollbackCreatedAssets(assetIds: readonly string[]): Promise<void> {
    for (const assetId of assetIds) {
      const record = this.downloaded.get(assetId);
      if (record == null) {
        throw new QueryError(
          'STORAGE_CORRUPT',
          'rollback asset is absent from the acquisition'
        );
      }
      await this.assetStore.delete(record);
      this.downloaded.delete(assetId);
    }
  }

  public async commitJournal(): Promise<void> {
    await this.journal?.clear();
    this.journal = null;
  }

  public async cleanupUncommittedJournal(manifest: RegistryManifest): Promise<void> {
    await this.journal?.cleanupUncommitted(manifest);
    this.journal = null;
  }

  private openJournal(acquisitionId: string): BrowserAcquisitionJournal {
    if (this.journal == null) {
      this.journal = this.assetStore.openAcquisitionJournal(acquisitionId);
    }
    return this.journal;
  }
}

function operationIdentity(action: RustRemoteAction): {
  readonly acquisitionId: string;
  readonly memberId: number;
  readonly attempt: number;
} {
  return {
    acquisitionId: action.acquisitionId,
    memberId: action.memberId,
    attempt: action.attempt,
  };
}

function operationEvent(
  action: RustRemoteAction,
  kind: 'wait_completed' | 'cleanup_succeeded'
): Extract<RustRemoteEvent, { kind: typeof kind }> {
  return { ...operationIdentity(action), kind } as Extract<
    RustRemoteEvent,
    { kind: typeof kind }
  >;
}

function failed(
  action: RustRemoteAction,
  failure: RustRemoteFailure,
  createdAssetIds: readonly string[] = []
): RemoteHostResult {
  return {
    event: {
      ...operationIdentity(action),
      kind: 'operation_failed',
      failure,
      createdAssetIds,
    },
  };
}

function httpFailure(
  phase: 'metadata' | 'download',
  response: Response
): RustRemoteFailure {
  const retryAfter = response.headers.get('Retry-After')?.trim();
  return {
    phase,
    kind: 'http',
    status: response.status,
    ...(retryAfter == null || retryAfter.length === 0 ? {} : { retryAfter }),
    reason: `HTTP ${response.status}`,
  };
}

function hostFailure(
  phase: RustRemoteFailure['phase'],
  error: unknown
): RustRemoteFailure {
  if (error instanceof QueryError && error.code === 'STORAGE_CORRUPT') {
    return { phase, kind: 'integrity', reason: error.message };
  }
  return {
    phase,
    kind: 'storage',
    reason: error instanceof Error ? error.message : String(error),
  };
}

function headerText<K extends string>(
  headers: Headers,
  name: string,
  key: K
): Partial<Record<K, string>> {
  const value = headers.get(name)?.trim();
  return value == null || value.length === 0 ? {} : { [key]: value } as Record<K, string>;
}

function headerInteger<K extends string>(
  headers: Headers,
  name: string,
  key: K
): Partial<Record<K, number>> {
  const value = headers.get(name)?.trim();
  if (value == null || !/^\d+$/.test(value)) {
    return {};
  }
  return { [key]: Number(value) } as Record<K, number>;
}

async function waitForRetry(delayMs: number, signal?: AbortSignal): Promise<void> {
  if (signal?.aborted === true) {
    throw signal.reason;
  }
  await new Promise<void>((resolve, reject) => {
    const complete = (): void => {
      signal?.removeEventListener('abort', abort);
      resolve();
    };
    const abort = (): void => {
      clearTimeout(timeout);
      reject(signal?.reason);
    };
    const timeout = setTimeout(complete, delayMs);
    signal?.addEventListener('abort', abort, { once: true });
  });
}
