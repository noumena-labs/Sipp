import type {
  GgufSplitRuntime,
  RemoteAssetMetadata,
  RemoteDownloadPlan,
} from './asset-store.js';
import { AssetStore } from './asset-store.js';
import type { BrowserAcquisitionJournal } from './acquisition-journal.js';
import {
  QueryError,
  type AssetRecord,
  type ClassifiedAsset,
  type FallbackEvent,
  type ModelAddOptions,
  type RegistryManifest,
} from './types.js';
import type {
  RustRemoteAction,
  RustRemoteEvent,
  RustRemoteFailure,
} from '../wasm/wasm-bridge.js';
import { StreamStallError } from '../engine/file-system-storage.js';

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

interface RemoteAcquisitionOptions extends ModelAddOptions {
  readonly onWarning?: (event: FallbackEvent) => void;
}

class DownloadTransportError extends Error {
  public constructor(cause: unknown) {
    super(cause instanceof Error ? cause.message : 'request transport failed', { cause });
    this.name = 'DownloadTransportError';
  }
}

/** Executes Rust-selected HTTP and OPFS operations without owning acquisition policy. */
export class RemoteAcquisitionHost {
  private readonly downloaded = new Map<string, AssetRecord>();
  private journal: BrowserAcquisitionJournal | null = null;

  public constructor(
    private readonly assetStore: AssetStore,
    private readonly runtime: GgufSplitRuntime,
    private readonly manifest: RegistryManifest,
    private readonly classify: ClassifyAsset,
    private readonly options: RemoteAcquisitionOptions
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

    let plan: RemoteDownloadPlan;
    let response: Response | null = null;
    try {
      plan = await this.assetStore.prepareRemoteDownload(metadata, this.runtime);
      if (plan.startOffset < metadata.bytes) {
        const headers = downloadRangeHeaders(action, plan.startOffset);
        response = await fetchDownloadResponse(action.metadata.url, {
          headers: Object.keys(headers).length > 0 ? headers : undefined,
          signal: this.options.signal,
        });
        const invalidRangeResponse =
          plan.startOffset > 0 &&
          (response.status === 416 ||
            (response.ok && !responseRangeStartsAt(response, plan.startOffset)));
        if (invalidRangeResponse) {
          this.warnTransfer(
            action.metadata.name,
            response.status === 416
              ? 'range request was not satisfiable'
              : 'server did not honor the requested byte range'
          );
          // Release the ignored response before retrying without a range.
          try {
            await response.body?.cancel();
          } catch {}
          plan = await this.assetStore.resetRemoteDownload(plan);
          response = await fetchDownloadResponse(action.metadata.url, {
            signal: this.options.signal,
          });
        }
      }
    } catch (error) {
      if (this.options.signal?.aborted === true) {
        throw error;
      }
      return failed(
        action,
        error instanceof DownloadTransportError
          ? {
              phase: 'download',
              kind: 'transport',
              reason: error.message,
            }
          : hostFailure('download', error)
      );
    }

    if (response != null && !response.ok) {
      return failed(action, httpFailure('download', response));
    }

    if (response != null && response.body == null) {
      return failed(action, {
        phase: 'download',
        kind: 'transport',
        reason: 'response body is null',
      });
    }

    let createdAssetIds: readonly string[] = [];
    try {
      const journal = this.openJournal(action.acquisitionId);
      const receipt = await this.assetStore.downloadRemoteGguf(
        metadata,
        this.runtime,
        {
          plan,
          body: response?.body ?? null,
          signal: this.options.signal,
          onProgress: this.options.onProgress,
          journal,
          stallTimeoutMs: this.options.stallTimeoutMs ?? 30_000,
        }
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

  private warnTransfer(assetName: string, reason: string): void {
    this.options.onWarning?.({
      type: 'fallback-warning',
      kind: 'transfer',
      detail: `Resumable download fallback for "${assetName}": ${reason}.`,
      fallbackTo: 'full-download',
    });
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

function responseRangeStartsAt(response: Response, startOffset: number): boolean {
  if (response.status !== 206) {
    return false;
  }
  const contentRange = response.headers.get('Content-Range')?.trim();
  const match = contentRange?.match(/^bytes (\d+)-\d+\/\d+$/);
  return match != null && Number(match[1]) === startOffset;
}

function downloadRangeHeaders(
  action: Extract<RustRemoteAction, { kind: 'download' }>,
  startOffset: number
): Record<string, string> {
  if (startOffset === 0 || startOffset >= action.metadata.bytes) {
    return {};
  }
  return {
    Range: `bytes=${startOffset}-`,
    ...(action.metadata.etag != null
      ? { 'If-Range': action.metadata.etag }
      : action.metadata.lastModified != null
        ? { 'If-Range': action.metadata.lastModified }
        : {}),
  };
}

async function fetchDownloadResponse(url: string, init: RequestInit): Promise<Response> {
  try {
    return await fetch(url, init);
  } catch (cause) {
    throw new DownloadTransportError(cause);
  }
}

function hostFailure(
  phase: RustRemoteFailure['phase'],
  error: unknown
): RustRemoteFailure {
  if (error instanceof QueryError && error.code === 'STORAGE_CORRUPT') {
    return { phase, kind: 'integrity', reason: error.message };
  }
  if (phase === 'download' && error instanceof StreamStallError) {
    return {
      phase,
      kind: 'transport',
      reason: error.message,
    };
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
