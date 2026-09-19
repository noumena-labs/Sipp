import { FileSystemStorage } from '../engine/file-system-storage.js';
import {
  QueryError,
  type RegistryManifest,
} from './types.js';

const JOURNAL_VERSION = 1;
const PARTIAL_DOWNLOAD_VERSION = 1;
const JOURNAL_DIRECTORY = ['.incoming', 'journals'] as const;
const PARTIAL_DOWNLOAD_DIRECTORY = ['.incoming', 'partials'] as const;
const PARTIAL_DOWNLOAD_MAX_AGE_MS = 7 * 24 * 60 * 60 * 1000;
const REMOTE_DOWNLOAD_PATH = /^(?:asset-([0-9a-f]{64})|tmp-source-([0-9a-f]{24}))-/;
const REMOTE_DOWNLOAD_PREFIXES = ['asset-', 'tmp-source-'] as const;

interface AcquisitionJournalEntry {
  readonly storagePath: string;
}

interface AcquisitionJournalFile {
  readonly version: 1;
  readonly acquisitionId: string;
  readonly entries: readonly AcquisitionJournalEntry[];
}

interface PartialDownloadFile {
  readonly version: 1;
  readonly storagePath: string;
  readonly expectedBytes: number;
  readonly updatedAt: string;
}

function journalPath(acquisitionId: string): readonly string[] {
  return [...JOURNAL_DIRECTORY, `${acquisitionId}.json`];
}

function partialDownloadPath(storagePath: string): readonly string[] {
  const match = REMOTE_DOWNLOAD_PATH.exec(storagePath);
  const sourceKey = match?.[1] ?? match?.[2];
  if (sourceKey == null) {
    throw new QueryError(
      'STORAGE_CORRUPT',
      `Partial download has an invalid storage path: ${storagePath}.`
    );
  }
  return [...PARTIAL_DOWNLOAD_DIRECTORY, `${sourceKey}.json`];
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value != null && !Array.isArray(value);
}

function isValidStoragePath(value: string): boolean {
  return (
    value.length > 0 &&
    value !== '.' &&
    value !== '..' &&
    !value.includes('/') &&
    !value.includes('\\')
  );
}

function validateStoragePath(storagePath: string): string {
  if (!isValidStoragePath(storagePath)) {
    throw new QueryError(
      'STORAGE_CORRUPT',
      `Acquisition journal contains an invalid storage path: ${storagePath}.`
    );
  }
  return storagePath;
}

function parseJournal(text: string, fileName: string): AcquisitionJournalFile {
  const parsed = JSON.parse(text) as unknown;
  if (
    !isObject(parsed) ||
    parsed.version !== JOURNAL_VERSION ||
    typeof parsed.acquisitionId !== 'string' ||
    !Array.isArray(parsed.entries)
  ) {
    throw new QueryError('STORAGE_CORRUPT', `Invalid acquisition journal: ${fileName}.`);
  }
  return {
    version: JOURNAL_VERSION,
    acquisitionId: parsed.acquisitionId,
    entries: parsed.entries.map((entry) => {
      if (!isObject(entry) || typeof entry.storagePath !== 'string') {
        throw new QueryError('STORAGE_CORRUPT', `Invalid acquisition journal: ${fileName}.`);
      }
      return { storagePath: validateStoragePath(entry.storagePath) };
    }),
  };
}

function parsePartialDownload(text: string, fileName: string): PartialDownloadFile {
  const parsed = JSON.parse(text) as unknown;
  if (
    !isObject(parsed) ||
    parsed.version !== PARTIAL_DOWNLOAD_VERSION ||
    typeof parsed.storagePath !== 'string' ||
    !Number.isSafeInteger(parsed.expectedBytes) ||
    (parsed.expectedBytes as number) <= 0 ||
    typeof parsed.updatedAt !== 'string'
  ) {
    throw new QueryError('STORAGE_CORRUPT', `Invalid partial download record: ${fileName}.`);
  }
  const storagePath = validateStoragePath(parsed.storagePath);
  const expectedFileName = partialDownloadPath(storagePath).at(-1);
  const updatedAtMs = Date.parse(parsed.updatedAt);
  if (expectedFileName !== fileName || !Number.isFinite(updatedAtMs)) {
    throw new QueryError('STORAGE_CORRUPT', `Invalid partial download record: ${fileName}.`);
  }
  return {
    version: PARTIAL_DOWNLOAD_VERSION,
    storagePath,
    expectedBytes: parsed.expectedBytes as number,
    updatedAt: parsed.updatedAt,
  };
}

function protectedStoragePaths(manifest: RegistryManifest): Set<string> {
  return new Set(Object.values(manifest.assets).map((asset) => asset.storagePath));
}

async function recoverPartialDownloads(
  storage: FileSystemStorage,
  protectedPaths: ReadonlySet<string>
): Promise<Set<string>> {
  const retainedPaths = new Set<string>();
  const fileNames = await storage.listFileNamesAt(PARTIAL_DOWNLOAD_DIRECTORY);
  const now = Date.now();
  const oldestValidTimestamp = now - PARTIAL_DOWNLOAD_MAX_AGE_MS;
  for (const fileName of fileNames) {
    const path = [...PARTIAL_DOWNLOAD_DIRECTORY, fileName];
    const text = await storage.readTextAt(path);
    if (text == null) {
      continue;
    }
    let record: PartialDownloadFile;
    try {
      record = parsePartialDownload(text, fileName);
    } catch {
      await storage.deleteFileAt(path);
      continue;
    }
    if (protectedPaths.has(record.storagePath)) {
      await storage.deleteFileAt(path);
      continue;
    }
    const file = await storage.getFile(record.storagePath);
    const updatedAtMs = Date.parse(record.updatedAt);
    const isValid =
      file != null &&
      file.size > 0 &&
      file.size <= record.expectedBytes &&
      updatedAtMs >= oldestValidTimestamp &&
      updatedAtMs <= now;
    if (isValid) {
      retainedPaths.add(record.storagePath);
    } else {
      await storage.deleteFile(record.storagePath);
      await storage.deleteFileAt(path);
    }
  }
  return retainedPaths;
}

async function cleanupJournalEntries(
  storage: FileSystemStorage,
  journal: AcquisitionJournalFile,
  protectedPaths: ReadonlySet<string>,
  retainedPaths: ReadonlySet<string>,
  journalFilePath: readonly string[]
): Promise<void> {
  for (const entry of journal.entries) {
    if (!protectedPaths.has(entry.storagePath) && !retainedPaths.has(entry.storagePath)) {
      await storage.deleteFile(entry.storagePath);
    }
  }
  await storage.deleteFileAt(journalFilePath);
}

async function cleanupUntrackedRemoteDownloads(
  storage: FileSystemStorage,
  protectedPaths: ReadonlySet<string>,
  retainedPaths: ReadonlySet<string>
): Promise<void> {
  for (const fileName of await storage.listFileNames()) {
    if (
      REMOTE_DOWNLOAD_PREFIXES.some((prefix) => fileName.startsWith(prefix)) &&
      !protectedPaths.has(fileName) &&
      !retainedPaths.has(fileName)
    ) {
      await storage.deleteFile(fileName);
    }
  }
}

/** Recovers resumable downloads and rolls back incomplete browser acquisitions. */
export async function recoverBrowserAcquisitionState(
  storage: FileSystemStorage,
  manifest: RegistryManifest
): Promise<void> {
  const protectedPaths = protectedStoragePaths(manifest);
  const retainedPaths = await recoverPartialDownloads(storage, protectedPaths);
  const fileNames = await storage.listFileNamesAt(JOURNAL_DIRECTORY);
  for (const fileName of fileNames) {
    const path = [...JOURNAL_DIRECTORY, fileName];
    const text = await storage.readTextAt(path);
    if (text == null) {
      continue;
    }
    await cleanupJournalEntries(
      storage,
      parseJournal(text, fileName),
      protectedPaths,
      retainedPaths,
      path
    );
  }
  await cleanupUntrackedRemoteDownloads(storage, protectedPaths, retainedPaths);
}

/** Persists rollback ownership separately from reload-safe remote download state. */
export class BrowserAcquisitionJournal {
  private readonly entries = new Set<string>();
  private readonly partialDownloadPaths = new Set<string>();

  public constructor(
    private readonly storage: FileSystemStorage,
    private readonly acquisitionId: string
  ) {}

  /** Marks a deterministic remote source as safe to retain across reloads. */
  public async recordResumableDownload(
    storagePath: string,
    expectedBytes: number
  ): Promise<void> {
    if (!Number.isSafeInteger(expectedBytes) || expectedBytes <= 0) {
      throw new QueryError(
        'STORAGE_CORRUPT',
        `Partial download has an invalid expected size: ${expectedBytes}.`
      );
    }
    const validatedPath = validateStoragePath(storagePath);
    const path = partialDownloadPath(validatedPath);
    await this.storage.writeTextAt(
      path,
      JSON.stringify({
        version: PARTIAL_DOWNLOAD_VERSION,
        storagePath: validatedPath,
        expectedBytes,
        updatedAt: new Date().toISOString(),
      } satisfies PartialDownloadFile, null, 2)
    );
    this.partialDownloadPaths.add(path.join('/'));
  }

  /** Records generated artifacts that must be removed unless the acquisition commits. */
  public async recordTemporaryPaths(storagePaths: readonly string[]): Promise<void> {
    for (const storagePath of storagePaths) {
      this.entries.add(validateStoragePath(storagePath));
    }
    await this.write();
  }

  /** Rolls back generated artifacts while retaining resumable source bytes. */
  public async cleanupUncommitted(manifest: RegistryManifest): Promise<void> {
    await cleanupJournalEntries(
      this.storage,
      this.file(),
      protectedStoragePaths(manifest),
      new Set(),
      journalPath(this.acquisitionId)
    );
  }

  /** Removes recovery metadata after the manifest commit succeeds. */
  public async clear(): Promise<void> {
    await this.storage.deleteFileAt(journalPath(this.acquisitionId));
    for (const path of this.partialDownloadPaths) {
      await this.storage.deleteFileAt(path.split('/'));
    }
  }

  private async write(): Promise<void> {
    await this.storage.writeTextAt(
      journalPath(this.acquisitionId),
      JSON.stringify(this.file(), null, 2)
    );
  }

  private file(): AcquisitionJournalFile {
    return {
      version: JOURNAL_VERSION,
      acquisitionId: this.acquisitionId,
      entries: [...this.entries].sort().map((storagePath) => ({ storagePath })),
    };
  }
}
