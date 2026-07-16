import { FileSystemStorage } from '../engine/file-system-storage.js';
import {
  QueryError,
  type RegistryManifest,
} from './types.js';

const JOURNAL_VERSION = 1;
const JOURNAL_DIRECTORY = ['.incoming', 'journals'] as const;

interface AcquisitionJournalEntry {
  readonly storagePath: string;
}

interface AcquisitionJournalFile {
  readonly version: 1;
  readonly acquisitionId: string;
  readonly entries: readonly AcquisitionJournalEntry[];
}

function journalPath(acquisitionId: string): readonly string[] {
  return [...JOURNAL_DIRECTORY, `${acquisitionId}.json`];
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

async function cleanupJournalEntries(
  storage: FileSystemStorage,
  journal: AcquisitionJournalFile,
  manifest: RegistryManifest,
  journalFilePath: readonly string[]
): Promise<void> {
  const protectedPaths = new Set(
    Object.values(manifest.assets).map((asset) => asset.storagePath)
  );
  for (const entry of journal.entries) {
    if (!protectedPaths.has(entry.storagePath)) {
      await storage.deleteFile(entry.storagePath);
    }
  }
  await storage.deleteFileAt(journalFilePath);
}

export async function recoverBrowserAcquisitionJournals(
  storage: FileSystemStorage,
  manifest: RegistryManifest
): Promise<void> {
  const fileNames = await storage.listFileNamesAt(JOURNAL_DIRECTORY);
  for (const fileName of fileNames) {
    const path = [...JOURNAL_DIRECTORY, fileName];
    const text = await storage.readTextAt(path);
    if (text == null) {
      continue;
    }
    await cleanupJournalEntries(storage, parseJournal(text, fileName), manifest, path);
  }
}

export class BrowserAcquisitionJournal {
  private readonly entries = new Set<string>();

  public constructor(
    private readonly storage: FileSystemStorage,
    private readonly acquisitionId: string
  ) {}

  public async recordStoragePath(storagePath: string): Promise<void> {
    await this.recordStoragePaths([storagePath]);
  }

  public async recordStoragePaths(storagePaths: readonly string[]): Promise<void> {
    for (const storagePath of storagePaths) {
      this.entries.add(validateStoragePath(storagePath));
    }
    await this.write();
  }

  public async cleanupUncommitted(manifest: RegistryManifest): Promise<void> {
    await cleanupJournalEntries(
      this.storage,
      this.file(),
      manifest,
      journalPath(this.acquisitionId)
    );
  }

  public async clear(): Promise<void> {
    await this.storage.deleteFileAt(journalPath(this.acquisitionId));
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
