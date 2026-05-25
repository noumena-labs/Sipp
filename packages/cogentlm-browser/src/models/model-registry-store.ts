import { FileSystemStorage } from '../storage/file-system-storage.js';
import {
  QueryError,
  type RegistryManifest,
} from './types.js';

const REGISTRY_FILE_NAME = 'registry.json';

function emptyManifest(): RegistryManifest {
  return {
    version: 3,
    projectorIndexRevision: 0,
    assets: {},
    models: {},
  };
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value != null && !Array.isArray(value);
}

function parseManifest(text: string): RegistryManifest {
  const parsed = JSON.parse(text) as unknown;
  if (!isObject(parsed) || parsed.version !== 3) {
    throw new QueryError('STORAGE_CORRUPT', 'Model registry must be manifest version 3.');
  }
  if (!isObject(parsed.assets) || !isObject(parsed.models)) {
    throw new QueryError('STORAGE_CORRUPT', 'Model registry is missing assets or models.');
  }
  if (
    typeof parsed.projectorIndexRevision !== 'number' ||
    !Number.isInteger(parsed.projectorIndexRevision) ||
    parsed.projectorIndexRevision < 0
  ) {
    throw new QueryError(
      'STORAGE_CORRUPT',
      'Model registry is missing a valid projector index revision.'
    );
  }
  return {
    version: 3,
    projectorIndexRevision: parsed.projectorIndexRevision,
    assets: parsed.assets as RegistryManifest['assets'],
    models: parsed.models as RegistryManifest['models'],
  };
}

export class ModelRegistryStore {
  private manifest: RegistryManifest | null = null;
  private initPromise: Promise<void> | null = null;
  private operationChain: Promise<void> = Promise.resolve();

  constructor(private readonly storage = new FileSystemStorage()) {}

  public async read(): Promise<RegistryManifest> {
    await this.ensureInitialized();
    return this.clone(this.getManifest());
  }

  public async write(
    update: (manifest: RegistryManifest) => void | Promise<void>
  ): Promise<RegistryManifest> {
    return this.withLock(async () => {
      await this.ensureInitialized();
      const manifest = this.clone(this.getManifest());
      await update(manifest);
      await this.writeManifest(manifest);
      this.manifest = manifest;
      return this.clone(manifest);
    });
  }

  private async ensureInitialized(): Promise<void> {
    if (this.manifest != null) {
      return;
    }
    this.ensureAvailable();
    if (this.initPromise == null) {
      this.initPromise = (async () => {
        const text = await this.storage.readText(REGISTRY_FILE_NAME);
        if (text == null) {
          const manifest = emptyManifest();
          await this.writeManifest(manifest);
          this.manifest = manifest;
        } else {
          this.manifest = parseManifest(text);
        }
      })().finally(() => {
        this.initPromise = null;
      });
    }
    await this.initPromise;
  }

  private ensureAvailable(): void {
    if (!FileSystemStorage.isSupported()) {
      throw new QueryError(
        'STORAGE_UNAVAILABLE',
        'Managed model storage requires OPFS, but navigator.storage.getDirectory() is unavailable.'
      );
    }
  }

  private getManifest(): RegistryManifest {
    if (this.manifest == null) {
      throw new QueryError('STORAGE_CORRUPT', 'Model registry has not been initialized.');
    }
    return this.manifest;
  }

  private async writeManifest(manifest: RegistryManifest = this.getManifest()): Promise<void> {
    await this.storage.writeText(REGISTRY_FILE_NAME, JSON.stringify(manifest, null, 2));
  }

  private clone(manifest: RegistryManifest): RegistryManifest {
    return JSON.parse(JSON.stringify(manifest)) as RegistryManifest;
  }

  private async withLock<T>(operation: () => Promise<T>): Promise<T> {
    const previous = this.operationChain;
    let release!: () => void;
    this.operationChain = new Promise<void>((resolve) => {
      release = resolve;
    });
    await previous;
    try {
      return await operation();
    } finally {
      release();
    }
  }
}
