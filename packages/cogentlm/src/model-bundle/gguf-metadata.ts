const GGUF_MAGIC = 0x46554747;
const GGUF_SUPPORTED_VERSIONS = new Set([2, 3]);
const DEFAULT_INITIAL_PREFIX_BYTES = 64 * 1024;
const DEFAULT_MAX_PREFIX_BYTES = 8 * 1024 * 1024;
const DEFAULT_MAX_KV_ENTRIES = 256;

const EARLY_STOP_KEYS = new Set([
  'tokenizer.ggml.tokens',
  'tokenizer.ggml.scores',
  'tokenizer.ggml.merges',
  'tokenizer.huggingface.json',
]);

const TARGET_KEYS = new Set([
  'general.type',
  'general.architecture',
  'clip.projector_type',
  'clip.vision.projector_type',
  'clip.has_vision_encoder',
]);

const textDecoder = new TextDecoder();

enum GgufValueType {
  UINT8 = 0,
  INT8 = 1,
  UINT16 = 2,
  INT16 = 3,
  UINT32 = 4,
  INT32 = 5,
  FLOAT32 = 6,
  BOOL = 7,
  STRING = 8,
  ARRAY = 9,
  UINT64 = 10,
  INT64 = 11,
  FLOAT64 = 12,
}

export type GgufMetadataValue =
  | string
  | number
  | boolean
  | Array<string | number | boolean>;

export interface GgufMetadataInspection {
  generalType: string | null;
  generalArchitecture: string | null;
  clipProjectorType: string | null;
  clipVisionProjectorType: string | null;
  clipHasVisionEncoder: boolean | null;
  scannedKeyCount: number;
  stoppedEarlyAtKey: string | null;
}

export interface InspectGgufMetadataOptions {
  signal?: AbortSignal;
  initialPrefixBytes?: number;
  maxPrefixBytes?: number;
  maxKvEntries?: number;
}

class PrefixBlobReader {
  private prefixBuffer: ArrayBuffer | null = null;
  private prefixView: DataView | null = null;

  constructor(
    private readonly blob: Blob,
    private readonly initialPrefixBytes: number,
    private readonly maxPrefixBytes: number
  ) {}

  public async ensure(endOffset: number, signal?: AbortSignal): Promise<void> {
    if (endOffset < 0) {
      throw new Error(`Refusing to read a negative GGUF offset (${endOffset}).`);
    }

    throwIfAborted(signal);

    const currentLength = this.prefixBuffer?.byteLength ?? 0;
    if (currentLength >= endOffset) {
      return;
    }

    let nextLength = Math.max(this.initialPrefixBytes, currentLength || 0);
    while (nextLength < endOffset && nextLength < this.maxPrefixBytes) {
      nextLength *= 2;
    }
    nextLength = Math.max(endOffset, Math.min(nextLength, this.maxPrefixBytes));

    if (nextLength < endOffset) {
      throw new Error(
        `GGUF metadata prefix exceeded the configured ${this.maxPrefixBytes} byte limit.`
      );
    }

    const prefix = await this.blob.slice(0, nextLength).arrayBuffer();
    throwIfAborted(signal);

    this.prefixBuffer = prefix;
    this.prefixView = new DataView(prefix);
  }

  public view(): DataView {
    if (this.prefixView == null) {
      throw new Error('GGUF prefix view is not initialized.');
    }
    return this.prefixView;
  }
}

export async function inspectGgufMetadata(
  blob: Blob,
  options: InspectGgufMetadataOptions = {}
): Promise<GgufMetadataInspection | null> {
  const initialPrefixBytes = Math.max(
    16,
    options.initialPrefixBytes ?? DEFAULT_INITIAL_PREFIX_BYTES
  );
  const maxPrefixBytes = Math.max(
    initialPrefixBytes,
    options.maxPrefixBytes ?? DEFAULT_MAX_PREFIX_BYTES
  );
  const maxKvEntries = Math.max(1, options.maxKvEntries ?? DEFAULT_MAX_KV_ENTRIES);
  if (blob.size < 24) {
    return null;
  }
  const reader = new PrefixBlobReader(blob, initialPrefixBytes, maxPrefixBytes);

  await reader.ensure(24, options.signal);
  const view = reader.view();
  const magic = view.getUint32(0, true);
  if (magic !== GGUF_MAGIC) {
    return null;
  }

  const version = view.getUint32(4, true);
  if (!GGUF_SUPPORTED_VERSIONS.has(version)) {
    throw new Error(`Unsupported GGUF version ${version}.`);
  }

  const kvCount = readUint64AsNumber(view, 16);
  let offset = 24;

  const collected = new Map<string, GgufMetadataValue>();
  let scannedKeyCount = 0;
  let stoppedEarlyAtKey: string | null = null;

  for (let index = 0; index < kvCount; index += 1) {
    if (index >= maxKvEntries) {
      stoppedEarlyAtKey = '(max-entries)';
      break;
    }

    const keyString = await readString(reader, offset, options.signal);
    const key = keyString.value;
    offset = keyString.nextOffset;

    await reader.ensure(offset + 4, options.signal);
    const valueType = reader.view().getUint32(offset, true) as GgufValueType;
    offset += 4;
    scannedKeyCount += 1;

    if (EARLY_STOP_KEYS.has(key) && hasUsefulMetadata(collected)) {
      stoppedEarlyAtKey = key;
      break;
    }

    if (TARGET_KEYS.has(key)) {
      const parsed = await readValue(reader, offset, valueType, options.signal);
      collected.set(key, parsed.value);
      offset = parsed.nextOffset;
    } else {
      offset = await skipValue(reader, offset, valueType, options.signal);
    }
  }

  return {
    generalType: asStringValue(collected.get('general.type')),
    generalArchitecture: asStringValue(collected.get('general.architecture')),
    clipProjectorType: asStringValue(collected.get('clip.projector_type')),
    clipVisionProjectorType: asStringValue(collected.get('clip.vision.projector_type')),
    clipHasVisionEncoder: asBooleanValue(collected.get('clip.has_vision_encoder')),
    scannedKeyCount,
    stoppedEarlyAtKey,
  };
}

function hasUsefulMetadata(values: ReadonlyMap<string, GgufMetadataValue>): boolean {
  return (
    values.has('general.type') ||
    values.has('general.architecture') ||
    values.has('clip.projector_type') ||
    values.has('clip.vision.projector_type') ||
    values.has('clip.has_vision_encoder')
  );
}

async function readValue(
  reader: PrefixBlobReader,
  offset: number,
  valueType: GgufValueType,
  signal?: AbortSignal
): Promise<{ value: GgufMetadataValue; nextOffset: number }> {
  switch (valueType) {
    case GgufValueType.UINT8:
      return readScalar(reader, offset, 1, signal, (view, cursor) =>
        view.getUint8(cursor)
      );
    case GgufValueType.INT8:
      return readScalar(reader, offset, 1, signal, (view, cursor) =>
        view.getInt8(cursor)
      );
    case GgufValueType.UINT16:
      return readScalar(reader, offset, 2, signal, (view, cursor) =>
        view.getUint16(cursor, true)
      );
    case GgufValueType.INT16:
      return readScalar(reader, offset, 2, signal, (view, cursor) =>
        view.getInt16(cursor, true)
      );
    case GgufValueType.UINT32:
      return readScalar(reader, offset, 4, signal, (view, cursor) =>
        view.getUint32(cursor, true)
      );
    case GgufValueType.INT32:
      return readScalar(reader, offset, 4, signal, (view, cursor) =>
        view.getInt32(cursor, true)
      );
    case GgufValueType.FLOAT32:
      return readScalar(reader, offset, 4, signal, (view, cursor) =>
        view.getFloat32(cursor, true)
      );
    case GgufValueType.BOOL:
      return readScalar(reader, offset, 1, signal, (view, cursor) =>
        view.getUint8(cursor) !== 0
      );
    case GgufValueType.STRING:
      return readString(reader, offset, signal);
    case GgufValueType.UINT64:
      return readScalar(reader, offset, 8, signal, (view, cursor) =>
        readUint64AsNumber(view, cursor)
      );
    case GgufValueType.INT64:
      return readScalar(reader, offset, 8, signal, (view, cursor) =>
        readInt64AsNumber(view, cursor)
      );
    case GgufValueType.FLOAT64:
      return readScalar(reader, offset, 8, signal, (view, cursor) =>
        view.getFloat64(cursor, true)
      );
    case GgufValueType.ARRAY:
      return readArray(reader, offset, signal);
    default:
      throw new Error(`Unsupported GGUF value type ${valueType}.`);
  }
}

async function readArray(
  reader: PrefixBlobReader,
  offset: number,
  signal?: AbortSignal
): Promise<{ value: Array<string | number | boolean>; nextOffset: number }> {
  await reader.ensure(offset + 12, signal);
  const view = reader.view();
  const itemType = view.getUint32(offset, true) as GgufValueType;
  const itemCount = readUint64AsNumber(view, offset + 4);
  offset += 12;

  const values: Array<string | number | boolean> = [];
  for (let index = 0; index < itemCount; index += 1) {
    const item = await readValue(reader, offset, itemType, signal);
    if (Array.isArray(item.value)) {
      throw new Error('Nested GGUF arrays are not supported in metadata inspection.');
    }
    values.push(item.value);
    offset = item.nextOffset;
  }

  return {
    value: values,
    nextOffset: offset,
  };
}

async function skipValue(
  reader: PrefixBlobReader,
  offset: number,
  valueType: GgufValueType,
  signal?: AbortSignal
): Promise<number> {
  switch (valueType) {
    case GgufValueType.UINT8:
    case GgufValueType.INT8:
    case GgufValueType.BOOL:
      return offset + 1;
    case GgufValueType.UINT16:
    case GgufValueType.INT16:
      return offset + 2;
    case GgufValueType.UINT32:
    case GgufValueType.INT32:
    case GgufValueType.FLOAT32:
      return offset + 4;
    case GgufValueType.UINT64:
    case GgufValueType.INT64:
    case GgufValueType.FLOAT64:
      return offset + 8;
    case GgufValueType.STRING: {
      const stringValue = await readString(reader, offset, signal);
      return stringValue.nextOffset;
    }
    case GgufValueType.ARRAY: {
      await reader.ensure(offset + 12, signal);
      const view = reader.view();
      const itemType = view.getUint32(offset, true) as GgufValueType;
      const itemCount = readUint64AsNumber(view, offset + 4);
      offset += 12;
      for (let index = 0; index < itemCount; index += 1) {
        offset = await skipValue(reader, offset, itemType, signal);
      }
      return offset;
    }
    default:
      throw new Error(`Unsupported GGUF value type ${valueType}.`);
  }
}

async function readScalar<T extends string | number | boolean>(
  reader: PrefixBlobReader,
  offset: number,
  byteLength: number,
  signal: AbortSignal | undefined,
  read: (view: DataView, offset: number) => T
): Promise<{ value: T; nextOffset: number }> {
  await reader.ensure(offset + byteLength, signal);
  const value = read(reader.view(), offset);
  return {
    value,
    nextOffset: offset + byteLength,
  };
}

async function readString(
  reader: PrefixBlobReader,
  offset: number,
  signal?: AbortSignal
): Promise<{ value: string; nextOffset: number }> {
  await reader.ensure(offset + 8, signal);
  const view = reader.view();
  const byteLength = readUint64AsNumber(view, offset);
  const dataOffset = offset + 8;
  const endOffset = dataOffset + byteLength;
  await reader.ensure(endOffset, signal);
  const bytes = new Uint8Array(reader.view().buffer, dataOffset, byteLength);
  return {
    value: textDecoder.decode(bytes),
    nextOffset: endOffset,
  };
}

function readUint64AsNumber(view: DataView, offset: number): number {
  const value = Number(view.getBigUint64(offset, true));
  if (!Number.isSafeInteger(value)) {
    throw new Error(`GGUF uint64 value at ${offset} exceeds JS safe integer range.`);
  }
  return value;
}

function readInt64AsNumber(view: DataView, offset: number): number {
  const value = Number(view.getBigInt64(offset, true));
  if (!Number.isSafeInteger(value)) {
    throw new Error(`GGUF int64 value at ${offset} exceeds JS safe integer range.`);
  }
  return value;
}

function asStringValue(value: GgufMetadataValue | undefined): string | null {
  return typeof value === 'string' ? value : null;
}

function asBooleanValue(value: GgufMetadataValue | undefined): boolean | null {
  return typeof value === 'boolean' ? value : null;
}

function throwIfAborted(signal: AbortSignal | undefined): void {
  if (signal?.aborted) {
    throw new Error('GGUF metadata inspection was aborted.');
  }
}
