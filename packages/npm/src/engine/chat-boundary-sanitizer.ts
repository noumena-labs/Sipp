export interface ChatBoundaryInfo {
  readonly assistantPrefix: string;
  readonly assistantSuffix: string;
  readonly nextTurnPrefixes: readonly string[];
  readonly eosText: string;
}

export interface BoundarySplit {
  readonly safeText: string;
  readonly trailingText: string;
  readonly hitBoundary: boolean;
}

export interface BoundaryConsumeResult {
  readonly safeText: string;
  readonly hitBoundary: boolean;
}

export function sliceUnstreamedSuffix(
  streamedOutputText: string,
  finalOutputText: string
): string {
  if (streamedOutputText.length === 0) {
    return finalOutputText;
  }
  if (!finalOutputText.startsWith(streamedOutputText)) {
    return '';
  }
  return finalOutputText.slice(streamedOutputText.length);
}

export class StreamingBoundaryTextSanitizer {
  private pendingText = '';
  private stopped = false;

  public constructor(private readonly boundaryMarkers: readonly string[]) {}

  public get reachedBoundary(): boolean {
    return this.stopped;
  }

  public consume(text: string): BoundaryConsumeResult {
    if (text.length === 0 || this.stopped) {
      return { safeText: '', hitBoundary: false };
    }

    this.pendingText += text;
    const split = splitOnChatBoundary(this.pendingText, this.boundaryMarkers);
    this.pendingText = split.trailingText;
    if (split.hitBoundary) {
      this.pendingText = '';
      this.stopped = true;
    }
    return { safeText: split.safeText, hitBoundary: split.hitBoundary };
  }

  public flush(): string {
    if (this.stopped) {
      this.pendingText = '';
      return '';
    }
    const out = trimTrailingBoundaryPrefix(this.pendingText, this.boundaryMarkers);
    this.pendingText = '';
    return out;
  }
}

export function buildBoundaryMarkers(info: ChatBoundaryInfo): readonly string[] {
  const markers = new Set<string>();
  if (info.assistantSuffix.length > 0) {
    markers.add(info.assistantSuffix);
  }
  for (const prefix of info.nextTurnPrefixes) {
    if (prefix.length > 0) {
      markers.add(prefix);
    }
  }
  if (info.eosText.length > 0) {
    markers.add(info.eosText);
  }
  return Array.from(markers);
}

export function splitOnChatBoundary(
  text: string,
  boundaryMarkers: readonly string[]
): BoundarySplit {
  let earliestIndex = -1;
  let matchedMarker = '';

  for (const marker of boundaryMarkers) {
    if (marker.length === 0) {
      continue;
    }
    const index = text.indexOf(marker);
    if (index >= 0 && (earliestIndex < 0 || index < earliestIndex)) {
      earliestIndex = index;
      matchedMarker = marker;
    }
  }

  if (earliestIndex >= 0) {
    return {
      safeText: text.slice(0, earliestIndex),
      trailingText: text.slice(earliestIndex + matchedMarker.length),
      hitBoundary: true,
    };
  }

  let safeLength = text.length;
  for (const marker of boundaryMarkers) {
    if (marker.length <= 1) {
      continue;
    }
    const overlap = longestSuffixPrefixOverlap(text, marker);
    safeLength = Math.min(safeLength, text.length - overlap);
  }

  return {
    safeText: text.slice(0, safeLength),
    trailingText: text.slice(safeLength),
    hitBoundary: false,
  };
}

export function trimTrailingBoundaryPrefix(
  text: string,
  boundaryMarkers: readonly string[]
): string {
  let out = text;
  let changed = true;
  while (changed && out.length > 0) {
    changed = false;
    for (const marker of boundaryMarkers) {
      if (marker.length === 0) {
        continue;
      }
      if (marker.startsWith(out)) {
        out = '';
        changed = true;
        break;
      }
    }
  }
  return out;
}

function longestSuffixPrefixOverlap(source: string, marker: string): number {
  const maxOverlapLength = Math.min(source.length, marker.length - 1);
  for (let length = maxOverlapLength; length > 0; length -= 1) {
    if (source.endsWith(marker.slice(0, length))) {
      return length;
    }
  }
  return 0;
}
