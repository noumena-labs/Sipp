import type { ChatMessage } from './inference-types.js';

export interface ChatTemplateMessage {
  role: string;
  content: string;
}

export interface ChatTemplateMetadataProvider {
  applyChatTemplate(messages: ChatTemplateMessage[], addAssistant: boolean): Promise<string>;
  getChatTemplate?(): string | null;
  getEosText?(): string;
}

export interface AppliedChatTemplateContext {
  readonly promptText: string;
  readonly boundaryMarkers: readonly string[];
  readonly templateSource: string | null;
}

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

const BOUNDARY_SENTINELS = {
  system: '__CE_BOUNDARY_SYSTEM__',
  user1: '__CE_BOUNDARY_USER1__',
  assistant: '__CE_BOUNDARY_ASSISTANT__',
  user2: '__CE_BOUNDARY_USER2__',
} as const;

export class ChatTemplatePromptRuntime {
  private boundaryInfoPromise: Promise<ChatBoundaryInfo> | undefined;

  public constructor(private readonly provider: ChatTemplateMetadataProvider) {}

  public async render(messages: readonly ChatMessage[]): Promise<AppliedChatTemplateContext> {
    const boundaryInfo = await this.getBoundaryInfo();
    const promptText = await renderAppliedChatTemplate(this.provider, messages);

    return {
      promptText,
      boundaryMarkers: buildBoundaryMarkers(boundaryInfo),
      templateSource: this.provider.getChatTemplate?.() ?? null,
    };
  }

  public async getBoundaryMarkers(): Promise<readonly string[]> {
    return buildBoundaryMarkers(await this.getBoundaryInfo());
  }

  private getBoundaryInfo(): Promise<ChatBoundaryInfo> {
    if (!this.boundaryInfoPromise) {
      this.boundaryInfoPromise = probeChatTemplateBoundaryInfo(this.provider).catch((error) => {
        this.boundaryInfoPromise = undefined;
        throw error;
      });
    }
    return this.boundaryInfoPromise;
  }
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

export async function buildAppliedChatTemplateContext(
  provider: ChatTemplateMetadataProvider,
  messages: readonly ChatMessage[],
  boundaryInfo?: ChatBoundaryInfo
): Promise<AppliedChatTemplateContext> {
  const info = boundaryInfo == null ? await probeChatTemplateBoundaryInfo(provider) : boundaryInfo;
  const promptText = await renderAppliedChatTemplate(provider, messages);

  return {
    promptText,
    boundaryMarkers: buildBoundaryMarkers(info),
    templateSource: provider.getChatTemplate?.() ?? null,
  };
}

export async function renderAppliedChatTemplate(
  provider: ChatTemplateMetadataProvider,
  messages: readonly ChatMessage[]
): Promise<string> {
  const promptText = await provider.applyChatTemplate(toTemplateMessages(messages), true);
  if (promptText.length === 0) {
    throw new Error(
      'model chat_template did not produce a prompt. Ensure the loaded GGUF includes a valid chat template.'
    );
  }
  return promptText;
}

export async function probeChatTemplateBoundaryInfo(
  provider: ChatTemplateMetadataProvider
): Promise<ChatBoundaryInfo> {
  const eosText = provider.getEosText?.() ?? '';
  return getChatBoundaryInfo(provider, eosText);
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

export function sanitizeAssistantText(
  text: string,
  boundaryMarkers: readonly string[]
): string {
  const split = splitOnChatBoundary(text, boundaryMarkers);
  return trimTrailingBoundaryPrefix(split.safeText, boundaryMarkers).trim();
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

function toTemplateMessages(messages: readonly ChatMessage[]): ChatTemplateMessage[] {
  return messages.map((message) => ({ role: message.role, content: message.content }));
}

async function getChatBoundaryInfo(
  provider: ChatTemplateMetadataProvider,
  eosText: string
): Promise<ChatBoundaryInfo> {
  const systemMessage: ChatTemplateMessage = {
    role: 'system',
    content: BOUNDARY_SENTINELS.system,
  };
  const user1Message: ChatTemplateMessage = {
    role: 'user',
    content: BOUNDARY_SENTINELS.user1,
  };
  const assistantMessage: ChatTemplateMessage = {
    role: 'assistant',
    content: BOUNDARY_SENTINELS.assistant,
  };
  const user2Message: ChatTemplateMessage = {
    role: 'user',
    content: BOUNDARY_SENTINELS.user2,
  };

  const closedUserPrompt = await provider.applyChatTemplate([systemMessage, user1Message], false);
  const primedAssistantPrompt = await provider.applyChatTemplate([systemMessage, user1Message], true);
  const closedAssistantPrompt = await provider.applyChatTemplate(
    [systemMessage, user1Message, assistantMessage],
    false
  );
  const promptWithNextUser = await provider.applyChatTemplate(
    [systemMessage, user1Message, assistantMessage, user2Message],
    false
  );
  const systemOnlyPrompt = await provider.applyChatTemplate([systemMessage], false);

  const assistantPrefix =
    primedAssistantPrompt.startsWith(closedUserPrompt)
      ? primedAssistantPrompt.slice(closedUserPrompt.length)
      : '';
  const assistantSuffix = sliceAfterSentinel(
    closedAssistantPrompt,
    BOUNDARY_SENTINELS.assistant
  );
  const nextUserAppend =
    promptWithNextUser.startsWith(closedAssistantPrompt)
      ? promptWithNextUser.slice(closedAssistantPrompt.length)
      : '';
  const nextUserPrefix = sliceBeforeSentinel(nextUserAppend, BOUNDARY_SENTINELS.user2);
  const systemPrefix = sliceBeforeSentinel(systemOnlyPrompt, BOUNDARY_SENTINELS.system);

  return {
    assistantPrefix,
    assistantSuffix,
    nextTurnPrefixes: uniqueNonEmpty([systemPrefix, nextUserPrefix, assistantPrefix]),
    eosText,
  };
}

function sliceBeforeSentinel(source: string, sentinel: string): string {
  const index = source.indexOf(sentinel);
  if (index < 0) {
    return '';
  }
  return source.slice(0, index);
}

function sliceAfterSentinel(source: string, sentinel: string): string {
  const index = source.indexOf(sentinel);
  if (index < 0) {
    return '';
  }
  return source.slice(index + sentinel.length);
}

function uniqueNonEmpty(values: readonly string[]): readonly string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const value of values) {
    if (value.length === 0 || seen.has(value)) {
      continue;
    }
    seen.add(value);
    out.push(value);
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
