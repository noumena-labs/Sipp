import type { ChatMessage } from '../core/inference-types.js';
import type { ChatTemplateMessage } from '../wasm/wasm-bridge.js';

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

interface ChatBoundaryInfo {
  readonly assistantPrefix: string;
  readonly assistantSuffix: string;
  readonly nextTurnPrefixes: readonly string[];
  readonly eosText: string;
}

const BOUNDARY_SENTINELS = {
  system: '__CE_BOUNDARY_SYSTEM__',
  user1: '__CE_BOUNDARY_USER1__',
  assistant: '__CE_BOUNDARY_ASSISTANT__',
  user2: '__CE_BOUNDARY_USER2__',
} as const;

export async function buildAppliedChatTemplateContext(
  provider: ChatTemplateMetadataProvider,
  messages: ChatMessage[]
): Promise<AppliedChatTemplateContext> {
  const promptMessages = toTemplateMessages(messages);
  const promptText = await provider.applyChatTemplate(promptMessages, true);
  if (promptText.length === 0) {
    throw new Error(
      'CharacterAgent: model chat_template did not produce a prompt. Ensure the loaded GGUF includes a valid chat template.'
    );
  }

  const eosText = provider.getEosText?.() ?? '';
  const info = await getChatBoundaryInfo(provider, eosText);
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

  return {
    promptText,
    boundaryMarkers: Array.from(markers),
    templateSource: provider.getChatTemplate?.() ?? null,
  };
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
