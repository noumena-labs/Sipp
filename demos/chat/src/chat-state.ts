import type { ChatMessage, RequestStats } from '@noumena-labs/sipp';

export type ConversationMessageStatus = 'complete' | 'streaming' | 'error';

export interface ConversationMessage {
  readonly id: string;
  readonly role: 'user' | 'assistant';
  text: string;
  status: ConversationMessageStatus;
  readonly imageUrl?: string;
  readonly imageName?: string;
  stats?: RequestStats;
  note?: string;
}

export interface GenerationSettings {
  maxTokens: number;
  temperature: number;
  topP: number;
}

export const DEFAULT_GENERATION_SETTINGS: GenerationSettings = {
  maxTokens: 256,
  temperature: 0.7,
  topP: 0.9,
};

export function toChatMessages(
  messages: readonly ConversationMessage[]
): ChatMessage[] {
  return messages
    .filter(
      (message) =>
        message.status === 'complete' &&
        message.text.trim().length > 0
    )
    .map((message) => ({
      role: message.role,
      content: message.text,
    }));
}

export function formatRequestStats(stats: RequestStats): string {
  const fields: string[] = [];
  if (stats.decodeTokensPerSecond != null) {
    fields.push(`${stats.decodeTokensPerSecond.toFixed(1)} tok/s`);
  }
  if (stats.ttftMs != null) {
    fields.push(`${Math.round(stats.ttftMs)} ms TTFT`);
  }
  fields.push(`${stats.outputTokens} tokens`);
  return fields.join(' | ');
}
