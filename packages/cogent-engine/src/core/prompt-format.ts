import { PromptFormatMode } from '../types.js';

const CHAT_PREFIXES = ['<|im_start|>', '<|startoftext|>', '<|begin_of_text|>'];

function normalizeLineEndings(value: string): string {
  return value.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
}

function isChatFormatted(value: string): boolean {
  return CHAT_PREFIXES.some((prefix) => value.startsWith(prefix));
}

export function formatPromptText(promptText: string, promptFormat: PromptFormatMode): string {
  const normalized = normalizeLineEndings(promptText);
  if (promptFormat === 'raw') {
    return normalized;
  }

  const trimmed = normalized.trimStart();
  if (isChatFormatted(trimmed)) {
    return trimmed;
  }

  return `<|im_start|>user\n${trimmed}\n<|im_end|>\n<|im_start|>assistant\n`;
}
