import { PromptFormatMode } from '../types.js';

export function normalizePromptText(value: string): string {
  return value.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
}

export function resolveEffectivePromptFormat(
  promptFormat: PromptFormatMode,
  hasMedia: boolean
): PromptFormatMode {
  if (hasMedia && promptFormat === 'auto-chat') {
    return 'raw';
  }
  return promptFormat;
}
