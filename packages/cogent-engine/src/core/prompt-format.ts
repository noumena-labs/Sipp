import { PromptFormatMode } from '../types.js';
import type {
  ChatTemplateContentPart,
  ChatTemplateMessage,
} from '../wasm/wasm-bridge.js';

export function normalizePromptText(value: string): string {
  return value.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
}

export function resolveEffectivePromptFormat(
  promptFormat: PromptFormatMode,
  hasMedia: boolean
): PromptFormatMode {
  void hasMedia;
  return promptFormat;
}

function appendTextPart(parts: ChatTemplateContentPart[], text: string): void {
  if (text.length === 0) {
    return;
  }
  const lastPart = parts.at(-1);
  if (lastPart?.type === 'text') {
    lastPart.text += text;
    return;
  }
  parts.push({
    type: 'text',
    text,
  });
}

export function buildChatTemplateUserMessage(
  promptText: string,
  mediaMarker?: string | null
): ChatTemplateMessage {
  if (!mediaMarker) {
    return {
      role: 'user',
      content: promptText,
    };
  }

  const markerParts: ChatTemplateContentPart[] = [];
  let searchIndex = 0;
  while (true) {
    const markerIndex = promptText.indexOf(mediaMarker, searchIndex);
    if (markerIndex < 0) {
      appendTextPart(markerParts, promptText.slice(searchIndex));
      break;
    }
    appendTextPart(markerParts, promptText.slice(searchIndex, markerIndex));
    markerParts.push({
      type: 'media_marker',
      text: mediaMarker,
    });
    searchIndex = markerIndex + mediaMarker.length;
  }

  return {
    role: 'user',
    content: markerParts.length > 0 ? markerParts : promptText,
  };
}
