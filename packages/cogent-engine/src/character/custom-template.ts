// Custom chat-prompt builder that is cross-model-compatible.
//
// We deliberately avoid llama.cpp's native chat template application because
// llama.cpp's `common_chat_format_single` (used via CE_ApplyChatTemplate)
// formats only the last message as a delta against prior messages. When the
// caller passes `[system, user]` as the first turn, the system message is
// silently dropped from the rendered prompt, breaking persona conditioning.
//
// This module renders the full chat history explicitly into a raw text
// prompt. The caller then passes the result through to the runtime with
// `promptFormat: 'raw'`. BOS text is obtained from the model's vocabulary
// (via WasmBridge.getBosText) so every model can see its expected leading
// marker.

import type { ChatMessage } from '../core/inference-types.js';

export type ChatFormat =
  | 'chatml'
  | 'llama3'
  | 'llama2'
  | 'mistral'
  | 'gemma'
  | 'phi3';

export interface BuildChatPromptOptions {
  format: ChatFormat;
  messages: ChatMessage[];
  /**
   * BOS text to emit at the very start of the prompt. Obtain via
   * WasmBridge.getBosText(). Pass '' to suppress.
   */
  bosText: string;
  /** When true, append the trailing role marker that primes the model to
   *  begin an assistant turn. Default true.
   */
  addGenerationPrompt?: boolean;
}

/**
 * Attempts to infer a ChatFormat from a Jinja chat_template string
 * embedded in the GGUF. Returns null when no known signature matches.
 */
export function sniffChatFormat(templateSource: string | null): ChatFormat | null {
  if (!templateSource) {
    return null;
  }
  const t = templateSource;
  if (t.includes('<|start_header_id|>') && t.includes('<|end_header_id|>')) {
    return 'llama3';
  }
  if (t.includes('<|im_start|>') && t.includes('<|im_end|>')) {
    return 'chatml';
  }
  if (t.includes('<start_of_turn>') && t.includes('<end_of_turn>')) {
    return 'gemma';
  }
  if (t.includes('<|user|>') && t.includes('<|assistant|>') && t.includes('<|end|>')) {
    return 'phi3';
  }
  if (t.includes('[INST]') && t.includes('[/INST]')) {
    // Llama2 and Mistral both use [INST]/[/INST]; distinguish by <<SYS>>.
    if (t.includes('<<SYS>>')) {
      return 'llama2';
    }
    return 'mistral';
  }
  return null;
}

/**
 * Builds a complete prompt string suitable for `promptFormat: 'raw'`
 * inference requests.
 */
export function buildChatPrompt(options: BuildChatPromptOptions): string {
  const { format, messages, bosText } = options;
  const addGenerationPrompt = options.addGenerationPrompt ?? true;

  switch (format) {
    case 'chatml':
      return renderChatml(bosText, messages, addGenerationPrompt);
    case 'llama3':
      return renderLlama3(bosText, messages, addGenerationPrompt);
    case 'llama2':
      return renderLlama2(bosText, messages, addGenerationPrompt);
    case 'mistral':
      return renderMistral(bosText, messages, addGenerationPrompt);
    case 'gemma':
      return renderGemma(bosText, messages, addGenerationPrompt);
    case 'phi3':
      return renderPhi3(bosText, messages, addGenerationPrompt);
    default: {
      const never: never = format;
      throw new Error(`Unsupported chat format: ${String(never)}`);
    }
  }
}

// -----------------------------------------------------------------------------
// ChatML (OpenAI / Qwen / Hermes / LFM2 / many others)
// -----------------------------------------------------------------------------
function renderChatml(
  bosText: string,
  messages: ChatMessage[],
  addGenerationPrompt: boolean
): string {
  let out = bosText;
  for (const m of messages) {
    out += `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`;
  }
  if (addGenerationPrompt) {
    out += '<|im_start|>assistant\n';
  }
  return out;
}

// -----------------------------------------------------------------------------
// Llama 3
//   <|begin_of_text|>
//   <|start_header_id|>system<|end_header_id|>\n\n<content><|eot_id|>
//   <|start_header_id|>user<|end_header_id|>\n\n<content><|eot_id|>
//   <|start_header_id|>assistant<|end_header_id|>\n\n
// BOS is usually emitted by the vocab BOS text; we skip a second one here.
// -----------------------------------------------------------------------------
function renderLlama3(
  bosText: string,
  messages: ChatMessage[],
  addGenerationPrompt: boolean
): string {
  let out = bosText;
  for (const m of messages) {
    out += `<|start_header_id|>${m.role}<|end_header_id|>\n\n${m.content}<|eot_id|>`;
  }
  if (addGenerationPrompt) {
    out += '<|start_header_id|>assistant<|end_header_id|>\n\n';
  }
  return out;
}

// -----------------------------------------------------------------------------
// Llama 2
//   <s>[INST] <<SYS>>\n<system>\n<</SYS>>\n\n<user> [/INST] <assistant> </s>
//   <s>[INST] <user2> [/INST] <assistant2> </s> ...
// We fold any leading `system` into the first user turn's [INST] block.
// -----------------------------------------------------------------------------
function renderLlama2(
  bosText: string,
  messages: ChatMessage[],
  addGenerationPrompt: boolean
): string {
  let out = '';
  let pendingSystem: string | null = null;
  let inTurn = false;

  for (const m of messages) {
    if (m.role === 'system') {
      pendingSystem = m.content;
      continue;
    }
    if (m.role === 'user') {
      out += `${bosText}[INST] `;
      if (pendingSystem !== null) {
        out += `<<SYS>>\n${pendingSystem}\n<</SYS>>\n\n`;
        pendingSystem = null;
      }
      out += `${m.content} [/INST]`;
      inTurn = true;
      continue;
    }
    if (m.role === 'assistant') {
      out += ` ${m.content} </s>`;
      inTurn = false;
      continue;
    }
  }
  if (addGenerationPrompt && inTurn) {
    // Leave the trailing [/INST] in place; model generates next.
  }
  return out;
}

// -----------------------------------------------------------------------------
// Mistral
//   <s>[INST] <user> [/INST] <assistant></s>[INST] <user2> [/INST] ...
// Mistral has no dedicated system role; we prepend system text to the first
// user message.
// -----------------------------------------------------------------------------
function renderMistral(
  bosText: string,
  messages: ChatMessage[],
  addGenerationPrompt: boolean
): string {
  let out = bosText;
  let pendingSystem: string | null = null;
  let firstTurn = true;

  for (const m of messages) {
    if (m.role === 'system') {
      pendingSystem = m.content;
      continue;
    }
    if (m.role === 'user') {
      if (!firstTurn) {
        out += '[INST] ';
      } else {
        out += '[INST] ';
        firstTurn = false;
      }
      if (pendingSystem !== null) {
        out += `${pendingSystem}\n\n${m.content} [/INST]`;
        pendingSystem = null;
      } else {
        out += `${m.content} [/INST]`;
      }
      continue;
    }
    if (m.role === 'assistant') {
      out += ` ${m.content}</s>`;
      continue;
    }
  }
  void addGenerationPrompt;
  return out;
}

// -----------------------------------------------------------------------------
// Gemma
//   <bos><start_of_turn>user\n<content><end_of_turn>\n
//   <start_of_turn>model\n<content><end_of_turn>\n
//   <start_of_turn>model\n
// Gemma has no system role; system text is merged into the first user turn.
// -----------------------------------------------------------------------------
function renderGemma(
  bosText: string,
  messages: ChatMessage[],
  addGenerationPrompt: boolean
): string {
  let out = bosText;
  let pendingSystem: string | null = null;

  for (const m of messages) {
    if (m.role === 'system') {
      pendingSystem = m.content;
      continue;
    }
    const role = m.role === 'assistant' ? 'model' : 'user';
    let content = m.content;
    if (role === 'user' && pendingSystem !== null) {
      content = `${pendingSystem}\n\n${content}`;
      pendingSystem = null;
    }
    out += `<start_of_turn>${role}\n${content}<end_of_turn>\n`;
  }
  if (addGenerationPrompt) {
    out += '<start_of_turn>model\n';
  }
  return out;
}

// -----------------------------------------------------------------------------
// Phi-3
//   <|system|>\n<content><|end|>\n
//   <|user|>\n<content><|end|>\n
//   <|assistant|>\n<content><|end|>\n
//   <|assistant|>\n
// -----------------------------------------------------------------------------
function renderPhi3(
  bosText: string,
  messages: ChatMessage[],
  addGenerationPrompt: boolean
): string {
  let out = bosText;
  for (const m of messages) {
    out += `<|${m.role}|>\n${m.content}<|end|>\n`;
  }
  if (addGenerationPrompt) {
    out += '<|assistant|>\n';
  }
  return out;
}
