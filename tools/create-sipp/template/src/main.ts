import {
  Endpoint,
  SippClient,
  type ChatMessage,
  type EndpointRef,
} from '@sipphq/sipp';

const HF_QWEN_URL =
  'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf';

const statusBadge = requireElement<HTMLSpanElement>('status-badge');
const modelSourceSelect = requireElement<HTMLSelectElement>('model-source');
const customUrlInput = requireElement<HTMLInputElement>('custom-url-input');
const fileInput = requireElement<HTMLInputElement>('file-input');
const loadModelBtn = requireElement<HTMLButtonElement>('load-model-btn');
const progressContainer = requireElement<HTMLDivElement>('progress-container');
const progressBarFill = requireElement<HTMLDivElement>('progress-bar-fill');
const progressText = requireElement<HTMLSpanElement>('progress-text');
const messagesList = requireElement<HTMLDivElement>('messages-list');
const chatForm = requireElement<HTMLFormElement>('chat-form');
const userInput = requireElement<HTMLTextAreaElement>('user-input');
const sendBtn = requireElement<HTMLButtonElement>('send-btn');
const clearBtn = requireElement<HTMLButtonElement>('clear-btn');

let client: SippClient | null = null;
let activeEndpoint: EndpointRef | null = null;
const conversationHistory: ChatMessage[] = [
  { role: 'system', content: 'You are a helpful and concise AI assistant.' },
];

modelSourceSelect.addEventListener('change', () => {
  const value = modelSourceSelect.value;
  customUrlInput.classList.toggle('hidden', value !== 'custom-url');
  fileInput.classList.toggle('hidden', value !== 'file-upload');
});

loadModelBtn.addEventListener('click', async () => {
  loadModelBtn.disabled = true;
  modelSourceSelect.disabled = true;
  customUrlInput.disabled = true;
  fileInput.disabled = true;

  statusBadge.className = 'badge badge-loading';
  statusBadge.textContent = 'Initializing...';
  progressContainer.classList.remove('hidden');
  progressBarFill.style.width = '0%';
  progressText.textContent = '0%';

  try {
    if (!client) {
      client = new SippClient();
      client.subscribeEvents((event) => {
        if (event.type === 'fallback-warning') {
          appendSystemNotice(`${event.detail} Using ${event.fallbackTo}.`);
        }
      });
    }

    let modelSource: string | File;
    const selection = modelSourceSelect.value;

    if (selection === 'hf-qwen') {
      modelSource = HF_QWEN_URL;
    } else if (selection === 'custom-url') {
      const url = customUrlInput.value.trim();
      if (!url) {
        throw new Error('Please enter a valid GGUF URL');
      }
      modelSource = url;
    } else if (selection === 'file-upload') {
      const file = fileInput.files?.[0];
      if (!file) {
        throw new Error('Please select a .gguf file from your device');
      }
      modelSource = file;
    } else {
      throw new Error('Unknown model selection');
    }

    statusBadge.textContent = 'Acquiring Model...';

    const model = await client.models.add([modelSource], {
      onProgress: (progress) => {
        if (progress.totalBytes != null && progress.totalBytes > 0) {
          const pct = Math.min(
            100,
            Math.round((progress.loadedBytes / progress.totalBytes) * 100),
          );
          progressBarFill.style.width = `${pct}%`;
          const mb = (progress.loadedBytes / (1024 * 1024)).toFixed(1);
          const totalMb = (progress.totalBytes / (1024 * 1024)).toFixed(1);
          progressText.textContent = `${mb} / ${totalMb} MB (${pct}%)`;
        } else {
          const mb = (progress.loadedBytes / (1024 * 1024)).toFixed(1);
          progressText.textContent = `${mb} MB loaded`;
        }
      },
    });

    statusBadge.textContent = 'Spinning Up Runtime...';

    activeEndpoint = await client.add(
      'chat',
      Endpoint.local(model, {
        backend: 'auto',
        runtime: {
          context: { n_ctx: 2048 },
        },
      }),
    );

    statusBadge.className = 'badge badge-active';
    statusBadge.textContent = 'Model Ready';
    progressContainer.classList.add('hidden');

    userInput.disabled = false;
    sendBtn.disabled = false;
    userInput.focus();

    appendSystemNotice('Model loaded successfully! You can now start chatting.');
  } catch (err: unknown) {
    console.error('Failed to load model:', err);
    statusBadge.className = 'badge badge-idle';
    statusBadge.textContent = 'Load Failed';
    progressContainer.classList.add('hidden');
    loadModelBtn.disabled = false;
    modelSourceSelect.disabled = false;
    customUrlInput.disabled = false;
    fileInput.disabled = false;
    appendSystemNotice(`Error loading model: ${errorMessage(err)}`);
  }
});

chatForm.addEventListener('submit', async (e) => {
  e.preventDefault();
  const text = userInput.value.trim();
  if (!text || !client || !activeEndpoint) {
    return;
  }

  userInput.value = '';
  userInput.disabled = true;
  sendBtn.disabled = true;

  appendMessage('user', text);
  conversationHistory.push({ role: 'user', content: text });

  const assistantMsgElem = appendMessage('assistant', '');
  const contentElem =
    assistantMsgElem.querySelector<HTMLDivElement>('.message-content');
  if (!contentElem) {
    throw new Error('Assistant message content was not created.');
  }

  try {
    const run = client.chat(conversationHistory, {
      endpoint: activeEndpoint,
      emitTokens: true,
      maxTokens: 512,
      contextKey: 'chat-session',
    });

    let fullReply = '';
    for await (const batch of run.tokens) {
      fullReply += batch.text;
      contentElem.textContent = fullReply;
      messagesList.scrollTop = messagesList.scrollHeight;
    }

    const finalResponse = await run.response;
    if (!fullReply && finalResponse.text) {
      fullReply = finalResponse.text;
      contentElem.textContent = fullReply;
    }

    conversationHistory.push({ role: 'assistant', content: fullReply });
  } catch (err: unknown) {
    console.error('Chat error:', err);
    contentElem.textContent += `\n[Error generating response: ${errorMessage(err)}]`;
  } finally {
    userInput.disabled = false;
    sendBtn.disabled = false;
    userInput.focus();
    messagesList.scrollTop = messagesList.scrollHeight;
  }
});

userInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    chatForm.dispatchEvent(new Event('submit'));
  }
});

clearBtn.addEventListener('click', () => {
  conversationHistory.length = 1; // Keep initial system message
  messagesList.innerHTML = '';
  appendSystemNotice('Conversation cleared.');
});

function appendMessage(
  role: 'user' | 'assistant',
  content: string,
): HTMLDivElement {
  const msg = document.createElement('div');
  msg.className = `message ${role}`;
  const bubble = document.createElement('div');
  bubble.className = 'message-content';
  bubble.textContent = content;
  msg.appendChild(bubble);
  messagesList.appendChild(msg);
  messagesList.scrollTop = messagesList.scrollHeight;
  return msg;
}

function appendSystemNotice(text: string): void {
  const msg = document.createElement('div');
  msg.className = 'message system-message';
  const bubble = document.createElement('div');
  bubble.className = 'message-content';
  bubble.textContent = text;
  msg.appendChild(bubble);
  messagesList.appendChild(msg);
  messagesList.scrollTop = messagesList.scrollHeight;
}

function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function requireElement<T extends HTMLElement>(id: string): T {
  const element = document.getElementById(id);
  if (!element) {
    throw new Error(`Missing required element #${id}.`);
  }
  return element as T;
}

window.addEventListener('pagehide', () => {
  void client?.close();
});
