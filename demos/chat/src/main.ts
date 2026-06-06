import {
  CogentClient,
  QueryError,
  type BrowserTextRun,
  type ChatInput,
  type NativeRuntimeConfig,
} from '@noumena-labs/cogentlm';

import {
  DEFAULT_GENERATION_SETTINGS,
  formatRequestStats,
  toChatMessages,
  type ConversationMessage,
  type GenerationSettings,
} from './chat-state.js';
import {
  CURATED_MODELS,
  projectorRequirementMessage,
  resolveModelSelection,
  type ModelSelection,
  type ResolvedModelSelection,
} from './model-registry.js';

const DEFAULT_MODEL_ID = 'qwen2.5-0.5b-instruct';
const DEFAULT_RUNTIME: NativeRuntimeConfig = {
  placement: {
    gpu_layers: 'all',
  },
  context: {
    n_ctx: 2048,
    n_parallel: 1,
  },
  cache: {
    mode: 'live_slot_and_snapshot',
    retained_prefix_tokens: 256,
    snapshot_interval_tokens: 32,
  },
  sampling: {
    top_k: 40,
    min_p: 0.05,
    repeat_penalty: 1.05,
  },
};

interface LoadedModel {
  readonly selection: ResolvedModelSelection;
  readonly backend: string;
}

interface PendingImage {
  readonly bytes: Uint8Array;
  readonly name: string;
  readonly objectUrl: string;
}

type PickerMode = 'curated' | 'custom';
type CustomSourceMode = 'url' | 'file';

let client: CogentClient | null = null;
let loadedModel: LoadedModel | null = null;
let activeRun: BrowserTextRun | null = null;
let activeRunCancelled = false;
let messages: ConversationMessage[] = [];
let pendingImage: PendingImage | null = null;
let selectedModelId = DEFAULT_MODEL_ID;
let pickerMode: PickerMode = 'curated';
let customSourceMode: CustomSourceMode = 'url';
let customFile: File | null = null;
let settings: GenerationSettings = { ...DEFAULT_GENERATION_SETTINGS };
let sessionId = createSessionId();

const modelDialog = element<HTMLDialogElement>('model-dialog');
const settingsDialog = element<HTMLDialogElement>('settings-dialog');
const modelOptions = element<HTMLDivElement>('model-options');
const curatedPanel = element<HTMLElement>('curated-panel');
const customPanel = element<HTMLElement>('custom-panel');
const customUrlPanel = element<HTMLElement>('custom-url-panel');
const customFilePanel = element<HTMLElement>('custom-file-panel');
const customUrlInput = element<HTMLInputElement>('custom-model-url');
const customFileInput = element<HTMLInputElement>('custom-model-file');
const customFileName = element<HTMLElement>('custom-file-name');
const loadModelButton = element<HTMLButtonElement>('load-model-button');
const closeModelButton = element<HTMLButtonElement>('close-model-button');
const loadStatus = element<HTMLElement>('load-status');
const loadProgress = element<HTMLProgressElement>('load-progress');
const modelButton = element<HTMLButtonElement>('model-button');
const modelName = element<HTMLElement>('model-name');
const modelCapability = element<HTMLElement>('model-capability');
const statusDot = element<HTMLElement>('status-dot');
const statusText = element<HTMLElement>('status-text');
const backendValue = element<HTMLElement>('backend-value');
const throughputValue = element<HTMLElement>('throughput-value');
const transcript = element<HTMLElement>('transcript');
const emptyState = element<HTMLElement>('empty-state');
const emptyModelName = element<HTMLElement>('empty-model-name');
const composer = element<HTMLFormElement>('composer');
const promptInput = element<HTMLTextAreaElement>('prompt-input');
const sendButton = element<HTMLButtonElement>('send-button');
const stopButton = element<HTMLButtonElement>('stop-button');
const imageButton = element<HTMLButtonElement>('image-button');
const imageInput = element<HTMLInputElement>('image-input');
const attachment = element<HTMLElement>('attachment');
const attachmentImage = element<HTMLImageElement>('attachment-image');
const attachmentName = element<HTMLElement>('attachment-name');
const removeAttachmentButton = element<HTMLButtonElement>('remove-attachment-button');
const newChatButton = element<HTMLButtonElement>('new-chat-button');
const settingsButton = element<HTMLButtonElement>('settings-button');
const maxTokensInput = element<HTMLInputElement>('max-tokens-input');
const temperatureInput = element<HTMLInputElement>('temperature-input');
const temperatureValue = element<HTMLOutputElement>('temperature-value');
const topPInput = element<HTMLInputElement>('top-p-input');
const topPValue = element<HTMLOutputElement>('top-p-value');

initialize();

function initialize(): void {
  renderModelOptions();
  renderPickerMode();
  renderConversation();
  syncSettingsControls();
  setModelStatus('idle', 'No model loaded');
  syncComposerState();

  modelButton.addEventListener('click', openModelDialog);
  closeModelButton.addEventListener('click', () => modelDialog.close());
  loadModelButton.addEventListener('click', () => {
    void loadSelectedModel();
  });
  settingsButton.addEventListener('click', () => settingsDialog.showModal());
  newChatButton.addEventListener('click', resetConversation);
  composer.addEventListener('submit', (event) => {
    event.preventDefault();
    void sendMessage();
  });
  promptInput.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      void sendMessage();
    }
  });
  promptInput.addEventListener('input', resizeComposer);
  stopButton.addEventListener('click', stopGeneration);
  imageButton.addEventListener('click', () => imageInput.click());
  imageInput.addEventListener('change', () => {
    void attachSelectedImage();
  });
  removeAttachmentButton.addEventListener('click', clearPendingImage);
  customFileInput.addEventListener('change', () => {
    customFile = customFileInput.files?.[0] ?? null;
    customFileName.textContent = customFile?.name ?? 'No file selected';
  });

  document.querySelectorAll<HTMLButtonElement>('[data-picker-mode]').forEach((button) => {
    button.addEventListener('click', () => {
      pickerMode = button.dataset.pickerMode as PickerMode;
      renderPickerMode();
    });
  });
  document.querySelectorAll<HTMLButtonElement>('[data-custom-source]').forEach((button) => {
    button.addEventListener('click', () => {
      customSourceMode = button.dataset.customSource as CustomSourceMode;
      renderPickerMode();
    });
  });
  document.querySelectorAll<HTMLButtonElement>('[data-topic]').forEach((button) => {
    button.addEventListener('click', () => {
      promptInput.value = button.dataset.topic ?? '';
      resizeComposer();
      promptInput.focus();
    });
  });
  document.querySelectorAll<HTMLButtonElement>('[data-close-dialog]').forEach((button) => {
    button.addEventListener('click', () => {
      const dialog = button.closest('dialog');
      dialog?.close();
    });
  });

  maxTokensInput.addEventListener('change', updateSettings);
  temperatureInput.addEventListener('input', updateSettings);
  topPInput.addEventListener('input', updateSettings);

  modelDialog.addEventListener('cancel', (event) => {
    if (loadedModel == null) {
      event.preventDefault();
    }
  });

  requestAnimationFrame(openModelDialog);
}

function element<T extends HTMLElement>(id: string): T {
  const value = document.getElementById(id);
  if (value == null) {
    throw new Error(`Missing required element "#${id}".`);
  }
  return value as T;
}

function renderModelOptions(): void {
  modelOptions.replaceChildren(
    ...CURATED_MODELS.map((model) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'model-option';
      button.dataset.modelId = model.id;
      button.setAttribute('aria-pressed', String(model.id === selectedModelId));
      if (model.id === selectedModelId) {
        button.classList.add('selected');
      }

      const main = document.createElement('span');
      main.className = 'model-option-main';

      const title = document.createElement('strong');
      title.textContent = model.name;
      const detail = document.createElement('span');
      detail.textContent = `${model.publisher} | ${model.detail}`;
      main.append(title, detail);

      const metadata = document.createElement('span');
      metadata.className = 'model-option-metadata';
      const capability = document.createElement('span');
      capability.className = `capability-badge ${model.capability}`;
      capability.textContent = model.capability === 'vision' ? 'Text + Vision' : 'Text';
      const size = document.createElement('span');
      size.textContent = model.sizeLabel;
      metadata.append(capability, size);
      if (model.recommended === true) {
        const recommended = document.createElement('span');
        recommended.className = 'recommended-label';
        recommended.textContent = 'Recommended';
        metadata.append(recommended);
      }

      button.append(main, metadata);
      button.addEventListener('click', () => {
        selectedModelId = model.id;
        renderModelOptions();
      });
      return button;
    })
  );
}

function renderPickerMode(): void {
  document.querySelectorAll<HTMLButtonElement>('[data-picker-mode]').forEach((button) => {
    const selected = button.dataset.pickerMode === pickerMode;
    button.classList.toggle('selected', selected);
    button.setAttribute('aria-pressed', String(selected));
  });
  document.querySelectorAll<HTMLButtonElement>('[data-custom-source]').forEach((button) => {
    const selected = button.dataset.customSource === customSourceMode;
    button.classList.toggle('selected', selected);
    button.setAttribute('aria-pressed', String(selected));
  });

  curatedPanel.hidden = pickerMode !== 'curated';
  customPanel.hidden = pickerMode !== 'custom';
  customUrlPanel.hidden = customSourceMode !== 'url';
  customFilePanel.hidden = customSourceMode !== 'file';
}

function openModelDialog(): void {
  closeModelButton.hidden = loadedModel == null;
  loadStatus.textContent = loadedModel == null
    ? 'Select a model to begin.'
    : `Current model: ${loadedModel.selection.name}`;
  resetLoadProgress();
  if (!modelDialog.open) {
    modelDialog.showModal();
  }
}

async function loadSelectedModel(): Promise<void> {
  let resolved: ResolvedModelSelection;
  try {
    resolved = resolveModelSelection(currentModelSelection());
  } catch (error) {
    showLoadError(error);
    return;
  }
  const navigatorWithGpu = navigator as Navigator & { readonly gpu?: unknown };
  if (navigatorWithGpu.gpu == null) {
    showLoadError(new Error('WebGPU is unavailable in this browser.'));
    return;
  }

  loadModelButton.disabled = true;
  closeModelButton.disabled = true;
  loadProgress.hidden = false;
  loadProgress.removeAttribute('value');
  loadStatus.className = 'dialog-status';
  loadStatus.textContent = `Preparing ${resolved.name}...`;
  setModelStatus('loading', 'Loading model');

  const nextClient = new CogentClient();
  try {
    await nextClient.add('chat-model', {
      kind: 'local',
      source: resolved.source,
      options: {
        backend: 'webgpu',
        observability: 'runtime',
        runtime: DEFAULT_RUNTIME,
        onProgress: (progress) => {
          const phase = formatLoadPhase(progress.phase);
          const percent = progress.percent;
          loadStatus.textContent = percent == null
            ? `${phase} ${resolved.name}...`
            : `${phase} ${resolved.name}... ${Math.round(percent)}%`;
          if (percent == null) {
            loadProgress.removeAttribute('value');
          } else {
            loadProgress.value = percent;
          }
        },
      },
    });
    const info =
      nextClient.currentLocal() ??
      nextClient.observability.current().model;
    if (info == null) {
      throw new Error(`"${resolved.name}" did not create a local endpoint.`);
    }

    if (info.status === 'needs_projector') {
      throw new Error(projectorRequirementMessage(resolved));
    }
    if (!info.loaded) {
      throw new Error(`"${resolved.name}" was installed but did not become ready.`);
    }

    const previousClient = client;
    client = nextClient;
    const selectedBackend = nextClient.state().backend.selected;
    loadedModel = {
      selection: resolved,
      backend: selectedBackend === 'unknown' ? 'webgpu' : selectedBackend,
    };
    if (previousClient != null) {
      await previousClient.close().catch(() => undefined);
    }

    resetConversation();
    updateLoadedModelUi();
    modelDialog.close();
    promptInput.focus();
  } catch (error) {
    await nextClient.close().catch(() => undefined);
    showLoadError(error);
    if (loadedModel == null) {
      setModelStatus('error', 'Model load failed');
    } else {
      setModelStatus('ready', loadedModel.selection.name);
    }
  } finally {
    loadModelButton.disabled = false;
    closeModelButton.disabled = false;
    syncComposerState();
  }
}

function currentModelSelection(): ModelSelection {
  if (pickerMode === 'curated') {
    return { kind: 'curated', modelId: selectedModelId };
  }
  if (customSourceMode === 'file') {
    if (customFile == null) {
      throw new Error('Choose a local GGUF model file.');
    }
    return { kind: 'custom-file', file: customFile };
  }
  return { kind: 'custom-url', url: customUrlInput.value };
}

function updateLoadedModelUi(): void {
  if (loadedModel == null) {
    return;
  }
  const capability = loadedModel.selection.capability;
  modelName.textContent = loadedModel.selection.name;
  emptyModelName.textContent = loadedModel.selection.name;
  modelCapability.textContent = capability === 'vision' ? 'Text + Vision' : 'Text';
  modelCapability.className = `capability-badge ${capability}`;
  modelCapability.hidden = false;
  backendValue.textContent = loadedModel.backend.toUpperCase();
  throughputValue.textContent = '--';
  imageButton.hidden = capability !== 'vision';
  setModelStatus('ready', loadedModel.selection.name);
}

async function sendMessage(): Promise<void> {
  const currentClient = client;
  const currentModel = loadedModel;
  const text = promptInput.value.trim();
  if (currentClient == null || currentModel == null || activeRun != null || text.length === 0) {
    return;
  }
  if (pendingImage != null && currentModel.selection.capability !== 'vision') {
    showComposerError('The loaded model does not accept image input.');
    return;
  }

  const image = pendingImage;
  pendingImage = null;
  promptInput.value = '';
  resizeComposer();
  renderAttachment();

  const userMessage: ConversationMessage = {
    id: createMessageId(),
    role: 'user',
    text,
    status: 'complete',
    ...(image == null
      ? {}
      : {
          imageUrl: image.objectUrl,
          imageName: image.name,
        }),
  };
  messages.push(userMessage);
  const chatMessages = toChatMessages(messages);

  const assistantMessage: ConversationMessage = {
    id: createMessageId(),
    role: 'assistant',
    text: '',
    status: 'streaming',
  };
  messages.push(assistantMessage);
  renderConversation();

  const input: ChatInput = image == null
    ? chatMessages
    : {
        messages: chatMessages,
        media: [image.bytes],
      };
  const run = currentClient.chat(input, {
    emitTokens: true,
    session: sessionId,
    maxTokens: settings.maxTokens,
    temperature: settings.temperature,
    topP: settings.topP,
  });

  activeRun = run;
  activeRunCancelled = false;
  syncComposerState();
  setModelStatus('running', 'Generating');

  try {
    let streamedText = '';
    for await (const batch of run.tokens) {
      streamedText += batch.text;
      assistantMessage.text = streamedText;
      updateMessageElement(assistantMessage);
    }

    const result = await run.response;
    assistantMessage.text = result.text;
    assistantMessage.status = 'complete';
    assistantMessage.stats = result.stats;
    throughputValue.textContent = result.stats.decodeTokensPerSecond == null
      ? '--'
      : `${result.stats.decodeTokensPerSecond.toFixed(1)} tok/s`;
    updateMessageElement(assistantMessage);
  } catch (error) {
    if (activeRunCancelled) {
      assistantMessage.status = 'complete';
      assistantMessage.note = 'Stopped';
      if (assistantMessage.text.trim().length === 0) {
        assistantMessage.text = 'Generation stopped.';
      }
    } else {
      assistantMessage.status = 'error';
      assistantMessage.text = `Generation failed: ${errorMessage(error)}`;
    }
    updateMessageElement(assistantMessage);
  } finally {
    if (activeRun === run) {
      activeRun = null;
    }
    setModelStatus('ready', currentModel.selection.name);
    syncComposerState();
    promptInput.focus();
  }
}

function stopGeneration(): void {
  if (activeRun == null) {
    return;
  }
  activeRunCancelled = true;
  activeRun.cancel('Stopped by user.');
}

function resetConversation(): void {
  stopGeneration();
  for (const message of messages) {
    if (message.imageUrl != null) {
      URL.revokeObjectURL(message.imageUrl);
    }
  }
  messages = [];
  sessionId = createSessionId();
  clearPendingImage();
  throughputValue.textContent = '--';
  renderConversation();
}

function renderConversation(): void {
  transcript.replaceChildren();
  const hasMessages = messages.length > 0;
  emptyState.hidden = hasMessages;
  transcript.hidden = !hasMessages;
  if (!hasMessages) {
    return;
  }

  for (const message of messages) {
    transcript.append(createMessageElement(message));
  }
  scrollConversation();
}

function createMessageElement(message: ConversationMessage): HTMLElement {
  const article = document.createElement('article');
  article.className = `message ${message.role} ${message.status}`;
  article.dataset.messageId = message.id;

  const role = document.createElement('div');
  role.className = 'message-role';
  role.textContent = message.role === 'user'
    ? 'You'
    : loadedModel?.selection.name ?? 'Assistant';

  const body = document.createElement('div');
  body.className = 'message-body';

  if (message.imageUrl != null) {
    const image = document.createElement('img');
    image.className = 'message-image';
    image.src = message.imageUrl;
    image.alt = message.imageName ?? 'Attached image';
    body.append(image);
  }

  const content = document.createElement('div');
  content.className = 'message-content';
  content.textContent = message.text;
  body.append(content);

  const metrics = document.createElement('div');
  metrics.className = 'message-metrics';
  metrics.textContent = messageMetadata(message);
  metrics.hidden = metrics.textContent.length === 0;
  body.append(metrics);

  article.append(role, body);
  return article;
}

function updateMessageElement(message: ConversationMessage): void {
  const article = transcript.querySelector<HTMLElement>(
    `[data-message-id="${message.id}"]`
  );
  if (article == null) {
    renderConversation();
    return;
  }
  article.className = `message ${message.role} ${message.status}`;
  const content = article.querySelector<HTMLElement>('.message-content');
  const metrics = article.querySelector<HTMLElement>('.message-metrics');
  if (content != null) {
    content.textContent = message.text;
  }
  if (metrics != null) {
    metrics.textContent = messageMetadata(message);
    metrics.hidden = metrics.textContent.length === 0;
  }
  scrollConversation();
}

function messageMetadata(message: ConversationMessage): string {
  if (message.note != null) {
    return message.note;
  }
  return message.stats == null ? '' : formatRequestStats(message.stats);
}

async function attachSelectedImage(): Promise<void> {
  const file = imageInput.files?.[0];
  if (file == null) {
    return;
  }
  clearPendingImage();
  pendingImage = {
    bytes: new Uint8Array(await file.arrayBuffer()),
    name: file.name,
    objectUrl: URL.createObjectURL(file),
  };
  renderAttachment();
}

function clearPendingImage(): void {
  if (pendingImage != null) {
    URL.revokeObjectURL(pendingImage.objectUrl);
  }
  pendingImage = null;
  imageInput.value = '';
  renderAttachment();
}

function renderAttachment(): void {
  attachment.hidden = pendingImage == null;
  if (pendingImage == null) {
    attachmentImage.removeAttribute('src');
    attachmentName.textContent = '';
    return;
  }
  attachmentImage.src = pendingImage.objectUrl;
  attachmentName.textContent = pendingImage.name;
}

function syncComposerState(): void {
  const ready = client != null && loadedModel != null;
  const running = activeRun != null;
  promptInput.disabled = !ready;
  promptInput.placeholder = ready
    ? 'Message the model...'
    : 'Load a model to start chatting';
  sendButton.hidden = running;
  sendButton.disabled = !ready || running;
  stopButton.hidden = !running;
  imageButton.disabled = !ready || running;
  modelButton.disabled = running;
  newChatButton.disabled = !ready;
  settingsButton.disabled = !ready || running;
}

function updateSettings(): void {
  settings = {
    maxTokens: Math.round(clampNumber(Number(maxTokensInput.value), 16, 2048)),
    temperature: clampNumber(Number(temperatureInput.value), 0, 2),
    topP: clampNumber(Number(topPInput.value), 0.05, 1),
  };
  syncSettingsControls();
}

function syncSettingsControls(): void {
  maxTokensInput.value = String(settings.maxTokens);
  temperatureInput.value = String(settings.temperature);
  temperatureValue.value = settings.temperature.toFixed(2);
  topPInput.value = String(settings.topP);
  topPValue.value = settings.topP.toFixed(2);
}

function clampNumber(value: number, minimum: number, maximum: number): number {
  if (!Number.isFinite(value)) {
    return minimum;
  }
  return Math.min(maximum, Math.max(minimum, value));
}

function setModelStatus(
  state: 'idle' | 'loading' | 'ready' | 'running' | 'error',
  text: string
): void {
  statusDot.className = `status-dot ${state}`;
  statusText.textContent = text;
}

function resetLoadProgress(): void {
  loadProgress.hidden = true;
  loadProgress.value = 0;
  loadStatus.className = 'dialog-status';
}

function showLoadError(error: unknown): void {
  loadProgress.hidden = true;
  loadStatus.className = 'dialog-status error';
  loadStatus.textContent = errorMessage(error);
}

function showComposerError(message: string): void {
  const error = document.createElement('div');
  error.className = 'composer-error';
  error.textContent = message;
  composer.prepend(error);
  window.setTimeout(() => error.remove(), 4000);
}

function errorMessage(error: unknown): string {
  if (error instanceof QueryError) {
    return `${error.code}: ${error.message}`;
  }
  return error instanceof Error ? error.message : String(error);
}

function formatLoadPhase(phase: string): string {
  switch (phase) {
    case 'metadata':
      return 'Checking';
    case 'download':
      return 'Downloading';
    case 'split':
      return 'Preparing';
    case 'store':
      return 'Storing';
    case 'load':
      return 'Loading';
    default:
      return 'Preparing';
  }
}

function resizeComposer(): void {
  promptInput.style.height = 'auto';
  promptInput.style.height = `${Math.min(promptInput.scrollHeight, 160)}px`;
}

function scrollConversation(): void {
  requestAnimationFrame(() => {
    transcript.scrollTop = transcript.scrollHeight;
  });
}

function createSessionId(): string {
  return typeof crypto.randomUUID === 'function'
    ? `demos:chat:${crypto.randomUUID()}`
    : `demos:chat:${Date.now()}`;
}

function createMessageId(): string {
  return typeof crypto.randomUUID === 'function'
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}
