import { CogentEngine } from 'cogentlm';
import { basicChatExample } from './examples/basic-chat';
import { multimodalExample } from './examples/multimodal';
import { structuredOutputExample } from './examples/structured-output';
import { observabilityExample } from './examples/observability';
import { Example, ExampleContext } from './examples/base-example';

// State
let engine: CogentEngine | null = null;
let currentExample: Example = basicChatExample;
let currentMedia: Uint8Array[] | undefined = undefined;

const examples: Record<string, Example> = {
  '01-basic-chat': basicChatExample,
  '02-multimodal': multimodalExample,
  '03-structured-output': structuredOutputExample,
  '04-observability': observabilityExample,
};

// DOM Elements
const consoleOutput = document.getElementById('console-output')!;
const modelUrlInput = document.getElementById('model-url') as HTMLInputElement;
const projectorUrlInput = document.getElementById('projector-url') as HTMLInputElement;
const projectorGroup = document.getElementById('projector-group')!;
const initBtn = document.getElementById('init-engine-btn') as HTMLButtonElement;
const userInput = document.getElementById('user-input') as HTMLInputElement;
const sendBtn = document.getElementById('send-btn') as HTMLButtonElement;
const uploadBtn = document.getElementById('upload-btn') as HTMLButtonElement;
const fileInput = document.getElementById('file-input') as HTMLInputElement;
const imagePreviewArea = document.getElementById('image-preview-area')!;
const imagePreview = document.getElementById('image-preview') as HTMLImageElement;
const removeImageBtn = document.getElementById('remove-image-btn')!;
const clearBtn = document.getElementById('clear-console-btn')!;
const statusDot = document.querySelector('.status-dot')!;
const statusText = document.querySelector('.status-text')!;
const exampleTitle = document.getElementById('example-title')!;
const exampleDescription = document.getElementById('example-description')!;
const navItems = document.querySelectorAll('.nav-item');

// Helpers
function log(message: string, type: 'system' | 'user' | 'ai' | 'error' | 'dim' = 'system') {
  const line = document.createElement('div');
  line.className = `console-line ${type}`;
  line.innerText = message;
  consoleOutput.appendChild(line);
  consoleOutput.scrollTop = consoleOutput.scrollHeight;
  return line;
}

function updateStatus(state: 'disconnected' | 'loading' | 'connected', text: string) {
  statusDot.className = `status-dot ${state}`;
  statusText.textContent = text;
}

async function fileToUint8Array(file: File): Promise<Uint8Array> {
  const buffer = await file.arrayBuffer();
  return new Uint8Array(buffer);
}

// Logic
async function initEngine() {
  const modelUrl = modelUrlInput.value.trim();
  const projectorUrl = projectorUrlInput.value.trim();

  if (!modelUrl) {
    log('Please provide a valid model URL.', 'error');
    return;
  }

  try {
    initBtn.disabled = true;
    userInput.disabled = true;
    sendBtn.disabled = true;
    uploadBtn.disabled = true;
    
    updateStatus('loading', 'Initializing Engine...');
    
    // Only create engine if it doesn't exist
    if (!engine) {
      log('Creating engine instance...', 'system');
      engine = await CogentEngine.create();
    }

    log('Loading assets...', 'system');
    updateStatus('loading', 'Loading Assets...');

    const loadOptions = {
      onProgress: (p: any) => {
        if (p.percent !== null) {
          const pct = Math.round(p.percent);
          updateStatus('loading', `Loading: ${pct}%`);
        }
      }
    };

    // If it's a vision example and we have a projector URL, load both
    if (currentExample.id === '02-multimodal' && projectorUrl) {
      log(`Loading multimodal pair...`, 'dim');
      await engine.models.load({
        model: modelUrl,
        projector: projectorUrl
      }, loadOptions);
    } else {
      log(`Loading text model...`, 'dim');
      await engine.models.load(modelUrl, loadOptions);
    }

    log('Assets loaded successfully!', 'system');
    updateStatus('connected', 'Engine Ready');

    userInput.disabled = false;
    sendBtn.disabled = false;
    uploadBtn.disabled = false;
    initBtn.disabled = false; // Re-enable for reloading

    await currentExample.run({ engine, log, userInput: '' });

  } catch (err) {
    log(`Initialization failed: ${err}`, 'error');
    updateStatus('disconnected', 'Engine Error');
    initBtn.disabled = false;
    userInput.disabled = false; // Let them try again
    sendBtn.disabled = false;
  }
}

async function handleSend() {
  const text = userInput.value.trim();
  if (!text || !engine) return;

  userInput.value = '';
  const media = currentMedia;

  // Clear image after sending
  if (currentMedia) {
    currentMedia = undefined;
    imagePreviewArea.classList.add('hidden');
    fileInput.value = '';
  }

  if (currentExample.onUserInput) {
    await currentExample.onUserInput({ engine, log, userInput: text, media });
  }
}

// Event Listeners
initBtn.addEventListener('click', initEngine);
sendBtn.addEventListener('click', handleSend);
userInput.addEventListener('keypress', (e) => {
  if (e.key === 'Enter') handleSend();
});

uploadBtn.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', async (e) => {
  const file = (e.target as HTMLInputElement).files?.[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = (e) => {
    imagePreview.src = e.target?.result as string;
    imagePreviewArea.classList.remove('hidden');
  };
  reader.readAsDataURL(file);

  currentMedia = [await fileToUint8Array(file)];
  log(`Image attached: ${file.name}`, 'dim');
});

removeImageBtn.addEventListener('click', () => {
  currentMedia = undefined;
  imagePreviewArea.classList.add('hidden');
  fileInput.value = '';
});

clearBtn.addEventListener('click', () => {
  consoleOutput.innerHTML = '';
});

navItems.forEach(item => {
  item.addEventListener('click', () => {
    const id = item.getAttribute('data-example')!;
    if (!examples[id]) return;

    navItems.forEach(n => n.classList.remove('active'));
    item.classList.add('active');

    currentExample = examples[id];
    exampleTitle.textContent = currentExample.title;
    exampleDescription.textContent = currentExample.description;

    // Toggle projector URL visibility
    if (id === '02-multimodal') {
      projectorGroup.classList.remove('hidden');
      // Set Liquid LFM 2.5 defaults
      modelUrlInput.value = 'https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF/resolve/main/LFM2.5-VL-450M-F16.gguf';
      projectorUrlInput.value = 'https://huggingface.co/LiquidAI/LFM2.5-VL-450M-GGUF/resolve/main/mmproj-LFM2.5-VL-450m-F16.gguf';
    } else {
      projectorGroup.classList.add('hidden');
      if (modelUrlInput.value.includes('LiquidAI')) {
        modelUrlInput.value = 'https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf';
      }
    }

    log(`Switched to ${currentExample.title} example.`, 'system');

    if (engine) {
      currentExample.run({ engine, log, userInput: '' });
    }
  });
});
