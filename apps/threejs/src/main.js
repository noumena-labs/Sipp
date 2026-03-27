import * as THREE from 'three';
import { CogentEngine, getBundledRuntimeUrls } from 'cogent-engine';
import './style.css';

const app = document.querySelector('#app');
app.innerHTML = `
  <div class="panel">
    <h1>CogentEngine + Three.js</h1>
    <div class="row">
      <label>Model URL (optional if using local file)</label>
      <input id="modelUrl" placeholder="https://.../model.gguf" />
    </div>
    <div class="row">
      <label>Local GGUF File</label>
      <input id="modelFile" type="file" accept=".gguf" />
    </div>
    <div class="row">
      <button id="initRuntimeBtn">1) Init WASM Runtime</button>
    </div>
    <div class="row">
      <button id="loadModelBtn">2) Load Model + Init Engine</button>
    </div>
    <div class="row">
      <label>Prompt</label>
      <textarea id="promptText">Describe what this glowing object is seeing.</textarea>
    </div>
    <div class="row">
      <label>Max Tokens</label>
      <input id="tokenCount" type="number" min="1" max="512" value="64" />
    </div>
    <div class="row">
      <button id="runPromptBtn">3) Run Inference</button>
    </div>
    <p id="status" class="status">Status: idle</p>
    <div id="response" class="response"></div>
  </div>
`;

const engine = new CogentEngine(getBundledRuntimeUrls());
let runtimeReady = false;
let engineReady = false;
let sceneEnergyTarget = 0.45;
let sceneEnergy = sceneEnergyTarget;
let animationFrameId = 0;
let isDisposed = false;

const statusEl = document.querySelector('#status');
const responseEl = document.querySelector('#response');
const modelUrlInput = document.querySelector('#modelUrl');
const modelFileInput = document.querySelector('#modelFile');
const promptTextEl = document.querySelector('#promptText');
const tokenCountEl = document.querySelector('#tokenCount');
const initRuntimeBtn = document.querySelector('#initRuntimeBtn');
const loadModelBtn = document.querySelector('#loadModelBtn');
const runPromptBtn = document.querySelector('#runPromptBtn');

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
app.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color('#050814');
scene.fog = new THREE.Fog('#050814', 6, 14);

const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 0.6, 4.2);

const hemiLight = new THREE.HemisphereLight('#9dd7ff', '#091320', 1.2);
scene.add(hemiLight);

const keyLight = new THREE.DirectionalLight('#52b8ff', 1.8);
keyLight.position.set(3, 2, 2);
scene.add(keyLight);

const knot = new THREE.Mesh(
  new THREE.TorusKnotGeometry(0.72, 0.24, 220, 32),
  new THREE.MeshStandardMaterial({
    color: '#1e7cff',
    emissive: '#16306b',
    emissiveIntensity: 0.6,
    metalness: 0.25,
    roughness: 0.22
  })
);
scene.add(knot);

const shell = new THREE.Mesh(
  new THREE.IcosahedronGeometry(1.75, 3),
  new THREE.MeshBasicMaterial({
    color: '#1f55a2',
    transparent: true,
    opacity: 0.14,
    wireframe: true
  })
);
scene.add(shell);

function setStatus(message) {
  statusEl.textContent = `Status: ${message}`;
}

function errorMessage(error) {
  return error instanceof Error ? error.message : String(error);
}

function setBusy(isBusy) {
  initRuntimeBtn.disabled = isBusy;
  loadModelBtn.disabled = isBusy;
  runPromptBtn.disabled = isBusy;
}

function applyResponseColor(text) {
  let hash = 0;
  for (let i = 0; i < text.length; i += 1) {
    hash = (hash * 31 + text.charCodeAt(i)) >>> 0;
  }
  const hue = hash % 360;
  knot.material.color.setHSL(hue / 360, 0.8, 0.55);
  knot.material.emissive.setHSL(hue / 360, 0.7, 0.22);
}

initRuntimeBtn.addEventListener('click', async () => {
  setBusy(true);
  setStatus('initializing wasm runtime...');
  try {
    await engine.initModule();
    runtimeReady = true;
    setStatus('wasm runtime ready');
  } catch (error) {
    setStatus(`runtime init failed: ${errorMessage(error)}`);
  } finally {
    setBusy(false);
  }
});

loadModelBtn.addEventListener('click', async () => {
  setBusy(true);
  setStatus('loading model...');
  try {
    if (!runtimeReady) {
      await engine.initModule();
      runtimeReady = true;
    }

    const localFile = modelFileInput.files?.[0];
    let modelPath;

    if (localFile) {
      modelPath = await engine.loadModelFromFile(localFile, 'active-model.gguf', (pct) => {
        setStatus(`reading local model... ${pct}%`);
      });
      setStatus(`loaded local model: ${localFile.name}`);
    } else {
      const modelUrl = modelUrlInput.value.trim();
      if (!modelUrl) {
        throw new Error('Choose a local file or provide a model URL.');
      }
      modelPath = await engine.loadModelFromUrl(modelUrl, 'active-model.gguf', (pct) => {
        setStatus(`downloading model... ${pct}%`);
      });
    }

    await engine.initEngine(modelPath);
    engineReady = true;
    setStatus('engine initialized');
  } catch (error) {
    setStatus(`model init failed: ${errorMessage(error)}`);
  } finally {
    setBusy(false);
  }
});

runPromptBtn.addEventListener('click', async () => {
  setBusy(true);
  setStatus('running prompt...');
  try {
    if (!engineReady) {
      throw new Error('Engine is not initialized yet.');
    }

    const prompt = promptTextEl.value.trim();
    if (!prompt) {
      throw new Error('Prompt cannot be empty.');
    }

    const parsedTokenCount = Number.parseInt(tokenCountEl.value, 10);
    const tokenCount = Number.isFinite(parsedTokenCount)
      ? Math.min(512, Math.max(1, parsedTokenCount))
      : 64;
    tokenCountEl.value = String(tokenCount);

    const response = await engine.prompt('three-demo', prompt, tokenCount);
    responseEl.textContent = response;

    sceneEnergyTarget = Math.min(2.2, Math.max(0.55, response.length / 140));
    applyResponseColor(response);
    setStatus('inference complete');
  } catch (error) {
    setStatus(`inference failed: ${errorMessage(error)}`);
  } finally {
    setBusy(false);
  }
});

function handleResize() {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
}

window.addEventListener('resize', handleResize);

function disposeMaterial(material) {
  if (Array.isArray(material)) {
    material.forEach((entry) => entry.dispose());
    return;
  }
  material.dispose();
}

function disposeDemo() {
  if (isDisposed) {
    return;
  }
  isDisposed = true;

  window.removeEventListener('resize', handleResize);
  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = 0;
  }

  knot.geometry.dispose();
  disposeMaterial(knot.material);
  shell.geometry.dispose();
  disposeMaterial(shell.material);
  renderer.dispose();
  engine.close();
}

window.addEventListener('beforeunload', disposeDemo, { once: true });
if (import.meta.hot) {
  import.meta.hot.dispose(disposeDemo);
}

function animate() {
  sceneEnergy += (sceneEnergyTarget - sceneEnergy) * 0.03;
  knot.rotation.x += 0.0035 * sceneEnergy;
  knot.rotation.y += 0.0056 * sceneEnergy;
  shell.rotation.y -= 0.0024 * sceneEnergy;
  shell.rotation.z += 0.0008 * sceneEnergy;
  knot.material.emissiveIntensity = 0.45 + sceneEnergy * 0.35;

  renderer.render(scene, camera);
  animationFrameId = requestAnimationFrame(animate);
}

animate();
