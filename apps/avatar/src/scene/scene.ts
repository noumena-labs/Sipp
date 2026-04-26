//////////////////////////////////////////////////////////////////////////////
//
// scene.ts
//
// - Owns the three.js renderer/scene/camera for the avatar canvas.
// - Exposes a small imperative API so the React component can drive size
//   changes and inject/replace the avatar root without re-creating the
//   renderer on every render.
// - Uses a clock-driven animation loop; the binding layer plugs into the
//   per-frame tick to update poses and expressions.
//
//////////////////////////////////////////////////////////////////////////////

import * as THREE from 'three';
import type { AvatarLayout } from './vrm-loader';

export interface SceneHandle {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene: THREE.Scene;
  readonly camera: THREE.PerspectiveCamera;
  readonly avatarRoot: THREE.Group;
  /** Register a function to be invoked each frame with delta-time (seconds). */
  onFrame(callback: (deltaSeconds: number) => void): () => void;
  focusAvatar(layout: AvatarLayout): void;
  setSize(width: number, height: number): void;
  dispose(): void;
}

const DEFAULT_FOCUS = new THREE.Vector3(0, 1.3, 0);
const MIN_CAMERA_DISTANCE = 1.7;
const CAMERA_PADDING = 1.35;
const CAMERA_VERTICAL_BIAS = 0.18;

interface FantasyEnvironment {
  update(deltaSeconds: number, elapsedSeconds: number): void;
}

export function createScene(container: HTMLElement): SceneHandle {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();

  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 20);
  camera.position.set(0, DEFAULT_FOCUS.y + 0.05, 2.4);
  camera.lookAt(DEFAULT_FOCUS);

  scene.fog = new THREE.FogExp2(0x080716, 0.058);

  // Moonlit fantasy stage lighting: cool ambient, warm key, violet rim.
  const hemi = new THREE.HemisphereLight(0xdfeaff, 0x12091f, 0.72);
  scene.add(hemi);
  const dir = new THREE.DirectionalLight(0xffe1a8, 1.05);
  dir.position.set(-2.4, 4, 2.2);
  scene.add(dir);
  const rim = new THREE.DirectionalLight(0x9574ff, 1.4);
  rim.position.set(2.8, 2.4, -1.6);
  scene.add(rim);

  const environment = createFantasyEnvironment(scene);

  const avatarRoot = new THREE.Group();
  scene.add(avatarRoot);

  const callbacks = new Set<(deltaSeconds: number) => void>();
  const clock = new THREE.Clock();
  let elapsedSeconds = 0;
  const currentFocus = DEFAULT_FOCUS.clone();
  let lastLayout: AvatarLayout | null = null;
  let disposed = false;

  const tick = (): void => {
    if (disposed) {
      return;
    }
    const delta = clock.getDelta();
    elapsedSeconds += delta;
    environment.update(delta, elapsedSeconds);
    for (const cb of callbacks) {
      cb(delta);
    }
    renderer.render(scene, camera);
    renderer.setAnimationLoop(tick);
  };
  renderer.setAnimationLoop(tick);

  const setSize = (width: number, height: number): void => {
    const safeW = Math.max(1, Math.floor(width));
    const safeH = Math.max(1, Math.floor(height));
    renderer.setSize(safeW, safeH, false);
    camera.aspect = safeW / safeH;
    camera.updateProjectionMatrix();
    if (lastLayout) {
      applyCameraLayout(lastLayout);
    }
  };

  const applyCameraLayout = (layout: AvatarLayout): void => {
    lastLayout = layout;
    const focus = layout.focusPoint;
    const verticalFov = THREE.MathUtils.degToRad(camera.fov);
    const horizontalFov = 2 * Math.atan(Math.tan(verticalFov / 2) * camera.aspect);
    const distanceForHeight = layout.verticalExtent / Math.tan(verticalFov / 2);
    const distanceForWidth = layout.horizontalExtent / Math.tan(horizontalFov / 2);
    const distance = Math.min(
      6,
      Math.max(MIN_CAMERA_DISTANCE, Math.max(distanceForHeight, distanceForWidth) * CAMERA_PADDING)
    );
    currentFocus.copy(focus);
    currentFocus.y += layout.height * CAMERA_VERTICAL_BIAS;
    camera.position.set(0, focus.y + layout.height * (0.03 + CAMERA_VERTICAL_BIAS), distance);
    camera.lookAt(currentFocus);
    camera.near = 0.1;
    camera.far = Math.max(20, distance + layout.height * 4);
    camera.updateProjectionMatrix();
  };

  return {
    renderer,
    scene,
    camera,
    avatarRoot,
    onFrame(callback) {
      callbacks.add(callback);
      return () => callbacks.delete(callback);
    },
    focusAvatar(layout) {
      applyCameraLayout(layout);
    },
    setSize,
    dispose() {
      disposed = true;
      renderer.setAnimationLoop(null);
      callbacks.clear();
      renderer.dispose();
      if (renderer.domElement.parentElement === container) {
        container.removeChild(renderer.domElement);
      }
      scene.traverse((obj) => {
        if ((obj as THREE.Mesh).geometry) {
          (obj as THREE.Mesh).geometry.dispose();
        }
        const mat = (obj as THREE.Mesh).material;
        if (Array.isArray(mat)) {
          mat.forEach((m) => m.dispose());
        } else if (mat) {
          mat.dispose();
        }
      });
    },
  };
}

function createFantasyEnvironment(scene: THREE.Scene): FantasyEnvironment {
  const runeRing = new THREE.Group();
  const moteGeometry = new THREE.BufferGeometry();
  const moteCount = 90;
  const motePositions = new Float32Array(moteCount * 3);
  const moteSeeds = Array.from({ length: moteCount }, () => Math.random());

  const stone = new THREE.MeshStandardMaterial({
    color: 0x40344d,
    roughness: 0.92,
    metalness: 0.02,
  });
  const stoneDark = new THREE.MeshStandardMaterial({
    color: 0x231d32,
    roughness: 0.96,
  });
  const rune = new THREE.MeshBasicMaterial({
    color: 0x8dfce7,
    transparent: true,
    opacity: 0.46,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  const crystal = new THREE.MeshStandardMaterial({
    color: 0x76f7e4,
    emissive: 0x2adbbf,
    emissiveIntensity: 1.4,
    roughness: 0.34,
    metalness: 0.08,
    transparent: true,
    opacity: 0.82,
  });

  const dais = new THREE.Mesh(new THREE.CylinderGeometry(1.12, 1.2, 0.1, 80), stone);
  dais.position.y = -0.052;
  scene.add(dais);

  const daisLip = new THREE.Mesh(new THREE.TorusGeometry(1.14, 0.018, 10, 96), stoneDark);
  daisLip.rotation.x = Math.PI / 2;
  daisLip.position.y = 0.005;
  scene.add(daisLip);

  const innerRune = new THREE.Mesh(new THREE.TorusGeometry(0.84, 0.006, 8, 96), rune);
  const outerRune = new THREE.Mesh(new THREE.TorusGeometry(1.0, 0.008, 8, 96), rune.clone());
  innerRune.rotation.x = Math.PI / 2;
  outerRune.rotation.x = Math.PI / 2;
  innerRune.position.y = 0.008;
  outerRune.position.y = 0.01;
  runeRing.add(innerRune, outerRune);

  const sigilGeometry = new THREE.BoxGeometry(0.018, 0.12, 0.006);
  for (let index = 0; index < 18; index += 1) {
    const angle = (index / 18) * Math.PI * 2;
    const mark = new THREE.Mesh(sigilGeometry, rune.clone());
    mark.position.set(Math.cos(angle) * 0.93, 0.012, Math.sin(angle) * 0.93);
    mark.rotation.set(-Math.PI / 2, 0, -angle);
    runeRing.add(mark);
  }
  scene.add(runeRing);

  addBrokenPillar(scene, -1.55, -0.42, 1.18, stone, stoneDark);
  addBrokenPillar(scene, 1.46, -0.55, 0.88, stone, stoneDark);
  addBrokenPillar(scene, -1.18, -1.12, 0.72, stone, stoneDark);
  addBrokenPillar(scene, 1.34, -1.22, 1.36, stone, stoneDark);

  addCrystalCluster(scene, -1.05, 0.36, 0.44, crystal);
  addCrystalCluster(scene, 1.05, 0.28, 0.36, crystal.clone());
  const crystalLight = new THREE.PointLight(0x63ffe5, 1.6, 4.2);
  crystalLight.position.set(0, 0.46, 0.28);
  scene.add(crystalLight);

  for (let index = 0; index < moteCount; index += 1) {
    const radius = 0.8 + moteSeeds[index] * 2.1;
    const angle = moteSeeds[index] * Math.PI * 2 * 7.3;
    motePositions[index * 3] = Math.cos(angle) * radius;
    motePositions[index * 3 + 1] = 0.45 + Math.random() * 2.25;
    motePositions[index * 3 + 2] = Math.sin(angle) * radius - 0.6;
  }
  moteGeometry.setAttribute('position', new THREE.BufferAttribute(motePositions, 3));
  const motes = new THREE.Points(
    moteGeometry,
    new THREE.PointsMaterial({
      color: 0xffe4a3,
      size: 0.025,
      transparent: true,
      opacity: 0.62,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    })
  );
  scene.add(motes);

  return {
    update(_deltaSeconds, elapsedSeconds) {
      runeRing.rotation.y = elapsedSeconds * 0.08;
      innerRune.scale.setScalar(1 + Math.sin(elapsedSeconds * 1.2) * 0.015);
      outerRune.scale.setScalar(1 + Math.cos(elapsedSeconds * 1.1) * 0.012);
      crystalLight.intensity = 1.35 + Math.sin(elapsedSeconds * 2.1) * 0.24;
      const positions = moteGeometry.getAttribute('position') as THREE.BufferAttribute;
      for (let index = 0; index < moteCount; index += 1) {
        const seed = moteSeeds[index];
        positions.setY(index, 0.45 + ((elapsedSeconds * (0.08 + seed * 0.08) + seed * 3.8) % 2.35));
        positions.setX(index, motePositions[index * 3] + Math.sin(elapsedSeconds * 0.7 + seed * 20) * 0.035);
      }
      positions.needsUpdate = true;
    },
  };
}

function addBrokenPillar(
  scene: THREE.Scene,
  x: number,
  z: number,
  height: number,
  stone: THREE.Material,
  capMaterial: THREE.Material
): void {
  const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.13, 0.16, height, 7), stone);
  shaft.position.set(x, height / 2 - 0.04, z);
  shaft.rotation.z = (Math.random() - 0.5) * 0.18;
  scene.add(shaft);

  const cap = new THREE.Mesh(new THREE.CylinderGeometry(0.19, 0.19, 0.08, 7), capMaterial);
  cap.position.set(x + 0.02, height - 0.02, z - 0.015);
  cap.rotation.set(0.1, 0.2, shaft.rotation.z + 0.05);
  scene.add(cap);
}

function addCrystalCluster(
  scene: THREE.Scene,
  x: number,
  z: number,
  scale: number,
  material: THREE.Material
): void {
  const geometry = new THREE.ConeGeometry(0.12, 0.62, 5);
  for (let index = 0; index < 4; index += 1) {
    const shard = new THREE.Mesh(geometry, material);
    const angle = (index / 4) * Math.PI * 2;
    const radius = index === 0 ? 0 : 0.12;
    shard.position.set(x + Math.cos(angle) * radius, 0.25 * scale, z + Math.sin(angle) * radius);
    shard.scale.setScalar(scale * (index === 0 ? 1.15 : 0.74));
    shard.rotation.set((Math.random() - 0.5) * 0.26, angle, (Math.random() - 0.5) * 0.22);
    scene.add(shard);
  }
}
