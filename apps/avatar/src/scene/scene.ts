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

export function createScene(container: HTMLElement): SceneHandle {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();

  const camera = new THREE.PerspectiveCamera(30, 1, 0.1, 20);
  camera.position.set(0, DEFAULT_FOCUS.y + 0.05, 2.4);
  camera.lookAt(DEFAULT_FOCUS);

  // Lighting — a hemisphere for fill and a directional for shape.
  const hemi = new THREE.HemisphereLight(0xffffff, 0x223344, 0.9);
  scene.add(hemi);
  const dir = new THREE.DirectionalLight(0xffffff, 1.1);
  dir.position.set(2, 3, 2);
  scene.add(dir);

  // Ground shadow disc — cheap visual anchor without enabling shadow maps.
  const disc = new THREE.Mesh(
    new THREE.CircleGeometry(0.8, 32),
    new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.25 })
  );
  disc.rotation.x = -Math.PI / 2;
  disc.position.y = 0.001;
  scene.add(disc);

  const avatarRoot = new THREE.Group();
  scene.add(avatarRoot);

  const callbacks = new Set<(deltaSeconds: number) => void>();
  const clock = new THREE.Clock();
  const currentFocus = DEFAULT_FOCUS.clone();
  let lastLayout: AvatarLayout | null = null;
  let disposed = false;

  const tick = (): void => {
    if (disposed) {
      return;
    }
    const delta = clock.getDelta();
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
    camera.position.set(0, focus.y + layout.height * 0.03, distance);
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
