//////////////////////////////////////////////////////////////////////////////
//
// scene/scene.ts
//
// - Top-down 3D courtyard scene. Creates the three.js renderer/scene/camera
//   with a near-overhead tilted camera so agents and objects read clearly
//   as colored pucks/boxes with floating glyph billboards.
//
//////////////////////////////////////////////////////////////////////////////

import * as THREE from 'three';

export interface SimulationSceneHandle {
  readonly renderer: THREE.WebGLRenderer;
  readonly scene: THREE.Scene;
  readonly camera: THREE.PerspectiveCamera;
  readonly worldRoot: THREE.Group;
  onFrame(callback: (deltaSeconds: number) => void): () => void;
  setSize(width: number, height: number): void;
  dispose(): void;
}

export function createSimulationScene(
  container: HTMLElement,
  halfExtent: number
): SimulationSceneHandle {
  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x1a1f2b);

  // Tilted top-down camera. Positioned high on +Z so north (-Z) is up-screen.
  const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 200);
  const camDistance = halfExtent * 2.2;
  camera.position.set(0, camDistance, camDistance * 0.65);
  camera.lookAt(0, 0, 0);

  const hemi = new THREE.HemisphereLight(0xffffff, 0x223344, 0.9);
  scene.add(hemi);
  const dir = new THREE.DirectionalLight(0xffffff, 1.0);
  dir.position.set(5, 10, 5);
  scene.add(dir);

  // Ground plane. `halfExtent` is the maximum integer cell center, so the
  // visible grid extends half a cell past it on each side.
  const visualHalfExtent = halfExtent + 0.5;
  const groundSize = visualHalfExtent * 2;
  const ground = new THREE.Mesh(
    new THREE.PlaneGeometry(groundSize, groundSize),
    new THREE.MeshStandardMaterial({ color: 0x2a3142, roughness: 0.95, metalness: 0.0 })
  );
  ground.rotation.x = -Math.PI / 2;
  scene.add(ground);

  // Grid lines mark cell edges, placing integer world coordinates at cell centers.
  const grid = createCellCenteredGrid(visualHalfExtent);
  scene.add(grid);

  // Bounds outline.
  const outlineGeom = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-visualHalfExtent, 0.004, -visualHalfExtent),
    new THREE.Vector3(visualHalfExtent, 0.004, -visualHalfExtent),
    new THREE.Vector3(visualHalfExtent, 0.004, visualHalfExtent),
    new THREE.Vector3(-visualHalfExtent, 0.004, visualHalfExtent),
    new THREE.Vector3(-visualHalfExtent, 0.004, -visualHalfExtent),
  ]);
  const outline = new THREE.Line(
    outlineGeom,
    new THREE.LineBasicMaterial({ color: 0x6a7690 })
  );
  scene.add(outline);

  const worldRoot = new THREE.Group();
  scene.add(worldRoot);

  const callbacks = new Set<(deltaSeconds: number) => void>();
  const clock = new THREE.Clock();
  let disposed = false;

  const tick = (): void => {
    if (disposed) return;
    const delta = clock.getDelta();
    for (const cb of callbacks) cb(delta);
    renderer.render(scene, camera);
  };
  renderer.setAnimationLoop(tick);

  const setSize = (width: number, height: number): void => {
    const w = Math.max(1, Math.floor(width));
    const h = Math.max(1, Math.floor(height));
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  };

  return {
    renderer,
    scene,
    camera,
    worldRoot,
    onFrame(cb) {
      callbacks.add(cb);
      return () => callbacks.delete(cb);
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
        const mesh = obj as THREE.Mesh;
        mesh.geometry?.dispose?.();
        const mat = mesh.material;
        if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
        else if (mat) (mat as THREE.Material).dispose();
      });
    },
  };
}

function createCellCenteredGrid(visualHalfExtent: number): THREE.LineSegments {
  const positions: number[] = [];
  const min = -visualHalfExtent;
  const max = visualHalfExtent;

  for (let n = min; n <= max + 1e-6; n += 1) {
    positions.push(min, 0.002, n, max, 0.002, n);
    positions.push(n, 0.002, min, n, 0.002, max);
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  const material = new THREE.LineBasicMaterial({
    color: 0x2f364a,
    transparent: true,
    opacity: 0.6,
  });
  return new THREE.LineSegments(geometry, material);
}
