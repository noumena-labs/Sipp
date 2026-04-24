//////////////////////////////////////////////////////////////////////////////
//
// render/object-mesh.ts
//
// - Simple geometric mesh per object kind. A held object is moved by the
//   world binding to track its owning agent; here we just define shape
//   and color.
//
//////////////////////////////////////////////////////////////////////////////

import * as THREE from 'three';

export interface ObjectVisual {
  readonly root: THREE.Group;
  setPosition(x: number, z: number): void;
  setHeldBy(heldBy: string | null): void;
  setHovered(hovered: boolean): void;
  dispose(): void;
}

interface ShapeSpec {
  readonly geometry: THREE.BufferGeometry;
  readonly color: number;
  readonly y: number;
}

function specFor(kind: string): ShapeSpec {
  switch (kind) {
    case 'banana':
      return { geometry: new THREE.SphereGeometry(0.18, 16, 12), color: 0xffe066, y: 0.18 };
    case 'goal':
      return { geometry: new THREE.CylinderGeometry(1.15, 1.15, 0.05, 32), color: 0x7bd88f, y: 0.025 };
    case 'crate':
      return { geometry: new THREE.BoxGeometry(0.9, 0.75, 0.9), color: 0x9b6a3d, y: 0.375 };
    case 'rock':
      return { geometry: new THREE.DodecahedronGeometry(0.55, 0), color: 0x6f7482, y: 0.35 };
    case 'bench':
      return { geometry: new THREE.BoxGeometry(1.4, 0.25, 0.5), color: 0x8a6d3b, y: 0.125 };
    case 'fountain':
      return { geometry: new THREE.CylinderGeometry(0.6, 0.7, 0.4, 24), color: 0x6bb7c9, y: 0.2 };
    case 'plant':
      return { geometry: new THREE.ConeGeometry(0.3, 0.8, 12), color: 0x4c956c, y: 0.4 };
    default:
      return { geometry: new THREE.BoxGeometry(0.3, 0.3, 0.3), color: 0x888888, y: 0.15 };
  }
}

export function createObjectVisual(kind: string): ObjectVisual {
  const spec = specFor(kind);
  const material = new THREE.MeshStandardMaterial({
    color: spec.color,
    roughness: 0.6,
    metalness: 0.05,
  });
  const hoverEmissive = new THREE.Color(spec.color).multiplyScalar(0.28);
  const mesh = new THREE.Mesh(spec.geometry, material);
  mesh.position.y = spec.y;

  const root = new THREE.Group();
  root.add(mesh);

  let heldBy: string | null = null;
  let hovered = false;

  const applyVisualState = (): void => {
    mesh.position.y = heldBy ? spec.y + 0.8 : spec.y;
    const carryScale = heldBy ? 1.45 : 1;
    const hoverScale = hovered ? 1.1 : 1;
    mesh.scale.setScalar(carryScale * hoverScale);
    material.emissive.copy(hovered ? hoverEmissive : new THREE.Color(0x000000));
  };

  applyVisualState();

  return {
    root,
    setPosition(x, z) {
      root.position.set(x, 0, z);
    },
    setHeldBy(nextHeldBy) {
      heldBy = nextHeldBy;
      applyVisualState();
    },
    setHovered(nextHovered) {
      hovered = nextHovered;
      applyVisualState();
    },
    dispose() {
      spec.geometry.dispose();
      material.dispose();
    },
  };
}
