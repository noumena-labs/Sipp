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
  setActive(active: boolean): void;
  dispose(): void;
}

interface MeshPart {
  readonly mesh: THREE.Mesh;
  readonly baseY: number;
  readonly baseScale: { x: number; y: number; z: number };
}

interface ShapeSpec {
  readonly parts: readonly MeshPartSpec[];
  readonly hoverColor: number;
}

interface MeshPartSpec {
  readonly geometry: THREE.BufferGeometry;
  readonly color: number;
  readonly y: number;
  readonly rotationX?: number;
  readonly rotationZ?: number;
  readonly offsetX?: number;
  readonly offsetZ?: number;
  readonly scaleX?: number;
  readonly scaleY?: number;
  readonly scaleZ?: number;
}

function specFor(kind: string): ShapeSpec {
  switch (kind) {
    case 'banana':
      return {
        hoverColor: 0xffe066,
        parts: [
          { geometry: new THREE.SphereGeometry(0.18, 16, 12), color: 0xffe066, y: 0.18, scaleX: 1.25, scaleZ: 0.82 },
        ],
      };
    case 'goal':
      return {
        hoverColor: 0x7bd88f,
        parts: [
          { geometry: new THREE.CylinderGeometry(1.15, 1.15, 0.05, 32), color: 0x7bd88f, y: 0.025 },
        ],
      };
    case 'bat':
      return {
        hoverColor: 0xffc86a,
        parts: [
          { geometry: new THREE.CylinderGeometry(0.05, 0.05, 0.7, 12), color: 0xc58b4e, y: 0.34, rotationZ: Math.PI / 3 },
          { geometry: new THREE.CylinderGeometry(0.12, 0.08, 0.36, 16), color: 0xe8ba74, y: 0.56, offsetX: 0.16, rotationZ: Math.PI / 3 },
        ],
      };
    case 'ice_cube':
      return {
        hoverColor: 0x8fe7ff,
        parts: [
          { geometry: new THREE.BoxGeometry(0.34, 0.34, 0.34), color: 0x8fe7ff, y: 0.18, rotationX: 0.25, rotationZ: -0.18 },
        ],
      };
    case 'crate':
      return {
        hoverColor: 0x9b6a3d,
        parts: [
          { geometry: new THREE.BoxGeometry(0.9, 0.75, 0.9), color: 0x9b6a3d, y: 0.375 },
        ],
      };
    case 'rock':
      return {
        hoverColor: 0x6f7482,
        parts: [
          { geometry: new THREE.DodecahedronGeometry(0.55, 0), color: 0x6f7482, y: 0.35 },
        ],
      };
    case 'bench':
      return {
        hoverColor: 0x8a6d3b,
        parts: [
          { geometry: new THREE.BoxGeometry(1.4, 0.25, 0.5), color: 0x8a6d3b, y: 0.125 },
        ],
      };
    case 'fountain':
      return {
        hoverColor: 0x6bb7c9,
        parts: [
          { geometry: new THREE.CylinderGeometry(0.6, 0.7, 0.4, 24), color: 0x6bb7c9, y: 0.2 },
        ],
      };
    case 'plant':
      return {
        hoverColor: 0x4c956c,
        parts: [
          { geometry: new THREE.ConeGeometry(0.3, 0.8, 12), color: 0x4c956c, y: 0.4 },
        ],
      };
    default:
      return {
        hoverColor: 0x888888,
        parts: [
          { geometry: new THREE.BoxGeometry(0.3, 0.3, 0.3), color: 0x888888, y: 0.15 },
        ],
      };
  }
}

export function createObjectVisual(kind: string): ObjectVisual {
  const spec = specFor(kind);
  const root = new THREE.Group();
  const parts = spec.parts.map((part) => createPart(part, root));

  let heldBy: string | null = null;
  let hovered = false;
  let active = true;

  const applyVisualState = (): void => {
    const carryScale = heldBy ? 1.4 : 1;
    const hoverScale = hovered ? 1.1 : 1;
    const activeOpacity = active ? 1 : 0;
    for (const part of parts) {
      part.mesh.position.y = heldBy ? part.baseY + 0.8 : part.baseY;
      part.mesh.scale.set(
        part.baseScale.x * carryScale * hoverScale,
        part.baseScale.y * carryScale * hoverScale,
        part.baseScale.z * carryScale * hoverScale
      );
      const material = part.mesh.material as THREE.MeshStandardMaterial;
      material.transparent = activeOpacity < 1;
      material.opacity = activeOpacity;
      material.emissive.copy(hovered ? new THREE.Color(spec.hoverColor).multiplyScalar(0.28) : new THREE.Color(0x000000));
    }
    root.visible = active;
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
    setActive(nextActive) {
      active = nextActive;
      applyVisualState();
    },
    dispose() {
      for (const part of parts) {
        part.mesh.geometry.dispose();
        (part.mesh.material as THREE.Material).dispose();
      }
    },
  };
}

function createPart(spec: MeshPartSpec, root: THREE.Group): MeshPart {
  const material = new THREE.MeshStandardMaterial({
    color: spec.color,
    roughness: 0.6,
    metalness: 0.05,
  });
  const mesh = new THREE.Mesh(spec.geometry, material);
  mesh.position.set(spec.offsetX ?? 0, spec.y, spec.offsetZ ?? 0);
  mesh.rotation.x = spec.rotationX ?? 0;
  mesh.rotation.z = spec.rotationZ ?? 0;
  const baseScale = { x: spec.scaleX ?? 1, y: spec.scaleY ?? 1, z: spec.scaleZ ?? 1 };
  mesh.scale.set(baseScale.x, baseScale.y, baseScale.z);
  root.add(mesh);
  return { mesh, baseY: spec.y, baseScale };
}
