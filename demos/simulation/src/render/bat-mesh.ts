//////////////////////////////////////////////////////////////////////////////
//
// render/bat-mesh.ts
//
// - Shared bat mesh used by both the world object and the held prop so the
//   silhouette stays consistent. The bat is modeled along local +Y and callers
//   rotate/place the root for their context.
//
//////////////////////////////////////////////////////////////////////////////

import * as THREE from 'three';

type BatPartMesh = THREE.Mesh<THREE.BufferGeometry, THREE.MeshStandardMaterial>;

export interface BatMesh {
  readonly root: THREE.Group;
  readonly meshes: readonly BatPartMesh[];
  dispose(): void;
}

interface BatPartSpec {
  readonly createGeometry: () => THREE.BufferGeometry;
  readonly color: number;
  readonly roughness: number;
  readonly metalness: number;
  readonly y: number;
}

const BAT_PART_SPECS = [
  {
    createGeometry: () => new THREE.SphereGeometry(0.045, 14, 12),
    color: 0xaf7341,
    roughness: 0.64,
    metalness: 0.03,
    y: -0.48,
  },
  {
    createGeometry: () => new THREE.CylinderGeometry(0.034, 0.038, 0.43, 14),
    color: 0xc58b4e,
    roughness: 0.62,
    metalness: 0.04,
    y: -0.215,
  },
  {
    createGeometry: () => new THREE.CylinderGeometry(0.062, 0.038, 0.16, 16),
    color: 0xd9a86b,
    roughness: 0.58,
    metalness: 0.04,
    y: 0.08,
  },
  {
    createGeometry: () => new THREE.CylinderGeometry(0.114, 0.062, 0.4, 20),
    color: 0xe8ba74,
    roughness: 0.55,
    metalness: 0.05,
    y: 0.36,
  },
] as const satisfies readonly BatPartSpec[];

export function createBatMesh(): BatMesh {
  const root = new THREE.Group();
  const meshes: BatPartMesh[] = [];

  for (const spec of BAT_PART_SPECS) {
    const mesh = new THREE.Mesh(
      spec.createGeometry(),
      new THREE.MeshStandardMaterial({
        color: spec.color,
        roughness: spec.roughness,
        metalness: spec.metalness,
      })
    );
    mesh.position.y = spec.y;
    root.add(mesh);
    meshes.push(mesh);
  }

  return {
    root,
    meshes,
    dispose() {
      for (const mesh of meshes) {
        mesh.geometry.dispose();
        mesh.material.dispose();
      }
    },
  };
}
