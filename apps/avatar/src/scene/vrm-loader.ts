//////////////////////////////////////////////////////////////////////////////
//
// vrm-loader.ts
//
// - Loads a .vrm file via GLTFLoader + VRMLoaderPlugin. On failure (or when
//   no URL is provided) returns a simple primitive "stand-in" so the example
//   always shows something renderable.
//
//////////////////////////////////////////////////////////////////////////////

import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { VRM, VRMHumanBoneName, VRMLoaderPlugin, VRMUtils } from '@pixiv/three-vrm';

export interface AvatarLayout {
  readonly height: number;
  readonly focusPoint: THREE.Vector3;
  readonly verticalExtent: number;
  readonly horizontalExtent: number;
}

export interface LoadedAvatar {
  readonly root: THREE.Object3D;
  /** Present when a real VRM was loaded; null when using the primitive fallback. */
  readonly vrm: VRM | null;
  readonly layout: AvatarLayout;
  update(deltaSeconds: number): void;
  dispose(): void;
}

/**
 * Attempts to load the given .vrm URL. Returns a primitive fallback avatar
 * when the URL is empty, missing, or the fetch fails.
 */
export async function loadAvatar(vrmUrl: string | undefined | null): Promise<LoadedAvatar> {
  if (!vrmUrl) {
    return createPrimitiveAvatar();
  }
  try {
    const loader = new GLTFLoader();
    loader.register((parser) => new VRMLoaderPlugin(parser));
    const gltf = await loader.loadAsync(vrmUrl);
    const vrm: VRM | undefined = gltf.userData.vrm;
    if (!vrm) {
      throw new Error('Loaded GLTF did not include VRM user data.');
    }
    VRMUtils.removeUnnecessaryVertices(gltf.scene);
    VRMUtils.combineSkeletons(gltf.scene);
    vrm.scene.rotation.y = Math.PI; // Face the camera by default.
    const layout = centerAvatar(vrm.scene, vrm);
    return {
      root: vrm.scene,
      vrm,
      layout,
      update(delta) {
        vrm.update(delta);
      },
      dispose() {
        VRMUtils.deepDispose(vrm.scene);
      },
    };
  } catch (error) {
    console.warn('[avatar] VRM load failed, falling back to primitive:', error);
    return createPrimitiveAvatar();
  }
}

function createPrimitiveAvatar(): LoadedAvatar {
  const group = new THREE.Group();

  const bodyMat = new THREE.MeshStandardMaterial({ color: 0x6ea8ff, roughness: 0.6, metalness: 0.1 });
  const headMat = new THREE.MeshStandardMaterial({ color: 0xffd2a8, roughness: 0.8 });
  const eyeMat = new THREE.MeshStandardMaterial({ color: 0x111111 });

  const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.22, 0.6, 8, 16), bodyMat);
  body.position.y = 0.7;
  group.add(body);

  const head = new THREE.Mesh(new THREE.SphereGeometry(0.18, 24, 24), headMat);
  head.name = 'head';
  head.position.y = 1.3;
  group.add(head);

  const eyeL = new THREE.Mesh(new THREE.SphereGeometry(0.02, 8, 8), eyeMat);
  eyeL.position.set(-0.055, 1.33, 0.16);
  group.add(eyeL);
  const eyeR = new THREE.Mesh(new THREE.SphereGeometry(0.02, 8, 8), eyeMat);
  eyeR.position.set(0.055, 1.33, 0.16);
  group.add(eyeR);

  // Arms, named so the binding layer can find them for wave gestures.
  const armL = new THREE.Mesh(new THREE.CapsuleGeometry(0.06, 0.4, 6, 12), bodyMat);
  armL.name = 'armL';
  armL.position.set(-0.28, 0.85, 0);
  group.add(armL);
  const armR = new THREE.Mesh(new THREE.CapsuleGeometry(0.06, 0.4, 6, 12), bodyMat);
  armR.name = 'armR';
  armR.position.set(0.28, 0.85, 0);
  group.add(armR);

  return {
    root: group,
    vrm: null,
    layout: centerAvatar(group, null),
    update() {
      // Primitive avatar is static; gestures are driven by the binding.
    },
    dispose() {
      group.traverse((obj) => {
        const mesh = obj as THREE.Mesh;
        if (mesh.geometry) {
          mesh.geometry.dispose();
        }
        const mat = mesh.material;
        if (Array.isArray(mat)) {
          mat.forEach((m) => m.dispose());
        } else if (mat) {
          mat.dispose();
        }
      });
    },
  };
}

function centerAvatar(root: THREE.Object3D, vrm: VRM | null): AvatarLayout {
  const bounds = new THREE.Box3();
  const center = new THREE.Vector3();
  const size = new THREE.Vector3();
  const headPos = new THREE.Vector3();

  root.updateMatrixWorld(true);
  bounds.setFromObject(root);
  if (bounds.isEmpty()) {
    return {
      height: 1.8,
      focusPoint: new THREE.Vector3(0, 1.1, 0),
      verticalExtent: 1.1,
      horizontalExtent: 0.5,
    };
  }

  bounds.getCenter(center);
  root.position.x -= center.x;
  root.position.z -= center.z;
  root.position.y -= bounds.min.y;

  root.updateMatrixWorld(true);
  bounds.setFromObject(root);
  bounds.getSize(size);

  const height = Math.max(size.y, 0.8);
  const centerY = (bounds.min.y + bounds.max.y) * 0.5;
  const headNode =
    vrm?.humanoid?.getNormalizedBoneNode(VRMHumanBoneName.Head) ?? root.getObjectByName('head');
  const headY = headNode
    ? headNode.getWorldPosition(headPos).y
    : bounds.max.y - height * 0.12;
  const focusY = THREE.MathUtils.clamp(
    THREE.MathUtils.lerp(centerY, headY, 0.35),
    bounds.min.y + height * 0.38,
    bounds.max.y - height * 0.12
  );

  return {
    height,
    focusPoint: new THREE.Vector3(0, focusY, 0),
    verticalExtent: Math.max(focusY - bounds.min.y, bounds.max.y - focusY),
    horizontalExtent: Math.max(
      Math.abs(bounds.min.x),
      Math.abs(bounds.max.x),
      Math.abs(bounds.min.z),
      Math.abs(bounds.max.z),
      size.x * 0.5,
      0.35
    ),
  };
}
