//////////////////////////////////////////////////////////////////////////////
//
// vrm-loader.ts
//
// - Loads and recenters a .vrm file via GLTFLoader + VRMLoaderPlugin.
//   If no VRM is available, the scene simply renders without an avatar.
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
  readonly vrm: VRM;
  readonly layout: AvatarLayout;
  update(deltaSeconds: number): void;
  dispose(): void;
}

/**
 * Attempts to load the given .vrm URL. Returns `null` when the URL is empty,
 * missing, or the fetch / parse fails.
 */
export async function loadAvatar(vrmUrl: string | undefined | null): Promise<LoadedAvatar | null> {
  if (!vrmUrl) {
    return null;
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
    console.warn('[avatar] VRM load failed:', error);
    return null;
  }
}

export function getAvatarHeadNode(avatar: LoadedAvatar): THREE.Object3D | null {
  return resolveHeadNode(avatar.root, avatar.vrm);
}

function resolveHeadNode(root: THREE.Object3D, vrm: VRM): THREE.Object3D | null {
  return (
    vrm.humanoid?.getRawBoneNode(VRMHumanBoneName.Head) ??
    vrm.humanoid?.getNormalizedBoneNode(VRMHumanBoneName.Head) ??
    null
  );
}

function centerAvatar(root: THREE.Object3D, vrm: VRM): AvatarLayout {
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
  const headNode = resolveHeadNode(root, vrm);
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
