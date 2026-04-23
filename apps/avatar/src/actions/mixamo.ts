import * as THREE from 'three';
import { FBXLoader } from 'three/examples/jsm/loaders/FBXLoader.js';
import type { VRM, VRMHumanBoneName } from '@pixiv/three-vrm';

export const CLIP_ACTION_NAMES = ['wave', 'nod', 'shake_head'] as const;

export type ClipActionName = (typeof CLIP_ACTION_NAMES)[number];

const MIXAMO_VRM_RIG_MAP: Record<string, VRMHumanBoneName> = {
  mixamorigHips: 'hips',
  mixamorigSpine: 'spine',
  mixamorigSpine1: 'chest',
  mixamorigSpine2: 'upperChest',
  mixamorigNeck: 'neck',
  mixamorigHead: 'head',
  mixamorigLeftShoulder: 'leftShoulder',
  mixamorigLeftArm: 'leftUpperArm',
  mixamorigLeftForeArm: 'leftLowerArm',
  mixamorigLeftHand: 'leftHand',
  mixamorigLeftHandThumb1: 'leftThumbMetacarpal',
  mixamorigLeftHandThumb2: 'leftThumbProximal',
  mixamorigLeftHandThumb3: 'leftThumbDistal',
  mixamorigLeftHandIndex1: 'leftIndexProximal',
  mixamorigLeftHandIndex2: 'leftIndexIntermediate',
  mixamorigLeftHandIndex3: 'leftIndexDistal',
  mixamorigLeftHandMiddle1: 'leftMiddleProximal',
  mixamorigLeftHandMiddle2: 'leftMiddleIntermediate',
  mixamorigLeftHandMiddle3: 'leftMiddleDistal',
  mixamorigLeftHandRing1: 'leftRingProximal',
  mixamorigLeftHandRing2: 'leftRingIntermediate',
  mixamorigLeftHandRing3: 'leftRingDistal',
  mixamorigLeftHandPinky1: 'leftLittleProximal',
  mixamorigLeftHandPinky2: 'leftLittleIntermediate',
  mixamorigLeftHandPinky3: 'leftLittleDistal',
  mixamorigRightShoulder: 'rightShoulder',
  mixamorigRightArm: 'rightUpperArm',
  mixamorigRightForeArm: 'rightLowerArm',
  mixamorigRightHand: 'rightHand',
  mixamorigRightHandPinky1: 'rightLittleProximal',
  mixamorigRightHandPinky2: 'rightLittleIntermediate',
  mixamorigRightHandPinky3: 'rightLittleDistal',
  mixamorigRightHandRing1: 'rightRingProximal',
  mixamorigRightHandRing2: 'rightRingIntermediate',
  mixamorigRightHandRing3: 'rightRingDistal',
  mixamorigRightHandMiddle1: 'rightMiddleProximal',
  mixamorigRightHandMiddle2: 'rightMiddleIntermediate',
  mixamorigRightHandMiddle3: 'rightMiddleDistal',
  mixamorigRightHandIndex1: 'rightIndexProximal',
  mixamorigRightHandIndex2: 'rightIndexIntermediate',
  mixamorigRightHandIndex3: 'rightIndexDistal',
  mixamorigRightHandThumb1: 'rightThumbMetacarpal',
  mixamorigRightHandThumb2: 'rightThumbProximal',
  mixamorigRightHandThumb3: 'rightThumbDistal',
  mixamorigLeftUpLeg: 'leftUpperLeg',
  mixamorigLeftLeg: 'leftLowerLeg',
  mixamorigLeftFoot: 'leftFoot',
  mixamorigLeftToeBase: 'leftToes',
  mixamorigRightUpLeg: 'rightUpperLeg',
  mixamorigRightLeg: 'rightLowerLeg',
  mixamorigRightFoot: 'rightFoot',
  mixamorigRightToeBase: 'rightToes',
};

export function isClipActionName(name: string): name is ClipActionName {
  return (CLIP_ACTION_NAMES as readonly string[]).includes(name);
}

export async function loadMixamoAnimationClip(url: string, vrm: VRM): Promise<THREE.AnimationClip> {
  const loader = new FBXLoader();
  const asset = await loader.loadAsync(url);
  const clip = THREE.AnimationClip.findByName(asset.animations, 'mixamo.com') ?? asset.animations[0];
  if (!clip) {
    throw new Error(`Mixamo animation "${url}" did not include any animation clips.`);
  }

  const motionHips = asset.getObjectByName('mixamorigHips');
  const vrmHipsPosition = vrm.humanoid.normalizedRestPose.hips?.position;
  const vrmHipsHeight = vrmHipsPosition?.[1];
  if (!motionHips || vrmHipsHeight == null || motionHips.position.y === 0) {
    throw new Error(`Unable to retarget Mixamo animation "${url}" onto the current VRM.`);
  }

  const tracks: THREE.KeyframeTrack[] = [];
  const restRotationInverse = new THREE.Quaternion();
  const parentRestWorldRotation = new THREE.Quaternion();
  const quat = new THREE.Quaternion();
  const hipsPositionScale = vrmHipsHeight / motionHips.position.y;

  for (const track of clip.tracks) {
    const [mixamoRigName, propertyName] = track.name.split('.');
    const vrmBoneName = MIXAMO_VRM_RIG_MAP[mixamoRigName];
    if (!vrmBoneName) {
      continue;
    }

    const vrmNodeName = vrm.humanoid.getNormalizedBoneNode(vrmBoneName)?.name;
    const mixamoRigNode = asset.getObjectByName(mixamoRigName);
    if (!vrmNodeName || !mixamoRigNode) {
      continue;
    }

    mixamoRigNode.getWorldQuaternion(restRotationInverse).invert();
    if (mixamoRigNode.parent) {
      mixamoRigNode.parent.getWorldQuaternion(parentRestWorldRotation);
    } else {
      parentRestWorldRotation.identity();
    }

    if (track instanceof THREE.QuaternionKeyframeTrack) {
      const values = track.values.slice();
      for (let index = 0; index < values.length; index += 4) {
        const flatQuaternion = values.slice(index, index + 4);
        quat.fromArray(flatQuaternion);
        quat.premultiply(parentRestWorldRotation).multiply(restRotationInverse);
        quat.toArray(flatQuaternion);
        flatQuaternion.forEach((value, offset) => {
          values[index + offset] = value;
        });
      }

      tracks.push(
        new THREE.QuaternionKeyframeTrack(
          `${vrmNodeName}.${propertyName}`,
          track.times,
          values.map((value, index) =>
            vrm.meta?.metaVersion === '0' && index % 2 === 0 ? -value : value
          )
        )
      );
      continue;
    }

    if (track instanceof THREE.VectorKeyframeTrack) {
      tracks.push(
        new THREE.VectorKeyframeTrack(
          `${vrmNodeName}.${propertyName}`,
          track.times,
          track.values.map((value, index) =>
            (vrm.meta?.metaVersion === '0' && index % 3 !== 1 ? -value : value) * hipsPositionScale
          )
        )
      );
    }
  }

  if (tracks.length === 0) {
    throw new Error(`Mixamo animation "${url}" could not be retargeted to the current VRM rig.`);
  }

  return new THREE.AnimationClip('mixamo-vrm-animation', clip.duration, tracks);
}
