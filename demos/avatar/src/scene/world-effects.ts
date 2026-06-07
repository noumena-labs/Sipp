//////////////////////////////////////////////////////////////////////////////
//
// world-effects.ts
//
// - Small procedural fantasy effects used by avatar actions. These are
//   intentionally asset-free so the demo stays portable.
//
//////////////////////////////////////////////////////////////////////////////

import * as THREE from 'three';
import { VRMHumanBoneName } from '@pixiv/three-vrm';
import type { WorldEffectActionName } from '../actions/world-effects';
import type { LoadedAvatar } from './vrm-loader';

interface TimedEffect {
  readonly root: THREE.Object3D;
  readonly durationSeconds: number;
  ageSeconds: number;
  update(deltaSeconds: number, ageRatio: number): void;
}

const STAR_GOLD = new THREE.Color('#ffd98a');
const RUNE_BLUE = new THREE.Color('#7ddfff');
const WARD_VIOLET = new THREE.Color('#b48cff');
const FAMILIAR_TEAL = new THREE.Color('#69ffd6');

export class FantasyWorldEffects {
  private readonly scene: THREE.Scene;
  private readonly avatar: LoadedAvatar;
  private readonly effects: TimedEffect[] = [];
  private readonly tmpVec = new THREE.Vector3();
  private readonly tmpVec2 = new THREE.Vector3();

  public constructor(scene: THREE.Scene, avatar: LoadedAvatar) {
    this.scene = scene;
    this.avatar = avatar;
  }

  public trigger(name: WorldEffectActionName): void {
    switch (name) {
      case 'summon_familiar':
        this.addEffect(this.createFamiliarEffect());
        return;
      case 'cast_starbolt':
        this.addEffect(this.createStarboltEffect());
        return;
      case 'raise_ward':
        this.addEffect(this.createWardEffect());
        return;
      case 'summon_rune_circle':
        this.addEffect(this.createRuneCircleEffect());
        return;
    }
  }

  public tick(deltaSeconds: number): void {
    for (let index = this.effects.length - 1; index >= 0; index -= 1) {
      const effect = this.effects[index];
      effect.ageSeconds += deltaSeconds;
      const ageRatio = THREE.MathUtils.clamp(effect.ageSeconds / effect.durationSeconds, 0, 1);
      effect.update(deltaSeconds, ageRatio);
      if (effect.ageSeconds >= effect.durationSeconds) {
        this.scene.remove(effect.root);
        disposeObject(effect.root);
        this.effects.splice(index, 1);
      }
    }
  }

  public dispose(): void {
    for (const effect of this.effects) {
      this.scene.remove(effect.root);
      disposeObject(effect.root);
    }
    this.effects.length = 0;
  }

  private addEffect(effect: TimedEffect): void {
    this.effects.push(effect);
    this.scene.add(effect.root);
  }

  private createFamiliarEffect(): TimedEffect {
    const root = new THREE.Group();
    const core = new THREE.Mesh(
      new THREE.SphereGeometry(0.075, 24, 16),
      new THREE.MeshBasicMaterial({
        color: FAMILIAR_TEAL,
        transparent: true,
        opacity: 0.9,
        blending: THREE.AdditiveBlending,
      })
    );
    const aura = new THREE.Mesh(
      new THREE.SphereGeometry(0.18, 24, 16),
      new THREE.MeshBasicMaterial({
        color: FAMILIAR_TEAL,
        transparent: true,
        opacity: 0.18,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      })
    );
    const wingMaterial = new THREE.MeshBasicMaterial({
      color: '#d8fff4',
      transparent: true,
      opacity: 0.34,
      blending: THREE.AdditiveBlending,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const leftWing = new THREE.Mesh(new THREE.CircleGeometry(0.11, 18), wingMaterial);
    const rightWing = new THREE.Mesh(new THREE.CircleGeometry(0.11, 18), wingMaterial.clone());
    leftWing.scale.set(1.4, 0.52, 1);
    rightWing.scale.set(1.4, 0.52, 1);
    root.add(aura, core, leftWing, rightWing);

    return {
      root,
      durationSeconds: 8,
      ageSeconds: 0,
      update: (_deltaSeconds, ageRatio) => {
        const anchor = this.getAvatarAnchor(0.48, 0.78, 0.06, this.tmpVec);
        const orbit = ageRatio * Math.PI * 7;
        root.position.set(
          anchor.x + Math.cos(orbit) * 0.34,
          anchor.y + Math.sin(orbit * 1.7) * 0.055,
          anchor.z + Math.sin(orbit) * 0.2
        );
        const pulse = 0.9 + Math.sin(ageRatio * Math.PI * 34) * 0.12;
        core.scale.setScalar(pulse);
        aura.scale.setScalar(0.95 + Math.sin(ageRatio * Math.PI * 16) * 0.1);
        leftWing.position.set(-0.11, 0.025, 0.012);
        rightWing.position.set(0.11, 0.025, 0.012);
        leftWing.rotation.set(0.35, Math.sin(ageRatio * Math.PI * 32) * 0.6, 0.2);
        rightWing.rotation.set(0.35, -Math.sin(ageRatio * Math.PI * 32) * 0.6, -0.2);
        const fade = fadeInOut(ageRatio, 0.12, 0.18);
        setOpacity(core, 0.9 * fade);
        setOpacity(aura, 0.18 * fade);
        setOpacity(leftWing, 0.34 * fade);
        setOpacity(rightWing, 0.34 * fade);
      },
    };
  }

  private createStarboltEffect(): TimedEffect {
    const root = new THREE.Group();
    const bolt = new THREE.Mesh(
      new THREE.SphereGeometry(0.08, 24, 16),
      new THREE.MeshBasicMaterial({
        color: STAR_GOLD,
        transparent: true,
        opacity: 0.96,
        blending: THREE.AdditiveBlending,
      })
    );
    const halo = new THREE.Mesh(
      new THREE.SphereGeometry(0.2, 24, 16),
      new THREE.MeshBasicMaterial({
        color: STAR_GOLD,
        transparent: true,
        opacity: 0.2,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      })
    );
    const trail = new THREE.Mesh(
      new THREE.CylinderGeometry(0.022, 0.09, 0.75, 18, 1, true),
      new THREE.MeshBasicMaterial({
        color: '#fff3bc',
        transparent: true,
        opacity: 0.24,
        blending: THREE.AdditiveBlending,
        side: THREE.DoubleSide,
        depthWrite: false,
      })
    );
    trail.rotation.x = Math.PI / 2;
    trail.position.z = -0.34;
    root.add(trail, halo, bolt);

    const origin = this.getRightHandAnchor(this.tmpVec).clone();
    const destination = origin.clone().add(new THREE.Vector3(0.04, 0.08, 2.6));

    return {
      root,
      durationSeconds: 1.15,
      ageSeconds: 0,
      update: (_deltaSeconds, ageRatio) => {
        root.position.copy(origin).lerp(destination, easeOutCubic(ageRatio));
        root.rotation.z += 0.24;
        const burst = ageRatio > 0.78 ? 1 + (ageRatio - 0.78) * 9 : 1;
        bolt.scale.setScalar(burst);
        halo.scale.setScalar(1.1 + burst * 0.7);
        const fade = fadeInOut(ageRatio, 0.05, 0.24);
        setOpacity(bolt, 0.96 * fade);
        setOpacity(halo, 0.22 * fade);
        setOpacity(trail, 0.24 * fade * (1 - ageRatio * 0.6));
      },
    };
  }

  private createWardEffect(): TimedEffect {
    const root = new THREE.Group();
    const disc = new THREE.Mesh(
      new THREE.CircleGeometry(0.52, 64),
      new THREE.MeshBasicMaterial({
        color: WARD_VIOLET,
        transparent: true,
        opacity: 0.16,
        blending: THREE.AdditiveBlending,
        side: THREE.DoubleSide,
        depthWrite: false,
      })
    );
    const outer = new THREE.Mesh(
      new THREE.TorusGeometry(0.52, 0.012, 12, 80),
      new THREE.MeshBasicMaterial({
        color: '#d7c5ff',
        transparent: true,
        opacity: 0.84,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      })
    );
    const inner = new THREE.Mesh(
      new THREE.TorusGeometry(0.34, 0.007, 10, 64),
      (outer.material as THREE.MeshBasicMaterial).clone()
    );
    const spokes = createRadialMarks(10, 0.43, 0.1, '#efe7ff');
    root.add(disc, outer, inner, spokes);
    storeBaseOpacity(root);

    return {
      root,
      durationSeconds: 2.6,
      ageSeconds: 0,
      update: (_deltaSeconds, ageRatio) => {
        const anchor = this.getAvatarAnchor(0, 0.52, 0.86, this.tmpVec);
        root.position.copy(anchor);
        root.rotation.set(0, 0, ageRatio * Math.PI * 1.8);
        const scale = 0.2 + easeOutCubic(Math.min(ageRatio / 0.24, 1)) * 0.95;
        root.scale.setScalar(scale);
        const fade = fadeInOut(ageRatio, 0.08, 0.34);
        root.traverse((object) => {
          if (object instanceof THREE.Mesh) {
            const material = object.material as THREE.MeshBasicMaterial;
            material.opacity = material.userData.baseOpacity * fade;
          }
        });
      },
    };
  }

  private createRuneCircleEffect(): TimedEffect {
    const root = new THREE.Group();
    root.rotation.x = -Math.PI / 2;
    const outer = new THREE.Mesh(
      new THREE.TorusGeometry(0.9, 0.01, 10, 96),
      new THREE.MeshBasicMaterial({
        color: RUNE_BLUE,
        transparent: true,
        opacity: 0.75,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      })
    );
    const inner = new THREE.Mesh(
      new THREE.TorusGeometry(0.62, 0.007, 10, 80),
      (outer.material as THREE.MeshBasicMaterial).clone()
    );
    const center = new THREE.Mesh(
      new THREE.CircleGeometry(0.42, 64),
      new THREE.MeshBasicMaterial({
        color: '#8fffe8',
        transparent: true,
        opacity: 0.12,
        blending: THREE.AdditiveBlending,
        side: THREE.DoubleSide,
        depthWrite: false,
      })
    );
    const marks = createRadialMarks(18, 0.76, 0.07, '#c8fff8');
    root.add(center, outer, inner, marks);
    storeBaseOpacity(root);

    return {
      root,
      durationSeconds: 3.2,
      ageSeconds: 0,
      update: (_deltaSeconds, ageRatio) => {
        const anchor = this.getAvatarAnchor(0, 0.012, 0, this.tmpVec);
        root.position.copy(anchor);
        root.rotation.z = ageRatio * Math.PI * 2.2;
        const pulse = 0.82 + Math.sin(ageRatio * Math.PI * 7) * 0.08;
        root.scale.setScalar(pulse);
        const fade = fadeInOut(ageRatio, 0.1, 0.22);
        root.traverse((object) => {
          if (object instanceof THREE.Mesh) {
            const material = object.material as THREE.MeshBasicMaterial;
            material.opacity = material.userData.baseOpacity * fade;
          }
        });
      },
    };
  }

  private getRightHandAnchor(target: THREE.Vector3): THREE.Vector3 {
    const hand = this.avatar.vrm.humanoid?.getNormalizedBoneNode(VRMHumanBoneName.RightHand);
    if (hand) {
      hand.getWorldPosition(target);
      return target;
    }
    return this.getAvatarAnchor(0.24, 0.64, 0.24, target);
  }

  private getAvatarAnchor(xRatio: number, yRatio: number, zOffset: number, target: THREE.Vector3): THREE.Vector3 {
    this.avatar.root.getWorldPosition(target);
    this.avatar.root.getWorldScale(this.tmpVec2);
    target.x += this.avatar.layout.height * xRatio * this.tmpVec2.x;
    target.y += this.avatar.layout.height * yRatio * this.tmpVec2.y;
    target.z += zOffset;
    return target;
  }
}

function createRadialMarks(count: number, radius: number, length: number, color: string): THREE.Group {
  const group = new THREE.Group();
  const geometry = new THREE.BoxGeometry(0.018, length, 0.008);
  for (let index = 0; index < count; index += 1) {
    const angle = (index / count) * Math.PI * 2;
    const mark = new THREE.Mesh(
      geometry,
      new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity: 0.62,
        blending: THREE.AdditiveBlending,
        depthWrite: false,
      })
    );
    mark.position.set(Math.cos(angle) * radius, Math.sin(angle) * radius, 0.006);
    mark.rotation.z = angle;
    group.add(mark);
  }
  return group;
}

function fadeInOut(ageRatio: number, fadeIn: number, fadeOut: number): number {
  const intro = THREE.MathUtils.clamp(ageRatio / fadeIn, 0, 1);
  const outro = THREE.MathUtils.clamp((1 - ageRatio) / fadeOut, 0, 1);
  return easeOutCubic(Math.min(intro, outro));
}

function setOpacity(mesh: THREE.Mesh, opacity: number): void {
  const material = mesh.material;
  if (Array.isArray(material)) {
    material.forEach((entry) => {
      if ('opacity' in entry) {
        entry.opacity = opacity;
      }
    });
    return;
  }
  material.opacity = opacity;
}

function disposeObject(root: THREE.Object3D): void {
  const disposedGeometries = new Set<THREE.BufferGeometry>();
  const disposedMaterials = new Set<THREE.Material>();
  root.traverse((object) => {
    const mesh = object as THREE.Mesh;
    if (mesh.geometry && !disposedGeometries.has(mesh.geometry)) {
      disposedGeometries.add(mesh.geometry);
      mesh.geometry.dispose();
    }
    const material = mesh.material;
    if (Array.isArray(material)) {
      material.forEach((entry) => {
        if (!disposedMaterials.has(entry)) {
          disposedMaterials.add(entry);
          entry.dispose();
        }
      });
    } else if (material && !disposedMaterials.has(material)) {
      disposedMaterials.add(material);
      material.dispose();
    }
  });
}

function storeBaseOpacity(root: THREE.Object3D): void {
  root.traverse((object) => {
    if (!(object instanceof THREE.Mesh)) {
      return;
    }
    const materials = Array.isArray(object.material) ? object.material : [object.material];
    materials.forEach((material) => {
      material.userData.baseOpacity = material.opacity;
    });
  });
}

function easeOutCubic(t: number): number {
  return 1 - Math.pow(1 - t, 3);
}
