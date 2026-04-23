//////////////////////////////////////////////////////////////////////////////
//
// speech-bubble.ts
//
// - Renders a fluffy, hovering white cloud above the avatar head.
// - Cloud silhouette built from a cluster of overlapping spheres with an
//   unlit soft-white shader (subtle vertical gradient + rim feathering).
// - Streamed assistant text drawn to a CanvasTexture on a transparent plane
//   tucked inside the cloud silhouette.
//
//////////////////////////////////////////////////////////////////////////////

import * as THREE from 'three';
import type { LoadedAvatar } from './vrm-loader';
import { getAvatarHeadNode } from './vrm-loader';

const CANVAS_WIDTH = 1024;
const CANVAS_HEIGHT = 320;
const MAX_LINES = 5;
const TEXT_FONT = '600 38px "Trebuchet MS", "Segoe UI", sans-serif';
const TEXT_LINE_HEIGHT = 46;
const TEXT_AREA_WIDTH = 820;
const FLOAT_SPEED = 1.4;
const FLOAT_AMPLITUDE = 0.028;
const BUBBLE_SCALE_MULTIPLIER = 0.5;
const BUBBLE_CAMERA_PULL = 0.08;
const BUBBLE_HORIZONTAL_RATIO = 0.3;
const BUBBLE_VERTICAL_RATIO = 0.16;
const BUBBLE_VERTICAL_BIAS = 0.15;

// Unlit cloud shader: soft white body with a gentle top-to-bottom gradient
// and a feathered rim that fades at glancing angles, giving volumetric read.
const CLOUD_VERTEX_SHADER = `
varying vec3 vNormal;
varying vec3 vViewDir;
varying vec3 vLocalPos;

void main() {
  vLocalPos = position;
  vec4 worldPos = modelMatrix * vec4(position, 1.0);
  vec4 viewPos = viewMatrix * worldPos;
  vNormal = normalize(normalMatrix * normal);
  vViewDir = normalize(-viewPos.xyz);
  gl_Position = projectionMatrix * viewPos;
}
`;

const CLOUD_FRAGMENT_SHADER = `
uniform vec3 uTopColor;
uniform vec3 uBottomColor;
uniform vec3 uShadowColor;
uniform float uOpacity;
uniform float uGradientCenter;

varying vec3 vNormal;
varying vec3 vViewDir;
varying vec3 vLocalPos;

void main() {
  vec3 n = normalize(vNormal);
  vec3 v = normalize(vViewDir);
  float facing = clamp(dot(n, v), 0.0, 1.0);

  // Vertical gradient across the puff: brighter near the top, softly
  // shadowed underneath so the cloud reads as lit from above.
  float vertical = clamp(vLocalPos.y / max(uGradientCenter, 0.0001) * 0.5 + 0.5, 0.0, 1.0);
  float gradient = smoothstep(0.1, 0.9, vertical);
  vec3 base = mix(uBottomColor, uTopColor, gradient);

  // Under-shadow on the bottom faces, very subtle.
  float underShadow = smoothstep(0.55, 0.0, vertical) * 0.22;
  base = mix(base, uShadowColor, underShadow);

  // Feather the silhouette so sphere seams dissolve into the cloud edge.
  float edge = pow(facing, 1.6);
  float alpha = uOpacity * smoothstep(0.02, 0.55, edge);

  gl_FragColor = vec4(base, alpha);
}
`;

interface SpeechBubbleOptions {
  readonly scene: THREE.Scene;
  readonly camera: THREE.PerspectiveCamera;
  readonly avatar: LoadedAvatar;
}

interface CloudUniforms {
  readonly uTopColor: THREE.IUniform<THREE.Color>;
  readonly uBottomColor: THREE.IUniform<THREE.Color>;
  readonly uShadowColor: THREE.IUniform<THREE.Color>;
  readonly uOpacity: THREE.IUniform<number>;
  readonly uGradientCenter: THREE.IUniform<number>;
}

type CloudMaterial = THREE.ShaderMaterial & {
  readonly uniforms: CloudUniforms;
};

interface CloudPuff {
  readonly x: number;
  readonly y: number;
  readonly z: number;
  readonly radius: number;
}

// Hand-authored cluster that reads as a rounded, slightly wider-than-tall
// cartoon cloud. Coordinates are in the bubble's local space; Y up.
const CLOUD_PUFFS: readonly CloudPuff[] = [
  { x: 0.0, y: 0.02, z: 0.0, radius: 0.46 },
  { x: -0.52, y: -0.02, z: 0.0, radius: 0.38 },
  { x: 0.52, y: -0.02, z: 0.0, radius: 0.38 },
  { x: -0.28, y: 0.26, z: 0.04, radius: 0.34 },
  { x: 0.28, y: 0.26, z: 0.04, radius: 0.34 },
  { x: -0.78, y: -0.08, z: -0.02, radius: 0.26 },
  { x: 0.78, y: -0.08, z: -0.02, radius: 0.26 },
  { x: 0.0, y: 0.34, z: 0.02, radius: 0.3 },
  { x: -0.18, y: -0.26, z: 0.02, radius: 0.3 },
  { x: 0.22, y: -0.28, z: 0.02, radius: 0.3 },
  { x: -0.6, y: -0.26, z: 0.0, radius: 0.22 },
  { x: 0.6, y: -0.26, z: 0.0, radius: 0.22 },
];

const CLOUD_HALF_HEIGHT = 0.55;

export class SpeechBubble {
  private readonly scene: THREE.Scene;
  private readonly camera: THREE.PerspectiveCamera;
  private readonly avatar: LoadedAvatar;
  private readonly headNode: THREE.Object3D | null;
  private readonly root = new THREE.Group();
  private readonly cloudGroup = new THREE.Group();
  private readonly cloudMaterial: CloudMaterial;
  private readonly faceMaterial: THREE.MeshBasicMaterial;
  private readonly canvas: HTMLCanvasElement;
  private readonly context: CanvasRenderingContext2D;
  private readonly texture: THREE.CanvasTexture;
  private readonly headWorld = new THREE.Vector3();
  private readonly cameraOffset = new THREE.Vector3();
  private readonly worldTarget = new THREE.Vector3();
  private readonly scale: number;
  private text = '';
  private pending = false;
  private dirty = true;
  private visibility = 0;
  private targetVisibility = 0;
  private elapsedSeconds = 0;

  public constructor({ scene, camera, avatar }: SpeechBubbleOptions) {
    this.scene = scene;
    this.camera = camera;
    this.avatar = avatar;
    this.headNode = getAvatarHeadNode(avatar);
    this.scale =
      THREE.MathUtils.clamp(this.avatar.layout.height / 1.8, 0.86, 1.16) *
      BUBBLE_SCALE_MULTIPLIER;

    this.canvas = document.createElement('canvas');
    this.canvas.width = CANVAS_WIDTH;
    this.canvas.height = CANVAS_HEIGHT;
    const context = this.canvas.getContext('2d');
    if (!context) {
      throw new Error('Unable to create 2D canvas context for speech bubble.');
    }
    this.context = context;

    this.texture = new THREE.CanvasTexture(this.canvas);
    this.texture.colorSpace = THREE.SRGBColorSpace;
    this.texture.minFilter = THREE.LinearFilter;
    this.texture.magFilter = THREE.LinearFilter;
    this.texture.generateMipmaps = false;

    this.cloudMaterial = createCloudMaterial();

    const puffGeometry = new THREE.SphereGeometry(1, 32, 24);
    for (const puff of CLOUD_PUFFS) {
      const mesh = new THREE.Mesh(puffGeometry, this.cloudMaterial);
      mesh.position.set(puff.x, puff.y, puff.z);
      mesh.scale.setScalar(puff.radius);
      mesh.renderOrder = 10;
      this.cloudGroup.add(mesh);
    }

    const face = new THREE.Mesh(
      new THREE.PlaneGeometry(1.6, 0.62),
      new THREE.MeshBasicMaterial({
        map: this.texture,
        transparent: true,
        opacity: 0,
        depthWrite: false,
        depthTest: true,
        toneMapped: false,
      })
    );
    face.position.set(0, 0.04, 0.48);
    face.renderOrder = 12;
    this.faceMaterial = face.material as THREE.MeshBasicMaterial;

    this.root.add(this.cloudGroup, face);
    this.root.scale.setScalar(this.scale);
    this.root.visible = false;
    this.scene.add(this.root);
  }

  public setContent(text: string, pending: boolean): void {
    const nextText = text.trim();
    if (this.text === nextText && this.pending === pending) {
      return;
    }
    this.text = nextText;
    this.pending = pending;
    this.targetVisibility = nextText.length > 0 || pending ? 1 : 0;
    this.dirty = true;
  }

  public tick(deltaSeconds: number): void {
    this.elapsedSeconds += deltaSeconds;
    this.visibility = THREE.MathUtils.damp(this.visibility, this.targetVisibility, 9, deltaSeconds);
    if (this.dirty) {
      this.redraw();
      this.dirty = false;
    }

    if (this.visibility < 0.02) {
      this.root.visible = false;
      return;
    }

    this.root.visible = true;
    this.root.quaternion.copy(this.camera.quaternion);
    this.updatePosition();
    this.updateMaterials();
  }

  public dispose(): void {
    this.scene.remove(this.root);
    this.texture.dispose();
    this.root.traverse((object) => {
      const mesh = object as THREE.Mesh;
      if (mesh.geometry) {
        mesh.geometry.dispose();
      }
      const material = mesh.material;
      if (Array.isArray(material)) {
        material.forEach((entry) => entry.dispose());
      } else if (material) {
        material.dispose();
      }
    });
  }

  private redraw(): void {
    const context = this.context;
    const displayText = this.text.length > 0 ? this.text : this.pending ? '...' : '';

    context.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    if (displayText.length === 0) {
      this.texture.needsUpdate = true;
      return;
    }

    context.font = TEXT_FONT;
    context.textBaseline = 'alphabetic';
    context.fillStyle = '#2a2550';
    context.shadowColor = 'rgba(255, 255, 255, 0.9)';
    context.shadowBlur = 10;
    context.shadowOffsetX = 0;
    context.shadowOffsetY = 0;

    const { lines, truncated } = wrapText(context, displayText, TEXT_AREA_WIDTH, MAX_LINES);
    const bubbleLines = truncated ? appendEllipsis(lines) : lines;

    const totalHeight = bubbleLines.length * TEXT_LINE_HEIGHT;
    const startY = (CANVAS_HEIGHT - totalHeight) / 2 + TEXT_LINE_HEIGHT * 0.75;

    bubbleLines.forEach((line, index) => {
      const metrics = context.measureText(line);
      const x = (CANVAS_WIDTH - metrics.width) / 2;
      context.fillText(line, x, startY + index * TEXT_LINE_HEIGHT);
    });

    context.shadowBlur = 0;
    this.texture.needsUpdate = true;
  }

  private updatePosition(): void {
    if (this.headNode) {
      this.headNode.getWorldPosition(this.headWorld);
    } else {
      this.headWorld.set(0, this.avatar.layout.focusPoint.y + this.avatar.layout.height * 0.3, 0);
    }

    this.cameraOffset
      .copy(this.camera.position)
      .sub(this.headWorld)
      .normalize()
      .multiplyScalar(BUBBLE_CAMERA_PULL);
    const verticalOffset = this.avatar.layout.height * BUBBLE_VERTICAL_RATIO + BUBBLE_VERTICAL_BIAS;
    const floatOffset = Math.sin(this.elapsedSeconds * FLOAT_SPEED) * FLOAT_AMPLITUDE;
    const driftX = Math.sin(this.elapsedSeconds * FLOAT_SPEED * 0.6) * 0.012;

    this.worldTarget.copy(this.headWorld);
    this.worldTarget.x += this.cameraOffset.x + driftX;
    this.worldTarget.x += this.avatar.layout.height * BUBBLE_HORIZONTAL_RATIO;
    this.worldTarget.y += verticalOffset + floatOffset;
    this.worldTarget.z += this.cameraOffset.z;
    this.root.position.copy(this.worldTarget);
    this.root.scale.setScalar(this.scale * (0.95 + this.visibility * 0.05));
  }

  private updateMaterials(): void {
    this.cloudMaterial.uniforms.uOpacity.value = this.visibility;
    this.faceMaterial.opacity = this.visibility;
  }
}

function createCloudMaterial(): CloudMaterial {
  const uniforms: CloudUniforms = {
    uTopColor: { value: new THREE.Color('#ffffff') },
    uBottomColor: { value: new THREE.Color('#eef2ff') },
    uShadowColor: { value: new THREE.Color('#c9d3ef') },
    uOpacity: { value: 0 },
    uGradientCenter: { value: CLOUD_HALF_HEIGHT },
  };

  return new THREE.ShaderMaterial({
    uniforms,
    vertexShader: CLOUD_VERTEX_SHADER,
    fragmentShader: CLOUD_FRAGMENT_SHADER,
    transparent: true,
    depthWrite: false,
    depthTest: true,
    toneMapped: false,
  }) as CloudMaterial;
}

function wrapText(
  context: CanvasRenderingContext2D,
  text: string,
  maxWidth: number,
  maxLines: number
): { lines: string[]; truncated: boolean } {
  const paragraphs = text.replace(/\r/g, '').split(/\n+/);
  const lines: string[] = [];

  for (const paragraph of paragraphs) {
    if (paragraph.trim().length === 0) {
      if (lines.length >= maxLines) {
        return { lines, truncated: true };
      }
      lines.push('');
      continue;
    }

    const words = paragraph.split(/\s+/);
    let currentLine = '';
    for (const word of words) {
      if (word.length === 0) {
        continue;
      }

      const candidate = currentLine.length > 0 ? `${currentLine} ${word}` : word;
      if (context.measureText(candidate).width <= maxWidth) {
        currentLine = candidate;
        continue;
      }

      if (currentLine.length > 0) {
        if (lines.length >= maxLines) {
          return { lines, truncated: true };
        }
        lines.push(currentLine);
      }

      if (context.measureText(word).width <= maxWidth) {
        currentLine = word;
        continue;
      }

      const chunks = splitLongWord(context, word, maxWidth);
      for (let index = 0; index < chunks.length - 1; index += 1) {
        if (lines.length >= maxLines) {
          return { lines, truncated: true };
        }
        lines.push(chunks[index]);
      }
      currentLine = chunks[chunks.length - 1] ?? '';
    }

    if (currentLine.length > 0) {
      if (lines.length >= maxLines) {
        return { lines, truncated: true };
      }
      lines.push(currentLine);
    }
  }

  return { lines, truncated: false };
}

function splitLongWord(
  context: CanvasRenderingContext2D,
  word: string,
  maxWidth: number
): string[] {
  const parts: string[] = [];
  let current = '';
  for (const character of word) {
    const candidate = `${current}${character}`;
    if (current.length > 0 && context.measureText(candidate).width > maxWidth) {
      parts.push(current);
      current = character;
    } else {
      current = candidate;
    }
  }
  if (current.length > 0) {
    parts.push(current);
  }
  return parts;
}

function appendEllipsis(lines: string[]): string[] {
  if (lines.length === 0) {
    return ['...'];
  }
  const result = lines.slice(0, MAX_LINES);
  const lastLine = result[result.length - 1] ?? '';
  result[result.length - 1] = `${lastLine.replace(/[\s.]+$/u, '')}...`;
  return result;
}
