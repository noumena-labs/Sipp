//////////////////////////////////////////////////////////////////////////////
//
// speech-bubble.ts
//
// - Renders a world-space billboard bubble above the avatar head using a
//   canvas texture so long streaming text stays readable and cheap to update.
//
//////////////////////////////////////////////////////////////////////////////

import * as THREE from 'three';
import type { LoadedAvatar } from './vrm-loader';
import { getAvatarHeadNode } from './vrm-loader';

const CANVAS_WIDTH = 1024;
const CANVAS_HEIGHT = 640;
const MAX_LINES = 6;
const TEXT_FONT = '600 52px "Trebuchet MS", "Segoe UI", sans-serif';
const LABEL_FONT = '600 24px "Segoe UI", sans-serif';
const TEXT_LINE_HEIGHT = 68;
const TEXT_AREA_WIDTH = 828;
const FLOAT_SPEED = 1.75;

interface SpeechBubbleOptions {
  readonly scene: THREE.Scene;
  readonly camera: THREE.PerspectiveCamera;
  readonly avatar: LoadedAvatar;
}

export class SpeechBubble {
  private readonly scene: THREE.Scene;
  private readonly camera: THREE.PerspectiveCamera;
  private readonly avatar: LoadedAvatar;
  private readonly headNode: THREE.Object3D | null;
  private readonly root = new THREE.Group();
  private readonly glowMaterial: THREE.MeshBasicMaterial;
  private readonly backMaterial: THREE.MeshBasicMaterial;
  private readonly frontMaterial: THREE.MeshBasicMaterial;
  private readonly tailBackMaterial: THREE.MeshBasicMaterial;
  private readonly tailFrontMaterial: THREE.MeshBasicMaterial;
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
    this.scale = THREE.MathUtils.clamp(this.avatar.layout.height / 1.8, 0.86, 1.16);

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

    const glow = new THREE.Mesh(
      new THREE.PlaneGeometry(1.86, 1.18),
      new THREE.MeshBasicMaterial({
        color: 0x7bc4ff,
        transparent: true,
        opacity: 0,
        depthWrite: false,
        depthTest: false,
        toneMapped: false,
      })
    );
    glow.position.z = -0.05;
    this.glowMaterial = glow.material as THREE.MeshBasicMaterial;

    const backPlate = new THREE.Mesh(
      new THREE.PlaneGeometry(1.56, 0.98),
      new THREE.MeshBasicMaterial({
        color: 0x26123e,
        transparent: true,
        opacity: 0,
        depthWrite: false,
        depthTest: false,
        toneMapped: false,
      })
    );
    backPlate.position.set(0.04, -0.03, -0.03);
    this.backMaterial = backPlate.material as THREE.MeshBasicMaterial;

    const frontPlate = new THREE.Mesh(
      new THREE.PlaneGeometry(1.48, 0.92),
      new THREE.MeshBasicMaterial({
        map: this.texture,
        transparent: true,
        opacity: 0,
        depthWrite: false,
        depthTest: false,
        toneMapped: false,
      })
    );
    this.frontMaterial = frontPlate.material as THREE.MeshBasicMaterial;

    const tailBack = new THREE.Mesh(
      new THREE.PlaneGeometry(0.2, 0.2),
      new THREE.MeshBasicMaterial({
        color: 0x26123e,
        transparent: true,
        opacity: 0,
        depthWrite: false,
        depthTest: false,
        toneMapped: false,
      })
    );
    tailBack.position.set(-0.36, -0.53, -0.03);
    tailBack.rotation.z = Math.PI / 4;
    this.tailBackMaterial = tailBack.material as THREE.MeshBasicMaterial;

    const tailFront = new THREE.Mesh(
      new THREE.PlaneGeometry(0.16, 0.16),
      new THREE.MeshBasicMaterial({
        color: 0x90d8ff,
        transparent: true,
        opacity: 0,
        depthWrite: false,
        depthTest: false,
        toneMapped: false,
      })
    );
    tailFront.position.set(-0.34, -0.5, 0.01);
    tailFront.rotation.z = Math.PI / 4;
    this.tailFrontMaterial = tailFront.material as THREE.MeshBasicMaterial;

    this.root.add(glow, backPlate, tailBack, tailFront, frontPlate);
    this.root.scale.setScalar(this.scale);
    this.root.visible = false;
    this.root.renderOrder = 10;
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

    const gradient = context.createLinearGradient(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
    gradient.addColorStop(0, 'rgba(144, 216, 255, 0.96)');
    gradient.addColorStop(0.42, 'rgba(172, 199, 255, 0.94)');
    gradient.addColorStop(1, 'rgba(211, 191, 255, 0.94)');
    drawRoundedRect(context, 40, 28, CANVAS_WIDTH - 92, CANVAS_HEIGHT - 86, 70);
    context.fillStyle = gradient;
    context.fill();

    context.lineWidth = 8;
    context.strokeStyle = 'rgba(255, 255, 255, 0.85)';
    context.stroke();

    context.fillStyle = 'rgba(255, 255, 255, 0.16)';
    drawRoundedRect(context, 76, 64, CANVAS_WIDTH - 244, 92, 44);
    context.fill();

    context.fillStyle = '#25113b';
    context.font = LABEL_FONT;
    context.fillText('Reply', 110, 122);

    context.fillStyle = '#150a29';
    context.font = TEXT_FONT;
    const { lines, truncated } = wrapText(context, displayText, TEXT_AREA_WIDTH, MAX_LINES);
    const bubbleLines = truncated ? appendEllipsis(lines) : lines;
    bubbleLines.forEach((line, index) => {
      context.fillText(line, 110, 218 + index * TEXT_LINE_HEIGHT);
    });

    if (this.pending) {
      context.fillStyle = 'rgba(34, 13, 71, 0.64)';
      drawRoundedRect(context, CANVAS_WIDTH - 206, CANVAS_HEIGHT - 130, 110, 56, 24);
      context.fill();
      context.fillStyle = '#ffffff';
      context.font = '700 26px "Segoe UI", sans-serif';
      context.fillText('LIVE', CANVAS_WIDTH - 177, CANVAS_HEIGHT - 92);
    }

    this.texture.needsUpdate = true;
  }

  private updatePosition(): void {
    if (this.headNode) {
      this.headNode.getWorldPosition(this.headWorld);
    } else {
      this.headWorld.set(0, this.avatar.layout.focusPoint.y + this.avatar.layout.height * 0.3, 0);
    }

    this.cameraOffset.copy(this.camera.position).sub(this.headWorld).normalize().multiplyScalar(0.18);
    const verticalOffset = this.avatar.layout.height * 0.18 + 0.2;
    const floatOffset = Math.sin(this.elapsedSeconds * FLOAT_SPEED) * 0.03;

    this.worldTarget.copy(this.headWorld);
    this.worldTarget.x += this.cameraOffset.x;
    this.worldTarget.y += verticalOffset + floatOffset;
    this.worldTarget.z += this.cameraOffset.z;
    this.root.position.copy(this.worldTarget);
    this.root.scale.setScalar(this.scale * (0.92 + this.visibility * 0.08));
  }

  private updateMaterials(): void {
    this.glowMaterial.opacity = this.visibility * 0.16;
    this.backMaterial.opacity = this.visibility * 0.95;
    this.frontMaterial.opacity = this.visibility;
    this.tailBackMaterial.opacity = this.visibility * 0.95;
    this.tailFrontMaterial.opacity = this.visibility;
  }
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

function drawRoundedRect(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  width: number,
  height: number,
  radius: number
): void {
  const safeRadius = Math.min(radius, width / 2, height / 2);
  context.beginPath();
  context.moveTo(x + safeRadius, y);
  context.arcTo(x + width, y, x + width, y + height, safeRadius);
  context.arcTo(x + width, y + height, x, y + height, safeRadius);
  context.arcTo(x, y + height, x, y, safeRadius);
  context.arcTo(x, y, x + width, y, safeRadius);
  context.closePath();
}
