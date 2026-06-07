//////////////////////////////////////////////////////////////////////////////
//
// speech-bubble.ts
//
// - Renders a soft fantasy dialogue window near the avatar.
// - The full panel is drawn into one CanvasTexture so the fill, nameplate,
//   action chips, and text stay visually cohesive.
//
//////////////////////////////////////////////////////////////////////////////

import * as THREE from 'three';
import type { ChatMessage } from '../components/chat-types';
import type { LoadedAvatar } from './vrm-loader';
import { getAvatarHeadNode } from './vrm-loader';

const CANVAS_WIDTH = 1400;
const CANVAS_HEIGHT = 420;
const MAX_LINES = 4;
const TEXT_FONT = '600 38px "Trebuchet MS", "Segoe UI", sans-serif';
const NAME_FONT = '800 26px "Trebuchet MS", "Segoe UI", sans-serif';
const ACTION_FONT = '800 21px "Trebuchet MS", "Segoe UI", sans-serif';
const TEXT_LINE_HEIGHT = 49;
const TEXT_AREA_WIDTH = 1040;
const TEXT_START_X = 172;
const TEXT_CENTER_Y = 252;
const FLOAT_SPEED = 1.05;
const FLOAT_AMPLITUDE = 0.014;
const PANEL_SCALE_MULTIPLIER = 0.66;
const PANEL_CAMERA_PULL = 0.1;
const PANEL_HORIZONTAL_RATIO = 0.18;
const PANEL_VERTICAL_RATIO = 0.24;
const PANEL_VERTICAL_BIAS = 0.24;

interface SpeechBubbleOptions {
  readonly scene: THREE.Scene;
  readonly camera: THREE.PerspectiveCamera;
  readonly avatar: LoadedAvatar;
  readonly speakerName?: string;
}

type SpeechBubbleAction = ChatMessage['actions'][number];

export class SpeechBubble {
  private readonly scene: THREE.Scene;
  private readonly camera: THREE.PerspectiveCamera;
  private readonly avatar: LoadedAvatar;
  private readonly speakerName: string;
  private readonly headNode: THREE.Object3D | null;
  private readonly root = new THREE.Group();
  private readonly material: THREE.MeshBasicMaterial;
  private readonly canvas: HTMLCanvasElement;
  private readonly context: CanvasRenderingContext2D;
  private readonly texture: THREE.CanvasTexture;
  private readonly headWorld = new THREE.Vector3();
  private readonly cameraOffset = new THREE.Vector3();
  private readonly worldTarget = new THREE.Vector3();
  private readonly scale: number;
  private text = '';
  private pending = false;
  private actions: readonly SpeechBubbleAction[] = [];
  private dirty = true;
  private visibility = 0;
  private targetVisibility = 0;
  private elapsedSeconds = 0;

  public constructor({ scene, camera, avatar, speakerName = 'Aria' }: SpeechBubbleOptions) {
    this.scene = scene;
    this.camera = camera;
    this.avatar = avatar;
    this.speakerName = speakerName;
    this.headNode = getAvatarHeadNode(avatar);
    this.scale =
      THREE.MathUtils.clamp(this.avatar.layout.height / 1.8, 0.88, 1.18) *
      PANEL_SCALE_MULTIPLIER;

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

    this.material = new THREE.MeshBasicMaterial({
      map: this.texture,
      transparent: true,
      opacity: 0,
      depthWrite: false,
      depthTest: true,
      toneMapped: false,
    });

    const panel = new THREE.Mesh(new THREE.PlaneGeometry(2.28, 0.68), this.material);
    panel.renderOrder = 20;
    this.root.add(panel);
    this.root.scale.setScalar(this.scale);
    this.root.visible = false;
    this.scene.add(this.root);
    this.redraw();
  }

  public setContent(
    text: string,
    pending: boolean,
    actions: readonly SpeechBubbleAction[] = []
  ): void {
    const nextText = text.trim();
    if (this.text === nextText && this.pending === pending && sameActions(this.actions, actions)) {
      return;
    }
    this.text = nextText;
    this.pending = pending;
    this.actions = actions;
    this.targetVisibility = nextText.length > 0 || pending ? 1 : 0;
    this.dirty = true;
  }

  public tick(deltaSeconds: number): void {
    this.elapsedSeconds += deltaSeconds;
    this.visibility = THREE.MathUtils.damp(this.visibility, this.targetVisibility, 10, deltaSeconds);
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
    this.material.opacity = easeOutCubic(this.visibility);
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

    drawDialogueWindow(context, this.speakerName, this.actions);

    context.font = TEXT_FONT;
    context.textBaseline = 'alphabetic';
    context.fillStyle = '#fff6df';
    context.shadowColor = 'rgba(3, 2, 12, 0.82)';
    context.shadowBlur = 4;
    context.shadowOffsetX = 2;
    context.shadowOffsetY = 3;

    const { lines, truncated } = wrapText(context, displayText, TEXT_AREA_WIDTH, MAX_LINES);
    const bubbleLines = truncated ? appendEllipsis(lines) : lines;
    const totalHeight = bubbleLines.length * TEXT_LINE_HEIGHT;
    const startY = TEXT_CENTER_Y - totalHeight / 2 + TEXT_LINE_HEIGHT * 0.76;

    bubbleLines.forEach((line, index) => {
      context.fillText(line, TEXT_START_X, startY + index * TEXT_LINE_HEIGHT);
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
      .multiplyScalar(PANEL_CAMERA_PULL);
    const verticalOffset = this.avatar.layout.height * PANEL_VERTICAL_RATIO + PANEL_VERTICAL_BIAS;
    const floatOffset = Math.sin(this.elapsedSeconds * FLOAT_SPEED) * FLOAT_AMPLITUDE;
    const driftX = Math.sin(this.elapsedSeconds * FLOAT_SPEED * 0.5) * 0.006;

    this.worldTarget.copy(this.headWorld);
    this.worldTarget.x += this.cameraOffset.x + driftX;
    this.worldTarget.x += this.avatar.layout.height * PANEL_HORIZONTAL_RATIO;
    this.worldTarget.y += verticalOffset + floatOffset;
    this.worldTarget.z += this.cameraOffset.z;
    this.root.position.copy(this.worldTarget);
    this.root.scale.setScalar(this.scale * (0.97 + this.visibility * 0.03));
  }
}

function drawDialogueWindow(
  context: CanvasRenderingContext2D,
  speakerName: string,
  actions: readonly SpeechBubbleAction[]
): void {
  const panelX = 68;
  const panelY = 74;
  const panelW = 1264;
  const panelH = 300;
  const wedgeX = panelX + panelW * 0.26;
  const wedgeBaseY = panelY + panelH - 8;
  const wedgeTipY = CANVAS_HEIGHT - 12;

  context.save();
  context.shadowColor = 'rgba(3, 2, 12, 0.7)';
  context.shadowBlur = 34;
  context.shadowOffsetY = 14;

  context.beginPath();
  roundedRectPath(context, panelX, panelY, panelW, panelH, 26);
  context.moveTo(wedgeX - 54, wedgeBaseY);
  context.lineTo(wedgeX + 54, wedgeBaseY);
  context.lineTo(wedgeX - 12, wedgeTipY);
  context.closePath();
  const shellGradient = context.createLinearGradient(panelX, panelY, panelX + panelW, panelY + panelH);
  shellGradient.addColorStop(0, 'rgba(26, 17, 42, 0.96)');
  shellGradient.addColorStop(0.58, 'rgba(14, 10, 30, 0.98)');
  shellGradient.addColorStop(1, 'rgba(30, 17, 47, 0.96)');
  context.fillStyle = shellGradient;
  context.fill();

  // const fillGradient = context.createRadialGradient(
  //   panelX + panelW * 0.2,
  //   panelY,
  //   40,
  //   panelX + panelW * 0.42,
  //   panelY + panelH * 0.34,
  //   panelW * 0.72
  // );
  // fillGradient.addColorStop(0, 'rgba(255, 218, 143, 0.1)');
  // fillGradient.addColorStop(0.34, 'rgba(28, 18, 49, 0.9)');
  // fillGradient.addColorStop(1, 'rgba(8, 6, 20, 0.98)');
  // context.shadowBlur = 0;
  // context.fillStyle = fillGradient;
  // context.fill();

  // context.globalCompositeOperation = 'screen';
  // const warmGlow = context.createRadialGradient(
  //   panelX + panelW * 0.16,
  //   panelY + panelH * 0.04,
  //   0,
  //   panelX + panelW * 0.16,
  //   panelY + panelH * 0.04,
  //   panelW * 0.38
  // );
  // warmGlow.addColorStop(0, 'rgba(255, 218, 143, 0.16)');
  // warmGlow.addColorStop(1, 'rgba(255, 218, 143, 0)');
  // context.fillStyle = warmGlow;
  // context.fill();

  // const violetGlow = context.createRadialGradient(
  //   panelX + panelW * 0.8,
  //   panelY + panelH * 0.86,
  //   0,
  //   panelX + panelW * 0.8,
  //   panelY + panelH * 0.86,
  //   panelW * 0.4
  // );
  // violetGlow.addColorStop(0, 'rgba(151, 105, 255, 0.16)');
  // violetGlow.addColorStop(1, 'rgba(151, 105, 255, 0)');
  // context.fillStyle = violetGlow;
  // context.fill();
  // context.globalCompositeOperation = 'source-over';

  context.globalAlpha = 0.1;
  context.beginPath();
  roundedRectPath(context, panelX + 26, panelY + 24, panelW - 52, panelH - 48, 22);
  const innerSheen = context.createLinearGradient(panelX, panelY, panelX + panelW, panelY + panelH);
  innerSheen.addColorStop(0, 'rgba(255, 238, 190, 0.18)');
  innerSheen.addColorStop(0.52, 'rgba(151, 105, 255, 0.08)');
  innerSheen.addColorStop(1, 'rgba(82, 255, 211, 0.08)');
  context.fillStyle = innerSheen;
  context.fill();
  context.globalAlpha = 1;

  drawNameplate(context, speakerName, panelX + 86, panelY + 34);
  drawActionCapsules(context, actions, panelX + panelW - 86, panelY + 45);
  context.restore();
}

function drawNameplate(
  context: CanvasRenderingContext2D,
  speakerName: string,
  x: number,
  y: number
): void {
  context.font = NAME_FONT;
  const width = Math.max(150, context.measureText(speakerName).width + 76);
  const height = 40;

  context.save();
  context.shadowColor = 'rgba(3, 2, 12, 0.38)';
  context.shadowBlur = 12;
  const gradient = context.createLinearGradient(x, y, x + width, y + height);
  gradient.addColorStop(0, 'rgba(255, 218, 143, 0.2)');
  gradient.addColorStop(0.5, 'rgba(151, 105, 255, 0.2)');
  gradient.addColorStop(1, 'rgba(82, 255, 211, 0.12)');
  roundedRectPath(context, x, y, width, height, 14);
  context.fillStyle = gradient;
  context.fill();

  context.textBaseline = 'middle';
  context.fillStyle = '#fff6df';
  context.shadowColor = 'rgba(0, 0, 0, 0.72)';
  context.shadowBlur = 4;
  context.fillText(speakerName, x + 26, y + height / 2 + 1);
  context.restore();
}

function drawActionCapsules(
  context: CanvasRenderingContext2D,
  actions: readonly SpeechBubbleAction[],
  rightX: number,
  y: number
): void {
  if (actions.length === 0) {
    return;
  }

  context.save();
  context.font = ACTION_FONT;
  context.textBaseline = 'middle';
  let x = rightX;
  const visibleActions = actions.slice(-3).reverse();
  visibleActions.forEach((action) => {
    const label = formatActionLabel(action.label);
    const width = Math.min(260, Math.max(96, context.measureText(label).width + 42));
    x -= width;
    const gradient = context.createLinearGradient(x, y, x + width, y + 34);
    gradient.addColorStop(0, 'rgba(82, 255, 211, 0.2)');
    gradient.addColorStop(1, 'rgba(151, 105, 255, 0.18)');
    roundedRectPath(context, x, y, width, 34, 17);
    context.shadowColor = 'rgba(3, 2, 12, 0.3)';
    context.shadowBlur = 8;
    context.fillStyle = gradient;
    context.fill();
    context.fillStyle = '#c4fff4';
    context.shadowColor = 'rgba(0, 0, 0, 0.55)';
    context.shadowBlur = 3;
    context.fillText(label, x + 21, y + 18);
    context.shadowBlur = 0;
    x -= 12;
  });
  context.restore();
}

function formatActionLabel(label: string): string {
  return label.replace(/_/g, ' ').replace(/\b\w/g, (match) => match.toUpperCase());
}

function sameActions(
  current: readonly SpeechBubbleAction[],
  next: readonly SpeechBubbleAction[]
): boolean {
  if (current.length !== next.length) {
    return false;
  }
  return current.every((action, index) => {
    const nextAction = next[index];
    return nextAction?.id === action.id && nextAction.label === action.label;
  });
}

function roundedRectPath(
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
  context.lineTo(x + width - safeRadius, y);
  context.quadraticCurveTo(x + width, y, x + width, y + safeRadius);
  context.lineTo(x + width, y + height - safeRadius);
  context.quadraticCurveTo(x + width, y + height, x + width - safeRadius, y + height);
  context.lineTo(x + safeRadius, y + height);
  context.quadraticCurveTo(x, y + height, x, y + height - safeRadius);
  context.lineTo(x, y + safeRadius);
  context.quadraticCurveTo(x, y, x + safeRadius, y);
  context.closePath();
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

function easeOutCubic(t: number): number {
  return 1 - Math.pow(1 - t, 3);
}
