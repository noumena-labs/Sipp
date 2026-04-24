//////////////////////////////////////////////////////////////////////////////
//
// render/emoji-billboard.ts
//
// - Canvas-texture sprite used to render a single emoji glyph above an
//   agent. The glyph can be swapped by re-rendering the canvas, avoiding
//   allocation of new textures per frame.
//
//////////////////////////////////////////////////////////////////////////////

import * as THREE from 'three';

export const DEFAULT_EMOTION_GLYPHS: Readonly<Record<string, string>> = {
  thinking: '\u{1F914}',
  curious: '\u{1F9D0}',
  happy: '\u{1F60A}',
  confused: '\u{1F615}',
  alert: '\u{1F440}',
  frustrated: '\u{1F624}',
  sleepy: '\u{1F634}',
  celebrate: '\u{1F389}',
  surprised: '\u{1F4A2}',
};

export function emotionGlyphFor(emotion: string): string {
  return DEFAULT_EMOTION_GLYPHS[emotion] ?? '\u{2728}';
}

export interface EmojiBillboard {
  readonly sprite: THREE.Sprite;
  setGlyph(glyph: string): void;
  setVisible(visible: boolean): void;
  dispose(): void;
}

export function createEmojiBillboard(): EmojiBillboard {
  const canvas = document.createElement('canvas');
  canvas.width = 128;
  canvas.height = 128;
  const ctx = canvas.getContext('2d')!;
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  const material = new THREE.SpriteMaterial({
    map: texture,
    transparent: true,
    depthWrite: false,
  });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(0.9, 0.9, 1);

  let currentGlyph = '';

  const draw = (glyph: string): void => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!glyph) return;
    ctx.font = '96px "Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji", sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(glyph, canvas.width / 2, canvas.height / 2 + 4);
    texture.needsUpdate = true;
  };

  return {
    sprite,
    setGlyph(glyph) {
      if (glyph === currentGlyph) return;
      currentGlyph = glyph;
      draw(glyph);
    },
    setVisible(visible) {
      sprite.visible = visible;
    },
    dispose() {
      texture.dispose();
      material.dispose();
    },
  };
}
