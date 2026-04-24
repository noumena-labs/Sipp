//////////////////////////////////////////////////////////////////////////////
//
// render/agent-mesh.ts
//
// - Cylindrical puck for each agent, capped with a small "nose" to show
//   heading. Pairs with an emoji billboard for emotion display and a
//   canvas sprite for the agent name.
//
//////////////////////////////////////////////////////////////////////////////

import * as THREE from 'three';
import { createEmojiBillboard, type EmojiBillboard } from './emoji-billboard.js';

export interface AgentVisual {
  readonly root: THREE.Group;
  readonly body: THREE.Mesh;
  readonly nameSprite: THREE.Sprite;
  readonly emoji: EmojiBillboard;
  readonly propAnchor: THREE.Group;
  setPosition(x: number, z: number): void;
  setHeading(radians: number): void;
  setHighlighted(on: boolean): void;
  dispose(): void;
}

const BODY_HEIGHT = 0.6;
const BODY_RADIUS = 0.35;

export function createAgentVisual(name: string, color: string): AgentVisual {
  const root = new THREE.Group();

  const bodyMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color(color),
    roughness: 0.55,
    metalness: 0.05,
  });
  const body = new THREE.Mesh(
    new THREE.CylinderGeometry(BODY_RADIUS, BODY_RADIUS, BODY_HEIGHT, 24),
    bodyMat
  );
  body.position.y = BODY_HEIGHT / 2;
  root.add(body);

  // Heading nose.
  const noseMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color(color).offsetHSL(0, 0, -0.15),
    roughness: 0.5,
  });
  const nose = new THREE.Mesh(new THREE.ConeGeometry(0.12, 0.24, 16), noseMat);
  nose.rotation.z = -Math.PI / 2;
  nose.position.set(BODY_RADIUS + 0.05, BODY_HEIGHT / 2, 0);
  root.add(nose);

  // Name label sprite.
  const labelCanvas = document.createElement('canvas');
  labelCanvas.width = 256;
  labelCanvas.height = 64;
  const lctx = labelCanvas.getContext('2d')!;
  lctx.fillStyle = 'rgba(0,0,0,0)';
  lctx.fillRect(0, 0, 256, 64);
  lctx.font = 'bold 36px system-ui, sans-serif';
  lctx.textAlign = 'center';
  lctx.textBaseline = 'middle';
  lctx.strokeStyle = 'rgba(0,0,0,0.85)';
  lctx.lineWidth = 6;
  lctx.strokeText(name, 128, 32);
  lctx.fillStyle = '#ffffff';
  lctx.fillText(name, 128, 32);
  const labelTex = new THREE.CanvasTexture(labelCanvas);
  labelTex.colorSpace = THREE.SRGBColorSpace;
  const nameSprite = new THREE.Sprite(
    new THREE.SpriteMaterial({ map: labelTex, transparent: true, depthTest: false })
  );
  nameSprite.scale.set(1.6, 0.4, 1);
  nameSprite.position.set(0, BODY_HEIGHT + 0.25, 0);
  root.add(nameSprite);

  // Emoji billboard above.
  const emoji = createEmojiBillboard();
  emoji.sprite.position.set(0, BODY_HEIGHT + 0.95, 0);
  emoji.setVisible(false);
  root.add(emoji.sprite);

  const propAnchor = new THREE.Group();
  propAnchor.position.set(0, BODY_HEIGHT + 0.35, 0);
  root.add(propAnchor);

  // Highlight ring (hidden by default).
  const ring = new THREE.Mesh(
    new THREE.RingGeometry(BODY_RADIUS + 0.08, BODY_RADIUS + 0.18, 32),
    new THREE.MeshBasicMaterial({ color: 0xffd166, side: THREE.DoubleSide, transparent: true, opacity: 0.9 })
  );
  ring.rotation.x = -Math.PI / 2;
  ring.position.y = 0.01;
  ring.visible = false;
  root.add(ring);

  return {
    root,
    body,
    nameSprite,
    emoji,
    propAnchor,
    setPosition(x, z) {
      root.position.set(x, 0, z);
    },
    setHeading(rad) {
      root.rotation.y = -rad; // THREE Y rotation is left-handed vs our heading
    },
    setHighlighted(on) {
      ring.visible = on;
    },
    dispose() {
      body.geometry.dispose();
      bodyMat.dispose();
      nose.geometry.dispose();
      noseMat.dispose();
      labelTex.dispose();
      (nameSprite.material as THREE.SpriteMaterial).dispose();
      (ring.material as THREE.Material).dispose();
      ring.geometry.dispose();
      emoji.dispose();
    },
  };
}
