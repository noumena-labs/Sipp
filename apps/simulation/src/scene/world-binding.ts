//////////////////////////////////////////////////////////////////////////////
//
// scene/world-binding.ts
//
// - Mirrors the app-owned simulation runtime snapshots into the three.js scene.
// - Applies immediate visual reactions from simulation bus events so query,
//   bump, drop, and score feedback does not wait for the next tick snapshot.
//
//////////////////////////////////////////////////////////////////////////////

import * as THREE from 'three';
import type {
  SimulationBus,
  SimulationEvent,
} from '../runtime/bus.js';
import type { AgentIntent, SimulationAgentState, SimulationGameEvent, Vec2, WorldSnapshot } from '../runtime/types.js';
import { createAgentVisual, type AgentVisual } from '../render/agent-mesh.js';
import { createBatMesh } from '../render/bat-mesh.js';
import { createObjectVisual, type ObjectVisual } from '../render/object-mesh.js';
import { emotionGlyphFor } from '../render/emoji-billboard.js';
import { AGENT_COLOR_BY_ID } from '../scenarios/courtyard-snack.js';

const LERP_ALPHA = 0.22;
const PULSE_SECONDS = 0.8;
const QUERY_GLYPH = '...';
const FLASH_SECONDS = 0.32;
const IMPACT_SECONDS = 0.28;
const CONFETTI_SECONDS = 0.9;
const DROP_ARC_SECONDS = 0.75;
const ICE_THROW_ARC_SECONDS = 0.64;
const FROZEN_SHAKE_SECONDS = 0.18;

interface AgentEntry {
  name: string;
  readonly visual: AgentVisual;
  readonly iceShell: THREE.Mesh;
  readonly targetMarker: THREE.Mesh;
  readonly targetLine: THREE.Line;
  targetX: number;
  targetZ: number;
  targetHeading: number;
  status: string;
  holding: string | null;
  pulseUntil: number;
  glyphOverride: string | null;
  glyphOverrideUntil: number;
  flashColor: THREE.Color | null;
  flashUntil: number;
  joltUntil: number;
  joltDirection: Vec2 | null;
  baseColor: THREE.Color;
  propMesh: THREE.Object3D | null;
  propKind: string | null;
}

interface ObjectEntry {
  readonly id: string;
  readonly visual: ObjectVisual;
  kind: string;
  label: string;
  description: string;
  collisionRadius: number;
  targetX: number;
  targetZ: number;
  heldBy: string | null;
  toss: TossAnimation | null;
}

interface TossAnimation {
  readonly startX: number;
  readonly startZ: number;
  readonly endX: number;
  readonly endZ: number;
  readonly startAt: number;
  readonly endAt: number;
}

interface BurstEffect {
  readonly root: THREE.Group;
  readonly dispose: () => void;
  readonly endAt: number;
}

interface ProjectileEffect {
  readonly root: THREE.Group;
  readonly startX: number;
  readonly startZ: number;
  readonly endX: number;
  readonly endZ: number;
  readonly startAt: number;
  readonly endAt: number;
  dispose(): void;
}

export interface WorldBinding {
  applySnapshot(snapshot: WorldSnapshot): void;
  dispose(): void;
  pickObject(ray: THREE.Ray): HoveredSceneObject | null;
  setHighlightedAgent(agentId: string | null): void;
  setHoveredObject(objectId: string | null): void;
}

export interface HoveredSceneObject {
  readonly id: string;
  readonly label: string;
  readonly description: string;
}

export function bindWorldToScene(
  bus: SimulationBus,
  worldRoot: THREE.Group,
  onFrame: (cb: (dt: number) => void) => () => void
): WorldBinding {
  const agents = new Map<string, AgentEntry>();
  const objects = new Map<string, ObjectEntry>();
  const bursts = new Set<BurstEffect>();
  const projectiles = new Set<ProjectileEffect>();
  const hoverPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
  let highlightedAgent: string | null = null;
  let hoveredObject: string | null = null;
  let elapsedSeconds = 0;

  const ensureAgent = (id: string, name: string): AgentEntry => {
    let entry = agents.get(id);
    if (entry) {
      entry.name = name;
      return entry;
    }
    const colorHex = AGENT_COLOR_BY_ID.get(id) ?? '#c0c0c0';
    const baseColor = new THREE.Color(colorHex);
    const visual = createAgentVisual(name, colorHex);
    worldRoot.add(visual.root);
    const iceShell = new THREE.Mesh(
      new THREE.BoxGeometry(0.95, 1.25, 0.95),
      new THREE.MeshStandardMaterial({
        color: 0x9ae8ff,
        roughness: 0.14,
        metalness: 0.08,
        transparent: true,
        opacity: 0.45,
      })
    );
    iceShell.position.set(0, 0.62, 0);
    iceShell.visible = false;
    visual.root.add(iceShell);
    const targetMarker = new THREE.Mesh(
      new THREE.RingGeometry(0.28, 0.38, 32),
      new THREE.MeshBasicMaterial({
        color: baseColor,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.55,
        depthWrite: false,
      })
    );
    targetMarker.rotation.x = -Math.PI / 2;
    targetMarker.position.y = 0.018;
    targetMarker.visible = false;
    worldRoot.add(targetMarker);

    const targetLine = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(), new THREE.Vector3()]),
      new THREE.LineBasicMaterial({
        color: baseColor,
        transparent: true,
        opacity: 0.42,
        depthWrite: false,
      })
    );
    targetLine.visible = false;
    worldRoot.add(targetLine);

    entry = {
      name,
      visual,
      iceShell,
      targetMarker,
      targetLine,
      targetX: 0,
      targetZ: 0,
      targetHeading: 0,
      status: '',
      holding: null,
      pulseUntil: 0,
      glyphOverride: null,
      glyphOverrideUntil: 0,
      flashColor: null,
      flashUntil: 0,
      joltUntil: 0,
      joltDirection: null,
      baseColor,
      propMesh: null,
      propKind: null,
    };
    agents.set(id, entry);
    if (highlightedAgent === id) visual.setHighlighted(true);
    return entry;
  };

  const ensureObject = (
    id: string,
    kind: string,
    label: string,
    description: string,
    collisionRadius: number
  ): ObjectEntry => {
    let entry = objects.get(id);
    if (entry) {
      entry.kind = kind;
      entry.label = label;
      entry.description = description;
      entry.collisionRadius = collisionRadius;
      return entry;
    }
    const visual = createObjectVisual(kind);
    worldRoot.add(visual.root);
    entry = {
      id,
      visual,
      kind,
      label,
      description,
      collisionRadius,
      targetX: 0,
      targetZ: 0,
      heldBy: null,
      toss: null,
    };
    objects.set(id, entry);
    entry.visual.setHovered(id === hoveredObject);
    return entry;
  };

  const applySnapshot = (snap: WorldSnapshot): void => {
    const seenAgents = new Set<string>();
    for (const a of snap.agents) {
      seenAgents.add(a.id);
      const entry = ensureAgent(a.id, a.name);
      entry.targetX = a.position.x;
      entry.targetZ = a.position.z;
      entry.targetHeading = a.heading;
      if (entry.status !== a.status || entry.holding !== a.holding) {
        entry.status = a.status;
        entry.holding = a.holding;
        entry.pulseUntil = Math.max(entry.pulseUntil, elapsedSeconds + PULSE_SECONDS);
      }
      syncAgentProp(entry, a);
      entry.iceShell.visible = a.frozenUntilTick > snap.tick;
      setAgentTarget(entry, resolveIntentTarget(a.intent, snap));
      updateAgentGlyph(entry, a, snap);
    }
    for (const [id, entry] of agents) {
      if (!seenAgents.has(id)) {
        worldRoot.remove(entry.visual.root);
        worldRoot.remove(entry.targetMarker);
        worldRoot.remove(entry.targetLine);
        if (entry.propMesh) {
          entry.visual.propAnchor.remove(entry.propMesh);
          disposeObject3D(entry.propMesh);
        }
        entry.visual.dispose();
        disposeTargetVisuals(entry);
        agents.delete(id);
      }
    }

    const seenObjects = new Set<string>();
    for (const o of snap.objects) {
      seenObjects.add(o.id);
      const entry = ensureObject(o.id, o.kind, o.label, o.description, o.collisionRadius);
      entry.targetX = o.position.x;
      entry.targetZ = o.position.z;
      if (entry.heldBy !== o.heldBy) {
        entry.heldBy = o.heldBy;
        entry.visual.setHeldBy(o.heldBy);
      }
      entry.visual.setActive(o.active);
    }
    for (const [id, entry] of objects) {
      if (!seenObjects.has(id)) {
        if (hoveredObject === id) {
          hoveredObject = null;
        }
        worldRoot.remove(entry.visual.root);
        entry.visual.dispose();
        objects.delete(id);
      }
    }
  };

  const handleEvent = (event: SimulationEvent): void => {
    if (event.kind === 'tick-end') {
      applySnapshot(event.snapshot);
      return;
    }
    if (event.kind === 'agent-query-start') {
      const entry = agents.get(event.agentId);
      if (entry) {
        entry.glyphOverride = QUERY_GLYPH;
        entry.glyphOverrideUntil = Number.POSITIVE_INFINITY;
        entry.pulseUntil = Math.max(entry.pulseUntil, elapsedSeconds + PULSE_SECONDS);
        entry.visual.emoji.setGlyph(QUERY_GLYPH);
        entry.visual.emoji.setVisible(true);
      }
      return;
    }
    if (event.kind === 'agent-query-end') {
      const entry = agents.get(event.agentId);
      if (entry) {
        entry.glyphOverride = null;
        entry.glyphOverrideUntil = 0;
      }
      return;
    }
    if (event.kind === 'agent-intent') {
      const entry = agents.get(event.agentId);
      if (entry) {
        entry.status = event.status;
        entry.glyphOverride = glyphForIntent(event.intent);
        entry.glyphOverrideUntil = elapsedSeconds + 0.24;
        entry.visual.emoji.setGlyph(entry.glyphOverride);
        entry.visual.emoji.setVisible(true);
        entry.pulseUntil = elapsedSeconds + PULSE_SECONDS;
      }
      return;
    }
    if (event.kind === 'game-event') {
      applyGameEvent(event.event);
    }
  };
  const unsubscribe = bus.onAny(handleEvent);

  const stopFrame = onFrame((dt) => {
    elapsedSeconds += dt;
    for (const entry of agents.values()) {
      const pos = entry.visual.root.position;
      pos.x += (entry.targetX - pos.x) * LERP_ALPHA;
      pos.z += (entry.targetZ - pos.z) * LERP_ALPHA;
      let offsetX = 0;
      let offsetZ = 0;
      if (entry.joltUntil > elapsedSeconds && entry.joltDirection) {
        const strength = (entry.joltUntil - elapsedSeconds) / IMPACT_SECONDS;
        offsetX = entry.joltDirection.x * 0.14 * strength;
        offsetZ = entry.joltDirection.z * 0.14 * strength;
      }

      entry.visual.root.position.x = pos.x + offsetX;
      entry.visual.root.position.z = pos.z + offsetZ;

      const rot = entry.visual.root.rotation;
      const desired = -entry.targetHeading;
      let delta = desired - rot.y;
      while (delta > Math.PI) delta -= Math.PI * 2;
      while (delta < -Math.PI) delta += Math.PI * 2;
      rot.y += delta * LERP_ALPHA;

      const pulse = Math.max(0, entry.pulseUntil - elapsedSeconds) / PULSE_SECONDS;
      entry.visual.body.scale.set(1 + pulse * 0.18, 1 + pulse * 0.12, 1 + pulse * 0.18);

      if (entry.flashUntil > elapsedSeconds && entry.flashColor) {
        applyBodyFlash(entry, 1 - (entry.flashUntil - elapsedSeconds) / FLASH_SECONDS);
      } else {
        clearBodyFlash(entry);
      }

      if (entry.glyphOverride && entry.glyphOverrideUntil !== Number.POSITIVE_INFINITY && entry.glyphOverrideUntil <= elapsedSeconds) {
        entry.glyphOverride = null;
        entry.glyphOverrideUntil = 0;
      }

      if (entry.targetLine.visible) {
        const target = entry.targetMarker.position;
        entry.targetLine.geometry.setFromPoints([
          new THREE.Vector3(entry.visual.root.position.x, 0.06, entry.visual.root.position.z),
          new THREE.Vector3(target.x, 0.06, target.z),
        ]);
      }
    }

    for (const entry of objects.values()) {
      if (entry.toss && entry.toss.endAt > elapsedSeconds) {
        const progress = clamp01((elapsedSeconds - entry.toss.startAt) / (entry.toss.endAt - entry.toss.startAt));
        const arcHeight = Math.sin(progress * Math.PI) * 1.05;
        const bounce = progress > 0.72 ? Math.sin((progress - 0.72) / 0.28 * Math.PI * 5) * 0.08 * (1 - progress) : 0;
        entry.visual.root.position.x = lerp(entry.toss.startX, entry.toss.endX, progress);
        entry.visual.root.position.z = lerp(entry.toss.startZ, entry.toss.endZ, progress);
        entry.visual.root.position.y = arcHeight + Math.abs(bounce);
        if (progress >= 1) {
          entry.toss = null;
          entry.visual.root.position.y = 0;
        }
        continue;
      }
      entry.visual.root.position.y = 0;
      if (entry.heldBy) {
        const holder = agents.get(entry.heldBy);
        if (holder) {
          entry.visual.root.position.x = holder.visual.root.position.x + Math.sin(holder.targetHeading) * 0.42;
          entry.visual.root.position.z = holder.visual.root.position.z + Math.cos(holder.targetHeading) * 0.42;
          continue;
        }
      }
      const p = entry.visual.root.position;
      p.x += (entry.targetX - p.x) * LERP_ALPHA;
      p.z += (entry.targetZ - p.z) * LERP_ALPHA;
      entry.visual.setPosition(p.x, p.z);
    }

    for (const projectile of Array.from(projectiles)) {
      if (projectile.endAt <= elapsedSeconds) {
        projectile.dispose();
        projectiles.delete(projectile);
        continue;
      }
      const progress = clamp01((elapsedSeconds - projectile.startAt) / (projectile.endAt - projectile.startAt));
      const arcHeight = Math.sin(progress * Math.PI) * 1.15;
      projectile.root.position.x = lerp(projectile.startX, projectile.endX, progress);
      projectile.root.position.z = lerp(projectile.startZ, projectile.endZ, progress);
      projectile.root.position.y = 0.48 + arcHeight;
      projectile.root.rotation.x += dt * 6;
      projectile.root.rotation.z += dt * 5.5;
    }

    for (const burst of Array.from(bursts)) {
      if (burst.endAt <= elapsedSeconds) {
        burst.dispose();
        bursts.delete(burst);
      }
    }
  });

  function applyGameEvent(event: SimulationGameEvent): void {
    switch (event.kind) {
      case 'pickup': {
        const entry = agents.get(event.agentId);
        if (entry) {
          entry.glyphOverride = '✋';
          entry.glyphOverrideUntil = elapsedSeconds + 0.28;
          entry.pulseUntil = elapsedSeconds + PULSE_SECONDS;
        }
        const objectEntry = objects.get(event.objectId);
        if (objectEntry && (objectEntry.kind === 'bat' || objectEntry.kind === 'ice_cube')) {
          objectEntry.visual.setActive(false);
        }
        const burstColor = objectEntry?.kind === 'ice_cube' ? 0x8fe7ff : objectEntry?.kind === 'bat' ? 0xffc86a : 0xffe066;
        spawnBurst(event.position, burstColor, 6, 0.22);
        return;
      }
      case 'drop': {
        const entry = agents.get(event.agentId);
        if (entry) {
          entry.glyphOverride = '💢';
          entry.glyphOverrideUntil = elapsedSeconds + 0.35;
          entry.flashColor = new THREE.Color(0xff8a80);
          entry.flashUntil = elapsedSeconds + FLASH_SECONDS;
        }
        const objectEntry = objects.get(event.objectId);
        if (objectEntry) {
          objectEntry.toss = {
            startX: event.from.x,
            startZ: event.from.z,
            endX: event.to.x,
            endZ: event.to.z,
            startAt: elapsedSeconds,
            endAt: elapsedSeconds + DROP_ARC_SECONDS,
          };
          objectEntry.targetX = event.to.x;
          objectEntry.targetZ = event.to.z;
          objectEntry.heldBy = null;
          objectEntry.visual.setHeldBy(null);
          objectEntry.visual.setActive(true);
        }
        const burstColor = event.cause === 'ice'
          ? 0x8fe7ff
          : event.cause === 'bat'
            ? 0xffc86a
            : event.cause === 'bump'
              ? 0xff8a80
              : 0xffd166;
        spawnBurst(event.from, burstColor, 8, 0.28);
        return;
      }
      case 'forced_drop': {
        const attacker = agents.get(event.attackerAgentId);
        const target = agents.get(event.targetAgentId);
        if (attacker) {
          attacker.glyphOverride = '💥';
          attacker.glyphOverrideUntil = elapsedSeconds + 0.32;
          attacker.flashColor = new THREE.Color(0xffc107);
          attacker.flashUntil = elapsedSeconds + FLASH_SECONDS;
        }
        if (target) {
          target.glyphOverride = event.outcome === 'drop' ? '💢' : '‼';
          target.glyphOverrideUntil = elapsedSeconds + 0.32;
          target.flashColor = new THREE.Color(0xff8a80);
          target.flashUntil = elapsedSeconds + FLASH_SECONDS;
          target.joltUntil = elapsedSeconds + IMPACT_SECONDS;
          target.joltDirection = attacker
            ? {
                x: target.visual.root.position.x - attacker.visual.root.position.x,
                z: target.visual.root.position.z - attacker.visual.root.position.z,
              }
            : { x: 0.1, z: 0.1 };
        }
        spawnBurst(event.position, 0xf9aa33, 12, 0.34);
        return;
      }
      case 'bump_whiff': {
        const attacker = agents.get(event.attackerAgentId);
        if (attacker) {
          attacker.glyphOverride = '💨';
          attacker.glyphOverrideUntil = elapsedSeconds + 0.28;
          attacker.flashColor = new THREE.Color(0xf4d35e);
          attacker.flashUntil = elapsedSeconds + FLASH_SECONDS;
        }
        spawnBurst(event.position, 0xf4d35e, 7, 0.22);
        return;
      }
      case 'power_up_throw': {
        const attacker = agents.get(event.agentId);
        if (attacker) {
          attacker.glyphOverride = '🧊';
          attacker.glyphOverrideUntil = elapsedSeconds + 0.28;
          attacker.flashColor = new THREE.Color(0x8fe7ff);
          attacker.flashUntil = elapsedSeconds + FLASH_SECONDS;
        }
        spawnIceProjectile(event.from, event.targetAtLaunch);
        return;
      }
      case 'power_up_use': {
        const attacker = agents.get(event.agentId);
        const target = agents.get(event.targetAgentId);
        if (attacker) {
          attacker.glyphOverride = event.powerUp === 'bat' ? '🏏' : '🧊';
          attacker.glyphOverrideUntil = elapsedSeconds + 0.36;
          attacker.flashColor = new THREE.Color(event.powerUp === 'bat' ? 0xffc86a : 0x8fe7ff);
          attacker.flashUntil = elapsedSeconds + FLASH_SECONDS;
        }
        if (target) {
          target.glyphOverride = event.effect === 'freeze' ? '⛄' : '💢';
          target.glyphOverrideUntil = elapsedSeconds + 0.4;
          target.flashColor = new THREE.Color(event.effect === 'freeze' ? 0x8fe7ff : 0xff8a80);
          target.flashUntil = elapsedSeconds + 0.5;
          target.joltUntil = elapsedSeconds + IMPACT_SECONDS;
          target.joltDirection = attacker
            ? {
                x: target.visual.root.position.x - attacker.visual.root.position.x,
                z: target.visual.root.position.z - attacker.visual.root.position.z,
              }
            : { x: 0.1, z: 0.1 };
        }
        spawnBurst(event.position, event.effect === 'freeze' ? 0x8fe7ff : 0xffc86a, 14, 0.34);
        return;
      }
      case 'delivery': {
        const entry = agents.get(event.agentId);
        if (entry) {
          entry.glyphOverride = '🎉';
          entry.glyphOverrideUntil = elapsedSeconds + CONFETTI_SECONDS;
          entry.flashColor = new THREE.Color(0x7bd88f);
          entry.flashUntil = elapsedSeconds + 0.55;
          entry.pulseUntil = elapsedSeconds + 0.9;
        }
        spawnBurst(event.position, 0x7bd88f, 18, CONFETTI_SECONDS);
        return;
      }
      case 'respawn': {
        const objectEntry = objects.get(event.objectId);
        if (objectEntry) {
          objectEntry.targetX = event.position.x;
          objectEntry.targetZ = event.position.z;
          objectEntry.heldBy = null;
          objectEntry.visual.setHeldBy(null);
          objectEntry.visual.setActive(true);
        }
        spawnBurst(event.position, 0xffe066, 10, 0.28);
        return;
      }
      case 'fallback':
        return;
    }
  }

  function spawnBurst(position: Vec2, colorHex: number, count: number, duration: number): void {
    const material = new THREE.MeshBasicMaterial({ color: colorHex, transparent: true, opacity: 0.9, depthWrite: false });
    const geometry = new THREE.PlaneGeometry(0.08, 0.18);
    const root = new THREE.Group();
    for (let i = 0; i < count; i += 1) {
      const particle = new THREE.Mesh(geometry, material.clone());
      const angle = (Math.PI * 2 * i) / count;
      particle.position.set(Math.sin(angle) * 0.2, 0.2 + (i % 3) * 0.08, Math.cos(angle) * 0.2);
      particle.rotation.x = -Math.PI / 2;
      particle.userData = { angle, speed: 0.35 + (i % 4) * 0.08 };
      root.add(particle);
    }
    root.position.set(position.x, 0.05, position.z);
    worldRoot.add(root);
    const burst: BurstEffect = {
      root,
      endAt: elapsedSeconds + duration,
      dispose: () => {
        worldRoot.remove(root);
        root.traverse((obj) => {
          const mesh = obj as THREE.Mesh;
          mesh.geometry?.dispose?.();
          const material = mesh.material;
          if (Array.isArray(material)) material.forEach((m) => m.dispose());
          else if (material) (material as THREE.Material).dispose();
        });
      },
    };
    bursts.add(burst);
    const start = elapsedSeconds;
    const stop = onFrame(() => {
      const progress = clamp01((elapsedSeconds - start) / duration);
      for (const child of root.children) {
        const mesh = child as THREE.Mesh;
        const user = mesh.userData as { angle: number; speed: number };
        mesh.position.x = Math.sin(user.angle) * user.speed * progress;
        mesh.position.z = Math.cos(user.angle) * user.speed * progress;
        mesh.position.y = 0.15 + Math.sin(progress * Math.PI) * 0.7;
        const material = mesh.material as THREE.MeshBasicMaterial;
        material.opacity = 0.9 * (1 - progress);
      }
      if (progress >= 1) {
        stop();
      }
    });
  }

  function spawnIceProjectile(from: Vec2, to: Vec2): void {
    const root = new THREE.Group();
    const cube = new THREE.Mesh(
      new THREE.BoxGeometry(0.28, 0.28, 0.28),
      new THREE.MeshStandardMaterial({ color: 0x8fe7ff, roughness: 0.18, metalness: 0.08, transparent: true, opacity: 0.92 })
    );
    cube.rotation.set(0.35, 0.15, -0.2);
    root.add(cube);
    root.position.set(from.x, 0.48, from.z);
    worldRoot.add(root);
    projectiles.add({
      root,
      startX: from.x,
      startZ: from.z,
      endX: to.x,
      endZ: to.z,
      startAt: elapsedSeconds,
      endAt: elapsedSeconds + ICE_THROW_ARC_SECONDS,
      dispose: () => {
        worldRoot.remove(root);
        disposeObject3D(root);
      },
    });
  }

  return {
    applySnapshot,
    dispose() {
      unsubscribe();
      stopFrame();
      for (const burst of bursts) {
        burst.dispose();
      }
      for (const entry of agents.values()) {
        worldRoot.remove(entry.visual.root);
        worldRoot.remove(entry.targetMarker);
        worldRoot.remove(entry.targetLine);
        if (entry.propMesh) {
          entry.visual.propAnchor.remove(entry.propMesh);
          disposeObject3D(entry.propMesh);
        }
        entry.visual.dispose();
        disposeTargetVisuals(entry);
      }
      for (const entry of objects.values()) {
        worldRoot.remove(entry.visual.root);
        entry.visual.dispose();
      }
      for (const projectile of projectiles) {
        projectile.dispose();
      }
      agents.clear();
      objects.clear();
      bursts.clear();
      projectiles.clear();
    },
    pickObject(ray) {
      const point = new THREE.Vector3();
      if (!ray.intersectPlane(hoverPlane, point)) {
        return null;
      }

      let best: { entry: ObjectEntry; score: number } | null = null;
      for (const entry of objects.values()) {
        if (entry.visual.root.visible === false) continue;
        const position = entry.visual.root.position;
        const dx = point.x - position.x;
        const dz = point.z - position.z;
        const distance = Math.sqrt(dx * dx + dz * dz);
        const hoverRadius = Math.max(0.65, entry.collisionRadius + 0.32);
        if (distance > hoverRadius) continue;
        const score = distance / hoverRadius;
        if (best == null || score < best.score) {
          best = { entry, score };
        }
      }

      if (!best) {
        return null;
      }

      return {
        id: best.entry.id,
        label: best.entry.label,
        description: best.entry.description,
      };
    },
    setHighlightedAgent(agentId) {
      highlightedAgent = agentId;
      for (const [id, entry] of agents) {
        entry.visual.setHighlighted(id === agentId);
      }
    },
    setHoveredObject(objectId) {
      hoveredObject = objectId;
      for (const [id, entry] of objects) {
        entry.visual.setHovered(id === objectId);
      }
    },
  };
}

function resolveIntentTarget(intent: AgentIntent | null, snap: WorldSnapshot): Vec2 | null {
  if (!intent) return null;
  switch (intent.kind) {
    case 'move_to':
      return intent.target;
    case 'go_to_object': {
      const target = snap.objects.find((object) => object.id === intent.objectId);
      return target?.position ?? null;
    }
    case 'approach_agent': {
      const target = snap.agents.find((agent) => agent.id === intent.agentId);
      return target?.position ?? null;
    }
    case 'sabotage': {
      const target = snap.agents.find((agent) => agent.id === intent.agentId);
      return target?.position ?? null;
    }
    case 'pick_up':
    case 'use':
    case 'deliver': {
      const target = snap.objects.find((object) => object.id === intent.objectId);
      return target?.position ?? null;
    }
    case 'wait':
    case 'drop':
      return null;
  }
}

function updateAgentGlyph(entry: AgentEntry, agent: SimulationAgentState, snap: WorldSnapshot): void {
  const glyph = entry.glyphOverride && entry.glyphOverrideUntil > 0
    ? entry.glyphOverride
    : activityGlyphFor(agent, snap);
  if (glyph) {
    entry.visual.emoji.setGlyph(glyph);
    entry.visual.emoji.setVisible(true);
  } else if (agent.emotion) {
    entry.visual.emoji.setGlyph(emotionGlyphFor(agent.emotion));
    entry.visual.emoji.setVisible(true);
  } else {
    entry.visual.emoji.setVisible(false);
  }
}

function activityGlyphFor(agent: SimulationAgentState, snap: WorldSnapshot): string | null {
  if (agent.holding === snap.game.bananaObjectId) return '🍌';
  if (agent.frozenUntilTick > snap.tick) return '⛄';
  if (agent.powerUp?.kind === 'bat') return '🏏';
  if (agent.powerUp?.kind === 'ice_cube') return '🧊';
  const intent = agent.intent;
  if (!intent) return null;
  switch (intent.kind) {
    case 'go_to_object':
    case 'move_to':
      return '🏃';
    case 'pick_up':
      return '✋';
    case 'deliver':
      return '🏁';
    case 'sabotage':
      return intent.method === 'bat' ? '🏏' : intent.method === 'ice_cube' ? '🧊' : '💥';
    case 'approach_agent':
      return '👀';
    case 'wait':
      return '⏳';
    case 'drop':
      return '💢';
    case 'use':
      return '✨';
  }
}

function glyphForIntent(intent: AgentIntent): string {
  switch (intent.kind) {
    case 'go_to_object':
    case 'move_to':
      return '🏃';
    case 'pick_up':
      return '✋';
    case 'deliver':
      return '🏁';
    case 'sabotage':
      return intent.method === 'bat' ? '🏏' : intent.method === 'ice_cube' ? '🧊' : '💥';
    case 'approach_agent':
      return '👀';
    case 'wait':
      return '⏳';
    case 'drop':
      return '💢';
    case 'use':
      return '✨';
  }
}

function applyBodyFlash(entry: AgentEntry, progress: number): void {
  const material = entry.visual.body.material as THREE.MeshStandardMaterial;
  const flash = entry.flashColor ?? entry.baseColor;
  material.color.copy(entry.baseColor).lerp(flash, 1 - progress);
}

function clearBodyFlash(entry: AgentEntry): void {
  const material = entry.visual.body.material as THREE.MeshStandardMaterial;
  material.color.copy(entry.baseColor);
  entry.flashColor = null;
}

function syncAgentProp(entry: AgentEntry, agent: SimulationAgentState): void {
  const nextKind = agent.powerUp?.kind ?? null;
  if (entry.propKind === nextKind) return;
  if (entry.propMesh) {
    entry.visual.propAnchor.remove(entry.propMesh);
    disposeObject3D(entry.propMesh);
    entry.propMesh = null;
    entry.propKind = null;
  }
  if (!nextKind) return;
  const mesh = createPowerUpProp(nextKind);
  entry.visual.propAnchor.add(mesh);
  entry.propMesh = mesh;
  entry.propKind = nextKind;
}

function createPowerUpProp(kind: 'bat' | 'ice_cube'): THREE.Object3D {
  if (kind === 'bat') {
    const bat = createBatMesh();
    bat.root.position.set(0.2, 0.13, 0.04);
    bat.root.rotation.set(0.22, -0.22, Math.PI / 3);
    bat.root.scale.setScalar(0.62);
    return bat.root;
  }
  const root = new THREE.Group();
  const cube = new THREE.Mesh(
    new THREE.BoxGeometry(0.2, 0.2, 0.2),
    new THREE.MeshStandardMaterial({ color: 0x8fe7ff, roughness: 0.2, metalness: 0.1, transparent: true, opacity: 0.9 })
  );
  cube.rotation.set(0.35, 0.15, -0.2);
  root.add(cube);
  return root;
}

function disposeObject3D(root: THREE.Object3D): void {
  root.traverse((obj) => {
    const mesh = obj as THREE.Mesh;
    mesh.geometry?.dispose?.();
    const material = mesh.material;
    if (Array.isArray(material)) material.forEach((entry) => entry.dispose());
    else material?.dispose?.();
  });
}

function setAgentTarget(entry: AgentEntry, target: Vec2 | null): void {
  if (!target) {
    entry.targetMarker.visible = false;
    entry.targetLine.visible = false;
    return;
  }
  entry.targetMarker.position.x = target.x;
  entry.targetMarker.position.z = target.z;
  entry.targetMarker.visible = true;
  entry.targetLine.visible = true;
}

function disposeTargetVisuals(entry: AgentEntry): void {
  entry.targetMarker.geometry.dispose();
  (entry.targetMarker.material as THREE.Material).dispose();
  entry.targetLine.geometry.dispose();
  (entry.targetLine.material as THREE.Material).dispose();
}

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}
