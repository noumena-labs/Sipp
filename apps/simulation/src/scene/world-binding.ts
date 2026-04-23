//////////////////////////////////////////////////////////////////////////////
//
// scene/world-binding.ts
//
// - Mirrors the app-owned simulation runtime snapshots into the three.js scene.
// - On every `tick-end` event it diffs the snapshot against the visual
//   caches and updates positions, emotions, and held-object mounts.
// - Held objects are re-parented under their holding agent root so they
//   follow along naturally between ticks.
//
//////////////////////////////////////////////////////////////////////////////

import * as THREE from 'three';
import type {
  SimulationBus,
  SimulationEvent,
} from '../runtime/bus.js';
import type { AgentIntent, Vec2, WorldSnapshot } from '../runtime/types.js';
import { createAgentVisual, type AgentVisual } from '../render/agent-mesh.js';
import { createObjectVisual, type ObjectVisual } from '../render/object-mesh.js';
import { emotionGlyphFor } from '../render/emoji-billboard.js';
import { AGENT_COLOR_BY_ID } from '../scenarios/courtyard-snack.js';

const LERP_ALPHA = 0.22;
const THINKING_SECONDS = 1.2;
const PULSE_SECONDS = 0.8;
const THINKING_GLYPH = '\u{1F914}';

interface AgentEntry {
  readonly visual: AgentVisual;
  readonly targetMarker: THREE.Mesh;
  readonly targetLine: THREE.Line;
  targetX: number;
  targetZ: number;
  targetHeading: number;
  status: string;
  holding: string | null;
  pulseUntil: number;
  thinkingUntil: number;
}

interface ObjectEntry {
  readonly visual: ObjectVisual;
  kind: string;
  targetX: number;
  targetZ: number;
  heldBy: string | null;
}

export interface WorldBinding {
  applySnapshot(snapshot: WorldSnapshot): void;
  dispose(): void;
  setHighlightedAgent(agentId: string | null): void;
}

export function bindWorldToScene(
  bus: SimulationBus,
  worldRoot: THREE.Group,
  onFrame: (cb: (dt: number) => void) => () => void
): WorldBinding {
  const agents = new Map<string, AgentEntry>();
  const objects = new Map<string, ObjectEntry>();
  let highlightedAgent: string | null = null;
  let elapsedSeconds = 0;

  const ensureAgent = (id: string, name: string): AgentEntry => {
    let entry = agents.get(id);
    if (entry) return entry;
    const color = AGENT_COLOR_BY_ID.get(id) ?? '#c0c0c0';
    const visual = createAgentVisual(name, color);
    worldRoot.add(visual.root);
    const targetMarker = new THREE.Mesh(
      new THREE.RingGeometry(0.28, 0.38, 32),
      new THREE.MeshBasicMaterial({
        color: new THREE.Color(color),
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
        color: new THREE.Color(color),
        transparent: true,
        opacity: 0.42,
        depthWrite: false,
      })
    );
    targetLine.visible = false;
    worldRoot.add(targetLine);

    entry = {
      visual,
      targetMarker,
      targetLine,
      targetX: 0,
      targetZ: 0,
      targetHeading: 0,
      status: '',
      holding: null,
      pulseUntil: 0,
      thinkingUntil: 0,
    };
    agents.set(id, entry);
    if (highlightedAgent === id) visual.setHighlighted(true);
    return entry;
  };

  const ensureObject = (id: string, kind: string): ObjectEntry => {
    let entry = objects.get(id);
    if (entry) return entry;
    const visual = createObjectVisual(kind);
    worldRoot.add(visual.root);
    entry = { visual, kind, targetX: 0, targetZ: 0, heldBy: null };
    objects.set(id, entry);
    return entry;
  };

  const applySnapshot = (snap: WorldSnapshot): void => {
    // Sync agents.
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
      setAgentTarget(entry, resolveIntentTarget(a.intent, snap));
      if (a.emotion) {
        entry.visual.emoji.setGlyph(emotionGlyphFor(a.emotion));
        entry.visual.emoji.setVisible(true);
      } else {
        entry.visual.emoji.setVisible(false);
      }
    }
    for (const [id, entry] of agents) {
      if (!seenAgents.has(id)) {
        worldRoot.remove(entry.visual.root);
        worldRoot.remove(entry.targetMarker);
        worldRoot.remove(entry.targetLine);
        entry.visual.dispose();
        disposeTargetVisuals(entry);
        agents.delete(id);
      }
    }

    // Sync objects.
    const seenObjects = new Set<string>();
    for (const o of snap.objects) {
      seenObjects.add(o.id);
      const entry = ensureObject(o.id, o.kind);
      entry.targetX = o.position.x;
      entry.targetZ = o.position.z;
      if (entry.heldBy !== o.heldBy) {
        entry.heldBy = o.heldBy;
        entry.visual.setHeldBy(o.heldBy);
      }
    }
    for (const [id, entry] of objects) {
      if (!seenObjects.has(id)) {
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
        entry.thinkingUntil = elapsedSeconds + THINKING_SECONDS;
        entry.pulseUntil = Math.max(entry.pulseUntil, elapsedSeconds + PULSE_SECONDS);
        entry.visual.emoji.setGlyph(THINKING_GLYPH);
        entry.visual.emoji.setVisible(true);
      }
      return;
    }
    if (event.kind === 'agent-intent') {
      const entry = agents.get(event.agentId);
      if (entry) {
        entry.status = event.status;
        entry.pulseUntil = elapsedSeconds + PULSE_SECONDS;
      }
    }
  };
  const unsubscribe = bus.onAny(handleEvent);

  const stopFrame = onFrame((dt) => {
    elapsedSeconds += dt;
    for (const entry of agents.values()) {
      const pos = entry.visual.root.position;
      pos.x += (entry.targetX - pos.x) * LERP_ALPHA;
      pos.z += (entry.targetZ - pos.z) * LERP_ALPHA;
      const rot = entry.visual.root.rotation;
      const desired = -entry.targetHeading;
      let delta = desired - rot.y;
      while (delta > Math.PI) delta -= Math.PI * 2;
      while (delta < -Math.PI) delta += Math.PI * 2;
      rot.y += delta * LERP_ALPHA;
      const pulse = Math.max(0, entry.pulseUntil - elapsedSeconds) / PULSE_SECONDS;
      entry.visual.body.scale.set(1 + pulse * 0.18, 1 + pulse * 0.12, 1 + pulse * 0.18);
      if (entry.thinkingUntil > elapsedSeconds) {
        entry.visual.emoji.setGlyph(THINKING_GLYPH);
        entry.visual.emoji.setVisible(true);
      }
      if (entry.targetLine.visible) {
        const target = entry.targetMarker.position;
        entry.targetLine.geometry.setFromPoints([
          new THREE.Vector3(pos.x, 0.06, pos.z),
          new THREE.Vector3(target.x, 0.06, target.z),
        ]);
      }
    }
    for (const entry of objects.values()) {
      if (entry.heldBy) {
        const holder = agents.get(entry.heldBy);
        if (holder) {
          entry.visual.root.position.x = holder.visual.root.position.x;
          entry.visual.root.position.z = holder.visual.root.position.z;
          continue;
        }
      }
      const p = entry.visual.root.position;
      p.x += (entry.targetX - p.x) * LERP_ALPHA;
      p.z += (entry.targetZ - p.z) * LERP_ALPHA;
    }
  });

  return {
    applySnapshot,
    dispose() {
      unsubscribe();
      stopFrame();
      for (const entry of agents.values()) {
        worldRoot.remove(entry.visual.root);
        worldRoot.remove(entry.targetMarker);
        worldRoot.remove(entry.targetLine);
        entry.visual.dispose();
        disposeTargetVisuals(entry);
      }
      for (const entry of objects.values()) {
        worldRoot.remove(entry.visual.root);
        entry.visual.dispose();
      }
      agents.clear();
      objects.clear();
    },
    setHighlightedAgent(agentId) {
      highlightedAgent = agentId;
      for (const [id, entry] of agents) {
        entry.visual.setHighlighted(id === agentId);
      }
    },
  };
}

function resolveIntentTarget(intent: AgentIntent | null, snap: WorldSnapshot): Vec2 | null {
  if (!intent) return null;
  switch (intent.kind) {
    case 'move_to':
      return intent.target;
    case 'approach_agent': {
      const target = snap.agents.find((agent) => agent.id === intent.agentId);
      return target?.position ?? null;
    }
    case 'pick_up':
    case 'use': {
      const target = snap.objects.find((object) => object.id === intent.objectId);
      return target?.position ?? null;
    }
    case 'wait':
    case 'drop':
      return null;
  }
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
