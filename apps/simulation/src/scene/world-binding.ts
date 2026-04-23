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
import type { WorldSnapshot } from '../runtime/types.js';
import { createAgentVisual, type AgentVisual } from '../render/agent-mesh.js';
import { createObjectVisual, type ObjectVisual } from '../render/object-mesh.js';
import { emotionGlyphFor } from '../render/emoji-billboard.js';
import { AGENT_COLOR_BY_ID } from '../scenarios/courtyard-snack.js';

const LERP_ALPHA = 0.22;

interface AgentEntry {
  readonly visual: AgentVisual;
  targetX: number;
  targetZ: number;
  targetHeading: number;
}

interface ObjectEntry {
  readonly visual: ObjectVisual;
  kind: string;
  targetX: number;
  targetZ: number;
  heldBy: string | null;
}

export interface WorldBinding {
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

  const ensureAgent = (id: string, name: string): AgentEntry => {
    let entry = agents.get(id);
    if (entry) return entry;
    const color = AGENT_COLOR_BY_ID.get(id) ?? '#c0c0c0';
    const visual = createAgentVisual(name, color);
    worldRoot.add(visual.root);
    entry = { visual, targetX: 0, targetZ: 0, targetHeading: 0 };
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
        entry.visual.dispose();
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
    }
  };
  const unsubscribe = bus.onAny(handleEvent);

  const stopFrame = onFrame(() => {
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
    }
    for (const entry of objects.values()) {
      if (entry.heldBy) {
        const holder = agents.get(entry.heldBy);
        if (holder) {
          entry.visual.root.position.x = holder.visual.root.position.x;
          entry.visual.root.position.z = holder.visual.root.position.z;
          return;
        }
      }
      const p = entry.visual.root.position;
      p.x += (entry.targetX - p.x) * LERP_ALPHA;
      p.z += (entry.targetZ - p.z) * LERP_ALPHA;
    }
  });

  return {
    dispose() {
      unsubscribe();
      stopFrame();
      for (const entry of agents.values()) {
        worldRoot.remove(entry.visual.root);
        entry.visual.dispose();
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
