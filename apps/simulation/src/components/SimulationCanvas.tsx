//////////////////////////////////////////////////////////////////////////////
//
// components/SimulationCanvas.tsx
//
// - Hosts the three.js simulation scene and binds it to the orchestrator
//   bus. Mount once and reuse; the binding is created here and disposed
//   when the component unmounts.
//
//////////////////////////////////////////////////////////////////////////////

import { useEffect, useRef } from 'react';
import type { SimulationBus } from '../runtime/bus.js';
import type { WorldBounds } from '../runtime/types.js';
import { createSimulationScene, type SimulationSceneHandle } from '../scene/scene.js';
import { bindWorldToScene, type WorldBinding } from '../scene/world-binding.js';

export interface SimulationCanvasProps {
  readonly bus: SimulationBus;
  readonly bounds: WorldBounds;
  readonly highlightedAgentId: string | null;
}

export function SimulationCanvas(props: SimulationCanvasProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const sceneRef = useRef<SimulationSceneHandle | null>(null);
  const bindingRef = useRef<WorldBinding | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const scene = createSimulationScene(container, props.bounds.halfExtent);
    sceneRef.current = scene;
    const binding = bindWorldToScene(props.bus, scene.worldRoot, scene.onFrame);
    bindingRef.current = binding;

    const resize = (): void => {
      scene.setSize(container.clientWidth, container.clientHeight);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(container);

    return () => {
      ro.disconnect();
      binding.dispose();
      scene.dispose();
      sceneRef.current = null;
      bindingRef.current = null;
    };
    // We intentionally mount once; bus + bounds are stable per session.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    bindingRef.current?.setHighlightedAgent(props.highlightedAgentId);
  }, [props.highlightedAgentId]);

  return <div ref={containerRef} className="sim-canvas" />;
}
