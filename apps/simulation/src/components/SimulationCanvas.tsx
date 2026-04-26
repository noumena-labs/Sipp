//////////////////////////////////////////////////////////////////////////////
//
// components/SimulationCanvas.tsx
//
// - Hosts the three.js simulation scene and binds it to the director
//   bus. Mount once and reuse; the binding is created here and disposed
//   when the component unmounts.
//
//////////////////////////////////////////////////////////////////////////////

import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import type { SimulationBus } from '../runtime/bus.js';
import type { WorldBounds, WorldSnapshot } from '../runtime/types.js';
import { createSimulationScene, type SimulationSceneHandle } from '../scene/scene.js';
import {
  bindWorldToScene,
  type HoveredSceneObject,
  type WorldBinding,
} from '../scene/world-binding.js';

export interface SimulationCanvasProps {
  readonly bus: SimulationBus;
  readonly bounds: WorldBounds;
  readonly highlightedAgentId: string | null;
  readonly highlightedObjectId?: string | null;
  readonly onBackgroundClick?: () => void;
  readonly onHoverObject?: (object: HoveredSceneObject | null) => void;
  readonly snapshot: WorldSnapshot | null;
}

export function SimulationCanvas(props: SimulationCanvasProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const sceneRef = useRef<SimulationSceneHandle | null>(null);
  const bindingRef = useRef<WorldBinding | null>(null);
  const raycasterRef = useRef(new THREE.Raycaster());
  const pointerRef = useRef(new THREE.Vector2());
  const hoveredObjectIdRef = useRef<string | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const scene = createSimulationScene(container, props.bounds.halfExtent);
    sceneRef.current = scene;
    const binding = bindWorldToScene(props.bus, scene.worldRoot, scene.onFrame);
    bindingRef.current = binding;

    const updateHover = (event: PointerEvent): HoveredSceneObject | null => {
      const currentScene = sceneRef.current;
      const currentBinding = bindingRef.current;
      const currentContainer = containerRef.current;
      if (!currentScene || !currentBinding || !currentContainer) {
        return null;
      }

      const rect = currentContainer.getBoundingClientRect();
      pointerRef.current.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointerRef.current.y = -(((event.clientY - rect.top) / rect.height) * 2 - 1);
      raycasterRef.current.setFromCamera(pointerRef.current, currentScene.camera);
      const hovered = currentBinding.pickObject(raycasterRef.current.ray);
      const nextHoveredId = hovered?.id ?? null;
      if (hoveredObjectIdRef.current !== nextHoveredId) {
        hoveredObjectIdRef.current = nextHoveredId;
        currentBinding.setHoveredObject(nextHoveredId);
        props.onHoverObject?.(hovered);
      }
      return hovered;
    };

    const clearHover = (): void => {
      if (hoveredObjectIdRef.current == null) return;
      hoveredObjectIdRef.current = null;
      binding.setHoveredObject(null);
      props.onHoverObject?.(null);
    };

    const handlePointerMove = (event: PointerEvent): void => {
      updateHover(event);
    };

    const handlePointerLeave = (): void => {
      clearHover();
    };

    const handleClick = (event: MouseEvent): void => {
      const hovered = updateHover(event as unknown as PointerEvent);
      if (!hovered) {
        props.onBackgroundClick?.();
      }
    };

    scene.renderer.domElement.addEventListener('pointermove', handlePointerMove);
    scene.renderer.domElement.addEventListener('pointerleave', handlePointerLeave);
    scene.renderer.domElement.addEventListener('click', handleClick);

    const resize = (): void => {
      scene.setSize(container.clientWidth, container.clientHeight);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(container);

    return () => {
      ro.disconnect();
      scene.renderer.domElement.removeEventListener('pointermove', handlePointerMove);
      scene.renderer.domElement.removeEventListener('pointerleave', handlePointerLeave);
      scene.renderer.domElement.removeEventListener('click', handleClick);
      binding.dispose();
      scene.dispose();
      sceneRef.current = null;
      bindingRef.current = null;
      hoveredObjectIdRef.current = null;
    };
    // We intentionally mount once; bus + bounds are stable per session.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    bindingRef.current?.setHighlightedAgent(props.highlightedAgentId);
  }, [props.highlightedAgentId]);

  useEffect(() => {
    bindingRef.current?.setHighlightedObject(props.highlightedObjectId ?? null);
  }, [props.highlightedObjectId]);

  useEffect(() => {
    return () => props.onHoverObject?.(null);
  }, [props.onHoverObject]);

  useEffect(() => {
    if (props.snapshot) {
      bindingRef.current?.applySnapshot(props.snapshot);
    }
  }, [props.snapshot]);

  return <div ref={containerRef} className="sim-canvas" />;
}
