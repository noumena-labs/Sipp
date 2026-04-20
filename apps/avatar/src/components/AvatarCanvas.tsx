//////////////////////////////////////////////////////////////////////////////
//
// AvatarCanvas.tsx
//
// - Mounts the three.js renderer into a div and wires the supplied
//   ActionBus to a ThreeVRMBinding. Handles container resize via
//   ResizeObserver.
//
//////////////////////////////////////////////////////////////////////////////

import { useEffect, useRef } from 'react';
import type { ActionBus } from 'cogent-engine/character';
import { createScene, type SceneHandle } from '../scene/scene';
import { loadAvatar, type LoadedAvatar } from '../scene/vrm-loader';
import { ThreeVRMBinding } from '../bindings/three-vrm-binding';

interface AvatarCanvasProps {
  readonly bus: ActionBus;
  readonly vrmUrl?: string;
  readonly status?: string;
}

export function AvatarCanvas({ bus, vrmUrl, status }: AvatarCanvasProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }
    let sceneHandle: SceneHandle | null = null;
    let avatar: LoadedAvatar | null = null;
    let binding: ThreeVRMBinding | null = null;
    let cancelled = false;

    sceneHandle = createScene(container);
    const rect = container.getBoundingClientRect();
    sceneHandle.setSize(rect.width, rect.height);

    const resizeObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        sceneHandle?.setSize(width, height);
      }
    });
    resizeObserver.observe(container);

    (async () => {
      const loaded = await loadAvatar(vrmUrl);
      if (cancelled || !sceneHandle) {
        loaded.dispose();
        return;
      }
      avatar = loaded;
      sceneHandle.avatarRoot.add(loaded.root);
      binding = new ThreeVRMBinding(bus, loaded);
      const offFrame = sceneHandle.onFrame((delta) => binding?.tick(delta));
      // Stash the off-frame on the binding so dispose can call it.
      (binding as unknown as { _off: () => void })._off = offFrame;
    })();

    return () => {
      cancelled = true;
      resizeObserver.disconnect();
      if (binding) {
        const off = (binding as unknown as { _off?: () => void })._off;
        off?.();
        binding.dispose();
      }
      if (avatar) {
        avatar.dispose();
      }
      sceneHandle?.dispose();
    };
  }, [bus, vrmUrl]);

  return (
    <div className="avatar-canvas" ref={containerRef}>
      {status ? <div className="overlay">{status}</div> : null}
    </div>
  );
}
