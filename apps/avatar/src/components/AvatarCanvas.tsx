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
import { SpeechBubble } from '../scene/speech-bubble';

interface AvatarCanvasProps {
  readonly bus: ActionBus;
  readonly vrmUrl?: string;
  readonly speaking?: boolean;
  readonly bubbleText?: string;
  readonly bubblePending?: boolean;
  readonly status?: string;
}

export function AvatarCanvas({
  bus,
  vrmUrl,
  speaking = false,
  bubbleText = '',
  bubblePending = false,
  status,
}: AvatarCanvasProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const bindingRef = useRef<ThreeVRMBinding | null>(null);
  const bubbleRef = useRef<SpeechBubble | null>(null);
  const speakingRef = useRef(speaking);
  const bubbleTextRef = useRef(bubbleText);
  const bubblePendingRef = useRef(bubblePending);

  useEffect(() => {
    speakingRef.current = speaking;
    bindingRef.current?.setSpeaking(speaking);
  }, [speaking]);

  useEffect(() => {
    bubbleTextRef.current = bubbleText;
    bubblePendingRef.current = bubblePending;
    bubbleRef.current?.setContent(bubbleText, bubblePending);
  }, [bubblePending, bubbleText]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }
    let sceneHandle: SceneHandle | null = null;
    let avatar: LoadedAvatar | null = null;
    let binding: ThreeVRMBinding | null = null;
    let speechBubble: SpeechBubble | null = null;
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
        loaded?.dispose();
        return;
      }
      if (!loaded) {
        if (bubbleRef.current) {
          bubbleRef.current.dispose();
          bubbleRef.current = null;
        }
        if (bindingRef.current) {
          bindingRef.current.dispose();
          bindingRef.current = null;
        }
        return;
      }
      avatar = loaded;
      sceneHandle.avatarRoot.add(loaded.root);
      sceneHandle.focusAvatar(loaded.layout);
      binding = new ThreeVRMBinding(bus, loaded);
      binding.setSpeaking(speakingRef.current);
      bindingRef.current = binding;
      speechBubble = new SpeechBubble({
        scene: sceneHandle.scene,
        camera: sceneHandle.camera,
        avatar: loaded,
      });
      speechBubble.setContent(bubbleTextRef.current, bubblePendingRef.current);
      bubbleRef.current = speechBubble;
      const offFrame = sceneHandle.onFrame((delta) => {
        binding?.tick(delta);
        speechBubble?.tick(delta);
      });
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
        if (bindingRef.current === binding) {
          bindingRef.current = null;
        }
      }
      if (speechBubble) {
        speechBubble.dispose();
        if (bubbleRef.current === speechBubble) {
          bubbleRef.current = null;
        }
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
