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
import type { AvatarRenderAssets } from '../characters/render-assets';

interface AvatarCanvasProps {
  readonly bus: ActionBus;
  readonly renderAssets?: AvatarRenderAssets;
  readonly actionNames?: readonly string[];
  readonly speaking?: boolean;
  readonly bubbleText?: string;
  readonly bubblePending?: boolean;
  readonly status?: string;
  readonly onError?: (message: string) => void;
}

export function AvatarCanvas({
  bus,
  renderAssets,
  actionNames = [],
  speaking = false,
  bubbleText = '',
  bubblePending = false,
  status,
  onError,
}: AvatarCanvasProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const bindingRef = useRef<ThreeVRMBinding | null>(null);
  const bubbleRef = useRef<SpeechBubble | null>(null);
  const speakingRef = useRef(speaking);
  const bubbleTextRef = useRef(bubbleText);
  const bubblePendingRef = useRef(bubblePending);
  const onErrorRef = useRef(onError);

  useEffect(() => {
    onErrorRef.current = onError;
  }, [onError]);

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
    if (!container || !renderAssets) {
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

    void (async () => {
      try {
        const loaded = await loadAvatar(renderAssets.vrmUrl);
        if (cancelled || !sceneHandle) {
          loaded.dispose();
          return;
        }

        avatar = loaded;
        sceneHandle.avatarRoot.add(loaded.root);
        sceneHandle.focusAvatar(loaded.layout);
        binding = new ThreeVRMBinding(bus, loaded, renderAssets);
        await binding.init(actionNames);
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
        (binding as unknown as { _off: () => void })._off = offFrame;
      } catch (error) {
        if (!cancelled) {
          if (binding) {
            const off = (binding as unknown as { _off?: () => void })._off;
            off?.();
            binding.dispose();
            binding = null;
            bindingRef.current = null;
          }
          if (speechBubble) {
            speechBubble.dispose();
            speechBubble = null;
            bubbleRef.current = null;
          }
          if (avatar) {
            avatar.dispose();
            avatar = null;
          }
          sceneHandle?.avatarRoot.clear();
          onErrorRef.current?.((error as Error).message);
        }
      }
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
  }, [actionNames, bus, renderAssets]);

  return (
    <div className="avatar-canvas" ref={containerRef}>
      {status ? <div className="overlay">{status}</div> : null}
    </div>
  );
}
