//////////////////////////////////////////////////////////////////////////////
//
// AvatarCanvas.tsx
//
// - Mounts the three.js renderer into a div and wires the supplied
//   CharacterEventBus to a ThreeVRMBinding. Handles container resize via
//   ResizeObserver.
//
//////////////////////////////////////////////////////////////////////////////

import { useEffect, useRef, useSyncExternalStore } from 'react';
import type { CharacterEventBus } from '@noumena-labs/cogentlm-browser/character';
import { createScene, type SceneHandle } from '../scene/scene';
import { loadAvatar, type LoadedAvatar } from '../scene/vrm-loader';
import { ThreeVRMBinding } from '../bindings/three-vrm-binding';
import { SpeechBubble } from '../scene/speech-bubble';
import type { AvatarRenderAssets } from '../characters/render-assets';
import type { ChatMessage } from './chat-types';

interface AvatarCanvasHmrState {
  sceneEpoch: number;
  listeners: Set<() => void>;
}

const avatarCanvasHmrData = import.meta.hot?.data as
  | { avatarCanvasSceneEpoch?: number }
  | undefined;
const avatarCanvasHmrState: AvatarCanvasHmrState = {
  sceneEpoch: avatarCanvasHmrData?.avatarCanvasSceneEpoch ?? 0,
  listeners: new Set(),
};

function subscribeAvatarCanvasSceneEpoch(listener: () => void): () => void {
  avatarCanvasHmrState.listeners.add(listener);
  return () => {
    avatarCanvasHmrState.listeners.delete(listener);
  };
}

function getAvatarCanvasSceneEpoch(): number {
  return avatarCanvasHmrState.sceneEpoch;
}

function bumpAvatarCanvasSceneEpoch(): void {
  avatarCanvasHmrState.sceneEpoch += 1;
  avatarCanvasHmrState.listeners.forEach((listener) => listener());
}

if (import.meta.hot) {
  // Remount the imperative scene owner when hot updates land in helpers whose
  // instances are created inside the long-lived setup effect below.
  import.meta.hot.accept(
    [
      '../bindings/three-vrm-binding',
      '../scene/scene',
      '../scene/speech-bubble',
      '../scene/vrm-loader',
    ],
    () => {
      bumpAvatarCanvasSceneEpoch();
    }
  );

  import.meta.hot.dispose((data) => {
    data.avatarCanvasSceneEpoch = avatarCanvasHmrState.sceneEpoch;
    avatarCanvasHmrState.listeners.clear();
  });
}

interface AvatarCanvasProps {
  readonly bus: CharacterEventBus;
  readonly renderAssets?: AvatarRenderAssets;
  readonly actionNames?: readonly string[];
  readonly speaking?: boolean;
  readonly bubbleText?: string;
  readonly bubblePending?: boolean;
  readonly bubbleActions?: ChatMessage['actions'];
  readonly characterName?: string;
  readonly status?: string;
  readonly onReady?: () => void;
  readonly onError?: (message: string) => void;
}

export function AvatarCanvas({
  bus,
  renderAssets,
  actionNames = [],
  speaking = false,
  bubbleText = '',
  bubblePending = false,
  bubbleActions = [],
  characterName = 'Aria',
  status,
  onReady,
  onError,
}: AvatarCanvasProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const bindingRef = useRef<ThreeVRMBinding | null>(null);
  const bubbleRef = useRef<SpeechBubble | null>(null);
  const sceneEpoch = useSyncExternalStore(
    subscribeAvatarCanvasSceneEpoch,
    getAvatarCanvasSceneEpoch,
    getAvatarCanvasSceneEpoch
  );
  const speakingRef = useRef(speaking);
  const bubbleTextRef = useRef(bubbleText);
  const bubblePendingRef = useRef(bubblePending);
  const bubbleActionsRef = useRef(bubbleActions);
  const onReadyRef = useRef(onReady);
  const onErrorRef = useRef(onError);

  useEffect(() => {
    onReadyRef.current = onReady;
  }, [onReady]);

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
    bubbleActionsRef.current = bubbleActions;
    // When the prop changes (e.g. at turn-end), we sync the finalized text.
    bubbleRef.current?.setContent(bubbleText, bubblePending, bubbleActions);
  }, [bubbleActions, bubblePending, bubbleText]);

  useEffect(() => {
    let liveText = '';
    const off = bus.onAny((event) => {
      if (event.kind === 'prose') {
        liveText += event.text;
        bubbleRef.current?.setContent(
          liveText,
          true,
          bubbleActionsRef.current
        );
      } else if (event.kind === 'turn-start') {
        liveText = '';
        bubbleRef.current?.setContent('', true, []);
      }
    });
    return () => off();
  }, [bus]);


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
        binding = new ThreeVRMBinding(
          bus,
          sceneHandle.scene,
          sceneHandle.camera,
          loaded,
          renderAssets
        );
        await binding.init(actionNames);
        if (cancelled) {
          return;
        }
        binding.setSpeaking(speakingRef.current);
        bindingRef.current = binding;
        speechBubble = new SpeechBubble({
          scene: sceneHandle.scene,
          camera: sceneHandle.camera,
          avatar: loaded,
          speakerName: characterName,
        });
        speechBubble.setContent(
          bubbleTextRef.current,
          bubblePendingRef.current,
          bubbleActionsRef.current
        );
        bubbleRef.current = speechBubble;
        const offFrame = sceneHandle.onFrame((delta) => {
          binding?.tick(delta);
          speechBubble?.tick(delta);
        });
        (binding as unknown as { _off: () => void })._off = offFrame;
        onReadyRef.current?.();
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
  }, [actionNames, bus, characterName, renderAssets, sceneEpoch]);

  return (
    <div className="avatar-canvas" ref={containerRef}>
      {status ? <div className="overlay">{status}</div> : null}
    </div>
  );
}
