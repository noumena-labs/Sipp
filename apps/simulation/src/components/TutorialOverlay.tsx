import { useLayoutEffect, useRef, useState } from 'react';

export type TutorialCardPlacement = 'center' | 'right' | 'left' | 'top' | 'bottom';

export interface TutorialOverlayProps {
  readonly open: boolean;
  readonly title: string;
  readonly body: string;
  readonly target: string | null;
  readonly placement: TutorialCardPlacement;
  readonly stepIndex: number;
  readonly stepCount: number;
  readonly neverShowAgain: boolean;
  readonly onNext: () => void;
  readonly onDismiss: () => void;
  readonly onNeverShowAgainChange: (value: boolean) => void;
}

interface CardPosition {
  readonly left: number;
  readonly top: number;
}

const VIEWPORT_PADDING = 16;
const TARGET_GAP = 18;

export function TutorialOverlay(props: TutorialOverlayProps) {
  const cardRef = useRef<HTMLDivElement | null>(null);
  const [position, setPosition] = useState<CardPosition | null>(null);
  const done = props.stepIndex >= props.stepCount - 1;

  useLayoutEffect(() => {
    if (!props.open) {
      setPosition(null);
      return;
    }

    const card = cardRef.current;
    if (!card) {
      return;
    }

    const target = props.target != null
      ? document.querySelector<HTMLElement>(`[data-tutorial-target="${props.target}"]`)
      : null;

    const updatePosition = (): void => {
      const cardRect = card.getBoundingClientRect();
      const viewportWidth = window.innerWidth;
      const viewportHeight = window.innerHeight;
      const targetRect = target?.getBoundingClientRect() ?? null;
      setPosition(resolveCardPosition(cardRect, targetRect, viewportWidth, viewportHeight, props.placement));
    };

    updatePosition();

    const resizeObserver = target != null ? new ResizeObserver(updatePosition) : null;
    resizeObserver?.observe(target);
    window.addEventListener('resize', updatePosition);

    return () => {
      resizeObserver?.disconnect();
      window.removeEventListener('resize', updatePosition);
    };
  }, [props.open, props.placement, props.stepIndex, props.target]);

  if (!props.open) {
    return null;
  }

  return (
    <>
      <div className="tutorial-backdrop" aria-hidden="true" />
      <div
        ref={cardRef}
        className="tutorial-card glass-panel"
        style={position == null ? undefined : { left: `${position.left}px`, top: `${position.top}px` }}
        role="dialog"
        aria-modal="true"
        aria-labelledby="tutorial-title"
      >
        <div className="tutorial-card-head">
          <span className="panel-eyebrow">Tutorial</span>
          <span className="tutorial-step-count">{props.stepIndex + 1} / {props.stepCount}</span>
        </div>
        <h2 id="tutorial-title" className="tutorial-title">{props.title}</h2>
        <p className="tutorial-copy">{props.body}</p>
        <label className="tutorial-checkbox-row">
          <input
            type="checkbox"
            checked={props.neverShowAgain}
            onChange={(event) => props.onNeverShowAgainChange(event.target.checked)}
          />
          <span>Never show again</span>
        </label>
        <div className="tutorial-actions">
          <button type="button" className="tutorial-dismiss" onClick={props.onDismiss}>
            Dismiss
          </button>
          <button type="button" className="tutorial-next" onClick={props.onNext}>
            {done ? 'Done' : 'Next'}
          </button>
        </div>
      </div>
    </>
  );
}

function resolveCardPosition(
  cardRect: DOMRect,
  targetRect: DOMRect | null,
  viewportWidth: number,
  viewportHeight: number,
  placement: TutorialCardPlacement
): CardPosition {
  if (targetRect == null || placement === 'center') {
    return clampCardPosition(
      (viewportWidth - cardRect.width) / 2,
      (viewportHeight - cardRect.height) / 2,
      cardRect,
      viewportWidth,
      viewportHeight
    );
  }

  const preferred = placeCard(targetRect, cardRect, placement);
  const horizontalFallback = placement === 'right'
    ? placeCard(targetRect, cardRect, 'left')
    : placement === 'left'
      ? placeCard(targetRect, cardRect, 'right')
      : preferred;
  const verticalFallback = placement === 'top'
    ? placeCard(targetRect, cardRect, 'bottom')
    : placement === 'bottom'
      ? placeCard(targetRect, cardRect, 'top')
      : preferred;

  const candidates = [preferred, horizontalFallback, verticalFallback];
  const visibleCandidate = candidates.find((candidate) => fitsViewport(candidate.left, candidate.top, cardRect, viewportWidth, viewportHeight));
  const chosen = visibleCandidate ?? preferred;
  return clampCardPosition(chosen.left, chosen.top, cardRect, viewportWidth, viewportHeight);
}

function placeCard(targetRect: DOMRect, cardRect: DOMRect, placement: Exclude<TutorialCardPlacement, 'center'>): CardPosition {
  switch (placement) {
    case 'right':
      return {
        left: targetRect.right + TARGET_GAP,
        top: targetRect.top + (targetRect.height - cardRect.height) / 2,
      };
    case 'left':
      return {
        left: targetRect.left - cardRect.width - TARGET_GAP,
        top: targetRect.top + (targetRect.height - cardRect.height) / 2,
      };
    case 'top':
      return {
        left: targetRect.left + (targetRect.width - cardRect.width) / 2,
        top: targetRect.top - cardRect.height - TARGET_GAP,
      };
    case 'bottom':
      return {
        left: targetRect.left + (targetRect.width - cardRect.width) / 2,
        top: targetRect.bottom + TARGET_GAP,
      };
  }
}

function fitsViewport(
  left: number,
  top: number,
  cardRect: DOMRect,
  viewportWidth: number,
  viewportHeight: number
): boolean {
  return left >= VIEWPORT_PADDING
    && top >= VIEWPORT_PADDING
    && left + cardRect.width <= viewportWidth - VIEWPORT_PADDING
    && top + cardRect.height <= viewportHeight - VIEWPORT_PADDING;
}

function clampCardPosition(
  left: number,
  top: number,
  cardRect: DOMRect,
  viewportWidth: number,
  viewportHeight: number
): CardPosition {
  return {
    left: clamp(left, VIEWPORT_PADDING, viewportWidth - cardRect.width - VIEWPORT_PADDING),
    top: clamp(top, VIEWPORT_PADDING, viewportHeight - cardRect.height - VIEWPORT_PADDING),
  };
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}
