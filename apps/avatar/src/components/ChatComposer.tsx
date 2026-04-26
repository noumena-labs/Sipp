//////////////////////////////////////////////////////////////////////////////
//
// ChatComposer.tsx
//
// - Bottom-docked prompt composer for the avatar stage.
//
//////////////////////////////////////////////////////////////////////////////

import { useEffect, useRef, useState } from 'react';
import type { FormEvent, KeyboardEvent } from 'react';

interface ChatComposerProps {
  readonly onSend: (text: string) => void;
  readonly disabled?: boolean;
  readonly characterName?: string;
}

export function ChatComposer({
  onSend,
  disabled,
  characterName = 'your companion',
}: ChatComposerProps) {
  const [draft, setDraft] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) {
      return;
    }
    textarea.style.height = '0px';
    textarea.style.height = `${Math.min(textarea.scrollHeight, 156)}px`;
  }, [draft]);

  const submitDraft = (): void => {
    const trimmed = draft.trim();
    if (!trimmed || disabled) {
      return;
    }
    onSend(trimmed);
    setDraft('');
  };

  const handleSubmit = (event: FormEvent): void => {
    event.preventDefault();
    submitDraft();
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>): void => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      submitDraft();
    }
  };

  return (
    <form className="chat-composer glass-panel" onSubmit={handleSubmit}>
      <div className="chat-composer-header">
        <div>
          <span className="panel-eyebrow">Text Chat</span>
          <div className="chat-composer-title">Talk to {characterName}</div>
        </div>
        <div className="chat-composer-hint">Enter to send</div>
      </div>
      <div className="chat-composer-row">
        <textarea
          ref={textareaRef}
          value={draft}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={handleKeyDown}
          rows={1}
          disabled={disabled}
          placeholder={
            disabled
              ? 'Load a model to start chatting...'
              : `Ask ${characterName} anything...`
          }
        />
        <button type="submit" disabled={disabled || draft.trim().length === 0}>
          Send
        </button>
      </div>
    </form>
  );
}
