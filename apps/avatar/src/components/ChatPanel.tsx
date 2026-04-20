//////////////////////////////////////////////////////////////////////////////
//
// ChatPanel.tsx
//
// - Renders the conversation log and an input box. Prose is rendered
//   incrementally as it arrives; action tags are surfaced as inline chips
//   so the user can see which gestures the model triggered.
//
//////////////////////////////////////////////////////////////////////////////

import { useEffect, useRef, useState } from 'react';
import type { FormEvent } from 'react';

export interface ChatMessage {
  readonly id: string;
  readonly role: 'user' | 'assistant';
  readonly text: string;
  readonly actions: ReadonlyArray<{ name: string; args: Readonly<Record<string, unknown>> }>;
  readonly pending?: boolean;
}

interface ChatPanelProps {
  readonly messages: readonly ChatMessage[];
  readonly onSend: (text: string) => void;
  readonly disabled?: boolean;
}

export function ChatPanel({ messages, onSend, disabled }: ChatPanelProps) {
  const [draft, setDraft] = useState('');
  const logRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const log = logRef.current;
    if (log) {
      log.scrollTop = log.scrollHeight;
    }
  }, [messages]);

  const handleSubmit = (event: FormEvent): void => {
    event.preventDefault();
    const trimmed = draft.trim();
    if (!trimmed || disabled) {
      return;
    }
    onSend(trimmed);
    setDraft('');
  };

  return (
    <div className="chat-panel">
      <div className="chat-log" ref={logRef}>
        {messages.map((msg) => (
          <div key={msg.id} className={`chat-entry ${msg.role}`}>
            {msg.actions.map((action, index) => (
              <span key={index} className="action-chip" title={JSON.stringify(action.args)}>
                {action.name}
              </span>
            ))}
            {msg.text}
            {msg.pending ? <span className="cursor">▍</span> : null}
          </div>
        ))}
      </div>
      <form className="chat-input" onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder={disabled ? 'Load a model to start chatting...' : 'Say something...'}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          disabled={disabled}
        />
        <button type="submit" disabled={disabled || draft.trim().length === 0}>
          Send
        </button>
      </form>
    </div>
  );
}
