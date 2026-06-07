//////////////////////////////////////////////////////////////////////////////
//
// TranscriptDrawer.tsx
//
// - Slide-out transcript containing the full conversation history.
//
//////////////////////////////////////////////////////////////////////////////

import { useEffect, useRef, useState } from 'react';
import type { CharacterEventBus } from '@noumena-labs/cogentlm/character';
import type { ChatMessage } from './chat-types';

interface TranscriptDrawerProps {
  readonly open: boolean;
  readonly messages: readonly ChatMessage[];
  readonly bus: CharacterEventBus;
  readonly onClose: () => void;
  readonly characterName?: string;
  readonly id?: string;
}

export function TranscriptDrawer({
  open,
  messages,
  bus,
  onClose,
  characterName = 'Companion',
  id = 'transcript-drawer',
}: TranscriptDrawerProps) {
  const logRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    const log = logRef.current;
    if (!log) {
      return;
    }
    log.scrollTop = log.scrollHeight;
  }, [messages, open]);

  return (
    <>
      <aside
        id={id}
        className={`transcript-drawer glass-panel${open ? ' open' : ''}`}
        aria-hidden={!open}
      >
        <div className="transcript-header">
          <div>
            <span className="panel-eyebrow">Conversation</span>
            <h2>Chat transcript</h2>
          </div>
          <button type="button" className="secondary-button" onClick={onClose}>
            Minimize
          </button>
        </div>

        <div className="transcript-log" ref={logRef}>
          {messages.length === 0 ? (
            <div className="transcript-empty">
              Full replies will collect here once the conversation starts.
            </div>
          ) : (
            messages.map((message, index) => {
              const isLatest = index === messages.length - 1;
              return (
                <article key={message.id} className={`transcript-entry ${message.role}`}>
                  <div className="transcript-meta-row">
                    <div className="transcript-meta">
                      <span className="transcript-role">
                        {message.role === 'user' ? 'You' : characterName}
                      </span>
                      {message.pending ? <span className="transcript-pending">Typing</span> : null}
                    </div>
                    {message.actions.length > 0 ? (
                      <div className="transcript-actions">
                        {message.actions.map((action, index) => (
                          <span key={`${message.id}-${index}`} className="action-chip" title={action.id}>
                            {action.label}
                          </span>
                        ))}
                      </div>
                    ) : null}
                  </div>
                  <div className="transcript-text">
                    <MessageContent
                      message={message}
                      bus={bus}
                      active={isLatest && message.pending === true}
                    />
                    {message.pending ? <span className="cursor"></span> : null}
                  </div>
                </article>
              );
            })
          )}
        </div>
      </aside>
    </>
  );
}

function MessageContent({
  message,
  bus,
  active,
}: {
  message: ChatMessage;
  bus: CharacterEventBus;
  active: boolean;
}) {
  const [liveText, setLiveText] = useState('');

  useEffect(() => {
    if (!active) {
      setLiveText('');
      return;
    }

    const off = bus.onAny((event) => {
      if (event.kind === 'prose') {
        setLiveText((prev) => prev + event.text);
      } else if (event.kind === 'turn-start') {
        setLiveText('');
      }
    });

    return () => off();
  }, [active, bus]);

  const displayedText = active ? liveText : message.text;
  const fallbackText = message.pending ? '...' : '[No visible response generated.]';

  return <>{displayedText.trim().length > 0 ? displayedText : fallbackText}</>;
}
