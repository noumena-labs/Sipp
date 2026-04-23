//////////////////////////////////////////////////////////////////////////////
//
// components/EventLog.tsx
//
// - Tail of the most recent simulation bus events, filtered to the
//   human-interesting kinds.
//
//////////////////////////////////////////////////////////////////////////////

export interface EventLogEntry {
  readonly id: number;
  readonly tick: number;
  readonly text: string;
  readonly kind: 'note' | 'intent' | 'conflict' | 'decision' | 'query';
}

export interface EventLogProps {
  readonly entries: readonly EventLogEntry[];
}

export function EventLog(props: EventLogProps) {
  return (
    <div className="event-log glass-panel">
      <div className="panel-eyebrow">Event log</div>
      <ul>
        {props.entries.slice(-40).reverse().map((e) => (
          <li key={e.id} className={`event-${e.kind}`}>
            <span className="tick">#{e.tick}</span>
            <span className="text">{e.text}</span>
          </li>
        ))}
        {props.entries.length === 0 ? (
          <li className="event-empty">(no events yet — start the simulation)</li>
        ) : null}
      </ul>
    </div>
  );
}
