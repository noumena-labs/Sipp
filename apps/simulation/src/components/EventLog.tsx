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
  readonly collapsed: boolean;
  readonly onToggle: () => void;
}

export function EventLog(props: EventLogProps) {
  const visibleEntries = props.collapsed
    ? props.entries.slice(-5).reverse()
    : props.entries.slice(-40).reverse();

  return (
    <div className={`event-log glass-panel${props.collapsed ? ' collapsed' : ' expanded'}`}>
      <button type="button" className="event-log-toggle" onClick={props.onToggle}>
        <span className="panel-eyebrow">Event log</span>
        <span className="event-log-toggle-label">{props.collapsed ? 'Expand' : 'Collapse'}</span>
      </button>
      <ul>
        {visibleEntries.map((e) => (
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
