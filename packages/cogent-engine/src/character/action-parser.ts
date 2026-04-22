//////////////////////////////////////////////////////////////////////////////
//
// action-parser.ts
//
// - Incremental parser that turns a stream of text chunks (as produced by
//   the grammar defined in action-grammar.ts) into two kinds of events:
//     * `{ kind: 'prose', text }` — a run of plain characters;
//     * `{ kind: 'action', name }` — a recognised bracketed cue.
//
// - Must be tolerant of chunk boundaries splitting in the middle of a cue.
//   Internal buffering is retained across calls to {@link consume}.
//
// - Cues are recognised by looking up the bracketed label in the cue map
//   built from the character's ActionSchema. Unknown labels are surfaced
//   verbatim as prose so the model's output is never silently dropped, but
//   they never produce action events.
//
//////////////////////////////////////////////////////////////////////////////

import { expandActionCues, type ActionCue, type ActionSchema } from './action-schema.js';

export interface ActionEvent {
  readonly kind: 'action';
  readonly name: string;
  /** Raw cue text including the surrounding brackets — useful for logs. */
  readonly raw: string;
}

export interface ProseEvent {
  readonly kind: 'prose';
  readonly text: string;
}

export type ParsedEvent = ActionEvent | ProseEvent;

/**
 * Error thrown when a bracketed cue is structurally malformed — currently
 * only produced by {@link parseActionCue}, which is a convenience helper
 * for tests. The streaming parser never throws: unknown or malformed cues
 * are surfaced as prose.
 */
export class ActionParseError extends Error {
  public readonly raw: string;

  public constructor(message: string, raw: string) {
    super(message);
    this.name = 'ActionParseError';
    this.raw = raw;
  }
}

const CUE_OPEN = '[';
const CUE_CLOSE = ']';

/**
 * Builds a label → cue lookup from an ActionSchema. Separated so callers
 * that already computed the cue list can pass it directly via
 * {@link StreamingActionParser.fromCues}.
 */
function buildCueMap(cues: readonly ActionCue[]): Map<string, ActionCue> {
  const map = new Map<string, ActionCue>();
  for (const cue of cues) {
    map.set(cue.label, cue);
  }
  return map;
}

/**
 * Stateful streaming parser. Instantiate once per turn and feed decoded
 * text chunks in order. Call {@link flush} at end-of-turn to surface any
 * trailing prose. The parser never emits a partial cue — it waits for the
 * closing `]` to arrive before resolving.
 */
export class StreamingActionParser {
  private buffer = '';
  private readonly cueMap: Map<string, ActionCue>;

  /**
   * Constructs a parser from an ActionSchema. The schema is expanded into
   * the cue vocabulary that the parser will recognise.
   */
  public constructor(schema: ActionSchema) {
    this.cueMap = buildCueMap(expandActionCues(schema));
  }

  /**
   * Alternative constructor for callers (e.g. tests) that have already
   * computed the cue list and want to avoid re-running schema validation.
   */
  public static fromCues(cues: readonly ActionCue[]): StreamingActionParser {
    // Bypass the schema-based constructor by assigning directly.
    const parser = Object.create(StreamingActionParser.prototype) as StreamingActionParser;
    (parser as unknown as { buffer: string }).buffer = '';
    (parser as unknown as { cueMap: Map<string, ActionCue> }).cueMap = buildCueMap(cues);
    return parser;
  }

  /**
   * Accepts a new chunk of text and returns zero or more events derived
   * from what has been seen so far, in stream order. Any unfinished cue
   * (open `[` without a matching `]`) is retained in the internal buffer.
   */
  public consume(chunk: string): ParsedEvent[] {
    if (chunk.length === 0) {
      return [];
    }
    this.buffer += chunk;
    return this.drain(/*flushing=*/ false);
  }

  /**
   * Emits any remaining buffered prose or unresolved cue material once the
   * stream is known to be complete. Call exactly once at end-of-turn.
   *
   * If an unterminated cue is still pending, it is surfaced as prose
   * verbatim (including the opening `[`) so nothing is silently dropped.
   */
  public flush(): ParsedEvent[] {
    const events = this.drain(/*flushing=*/ true);
    this.buffer = '';
    return events;
  }

  private drain(flushing: boolean): ParsedEvent[] {
    const events: ParsedEvent[] = [];

    while (this.buffer.length > 0) {
      const openIndex = this.buffer.indexOf(CUE_OPEN);

      if (openIndex === -1) {
        // No `[` anywhere in the buffer — everything is prose.
        this.appendProse(events, this.buffer);
        this.buffer = '';
        break;
      }

      // Prose prefix up to the `[`.
      if (openIndex > 0) {
        this.appendProse(events, this.buffer.slice(0, openIndex));
        this.buffer = this.buffer.slice(openIndex);
      }

      // Buffer now starts with `[`. Look for the matching `]`.
      const closeIndex = this.buffer.indexOf(CUE_CLOSE, 1);
      if (closeIndex === -1) {
        // Cue incomplete — wait for more input unless flushing.
        if (flushing) {
          this.appendProse(events, this.buffer);
          this.buffer = '';
        }
        break;
      }

      const raw = this.buffer.slice(0, closeIndex + 1);
      const label = this.buffer.slice(1, closeIndex);
      this.buffer = this.buffer.slice(closeIndex + 1);

      const cue = this.cueMap.get(label);
      if (cue != null) {
        events.push({ kind: 'action', name: cue.name, raw });
      } else {
        // Unknown cue — surface as prose so the text is not silently
        // dropped. The grammar, when enabled, prevents this from
        // happening at generation time.
        this.appendProse(events, raw);
      }
    }

    return events;
  }

  private appendProse(events: ParsedEvent[], text: string): void {
    if (text.length === 0) {
      return;
    }
    const last = events[events.length - 1];
    if (last && last.kind === 'prose') {
      // Coalesce adjacent prose events so downstream consumers see the
      // smallest possible stream of events.
      events[events.length - 1] = { kind: 'prose', text: last.text + text };
      return;
    }
    events.push({ kind: 'prose', text });
  }
}

/**
 * Resolves a fully buffered `[label]` string against a cue list. Separated
 * out so tests can exercise label→event mapping independently of the
 * streaming state machine. Throws {@link ActionParseError} when the input
 * is not a well-formed `[...]` envelope; unknown labels also throw.
 */
export function parseActionCue(raw: string, cues: readonly ActionCue[]): ActionEvent {
  if (raw.length < 2 || raw[0] !== CUE_OPEN || raw[raw.length - 1] !== CUE_CLOSE) {
    throw new ActionParseError(`Malformed action cue: ${JSON.stringify(raw)}`, raw);
  }
  const label = raw.slice(1, -1);
  const cueMap = buildCueMap(cues);
  const cue = cueMap.get(label);
  if (cue == null) {
    throw new ActionParseError(`Unknown action cue: [${label}]`, raw);
  }
  return { kind: 'action', name: cue.name, raw };
}
