//////////////////////////////////////////////////////////////////////////////
//
// action-parser.ts
//
// - Incremental parser that turns a stream of text chunks (as produced by
//   the grammar defined in action-grammar.ts) into two kinds of events:
//     * `{ kind: 'prose', text }` — a run of plain characters;
//     * `{ kind: 'action', name, args }` — a fully parsed action tag.
//
// - Must be tolerant of chunk boundaries splitting in the middle of a tag.
//   Internal buffering is retained across calls to {@link consume}.
//
//////////////////////////////////////////////////////////////////////////////

export interface ActionEvent {
  readonly kind: 'action';
  readonly name: string;
  readonly args: Readonly<Record<string, unknown>>;
  /** Raw tag text, useful for logs and testing. */
  readonly raw: string;
}

export interface ProseEvent {
  readonly kind: 'prose';
  readonly text: string;
}

export type ParsedEvent = ActionEvent | ProseEvent;

/**
 * Error thrown when an action tag is syntactically malformed despite the
 * grammar — this is a defensive check; well-behaved grammar-constrained
 * output should never trigger it.
 */
export class ActionParseError extends Error {
  public readonly raw: string;

  public constructor(message: string, raw: string) {
    super(message);
    this.name = 'ActionParseError';
    this.raw = raw;
  }
}

const TAG_PREFIX = '<action';
const TAG_TERMINATOR = '/>';

/**
 * Stateful streaming parser. Instantiate once per turn and feed decoded text
 * chunks in order. Call {@link flush} at end-of-turn to surface any trailing
 * prose. The parser never emits a partial action — it waits for the full
 * `/>` terminator to arrive before emitting.
 */
export class StreamingActionParser {
  private buffer = '';

  /**
   * Accepts a new chunk of text and returns zero or more events derived from
   * what has been seen so far, in stream order. Any unfinished tag or the
   * final sliver of prose that could be the start of a tag is retained in
   * the internal buffer.
   */
  public consume(chunk: string): ParsedEvent[] {
    if (chunk.length === 0) {
      return [];
    }
    this.buffer += chunk;
    return this.drain(/*allowTrailingProse=*/ false);
  }

  /**
   * Emits any remaining buffered prose once the stream is known to be
   * complete. Call exactly once at end-of-turn.
   *
   * If an unterminated action tag is still pending when flush is called, it
   * is surfaced as a prose event (verbatim) so that data is never silently
   * dropped.
   */
  public flush(): ParsedEvent[] {
    const events = this.drain(/*allowTrailingProse=*/ true);
    this.buffer = '';
    return events;
  }

  private drain(allowTrailingProse: boolean): ParsedEvent[] {
    const events: ParsedEvent[] = [];

    while (this.buffer.length > 0) {
      const tagStart = this.buffer.indexOf(TAG_PREFIX);

      if (tagStart === -1) {
        // No tag start anywhere in buffer. Flush prose, but keep the last
        // few characters around in case they form the beginning of a tag in
        // the next chunk (length of TAG_PREFIX - 1). On flush, dump all.
        if (allowTrailingProse) {
          this.appendProse(events, this.buffer);
          this.buffer = '';
        } else {
          const safeLen = Math.max(0, this.buffer.length - (TAG_PREFIX.length - 1));
          if (safeLen > 0) {
            this.appendProse(events, this.buffer.slice(0, safeLen));
            this.buffer = this.buffer.slice(safeLen);
          }
        }
        break;
      }

      // Prose prefix up to the tag starts.
      if (tagStart > 0) {
        this.appendProse(events, this.buffer.slice(0, tagStart));
        this.buffer = this.buffer.slice(tagStart);
      }

      // Buffer now starts with TAG_PREFIX. Look for the terminator.
      const terminatorIndex = this.buffer.indexOf(TAG_TERMINATOR);
      if (terminatorIndex === -1) {
        // Tag incomplete; wait for more input, unless the stream is flushing.
        if (allowTrailingProse) {
          this.appendProse(events, this.buffer);
          this.buffer = '';
        }
        break;
      }

      const rawTag = this.buffer.slice(0, terminatorIndex + TAG_TERMINATOR.length);
      this.buffer = this.buffer.slice(terminatorIndex + TAG_TERMINATOR.length);

      events.push(parseActionTag(rawTag));
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

const NAME_ATTR_RE = /<action\s+name="([^"]+)"(\s+args=(\{[\s\S]*?\}))?\s*\/>/;

/**
 * Parses a fully buffered `<action .../>` tag. Separated out so tests can
 * exercise parsing independently of the streaming state machine.
 */
export function parseActionTag(raw: string): ActionEvent {
  const match = NAME_ATTR_RE.exec(raw);
  if (!match) {
    throw new ActionParseError(`Malformed action tag: ${raw}`, raw);
  }
  const name = match[1];
  const argsJson = match[3];
  let args: Record<string, unknown> = {};
  if (argsJson) {
    try {
      args = JSON.parse(argsJson) as Record<string, unknown>;
    } catch (error) {
      throw new ActionParseError(
        `Invalid JSON payload in action "${name}": ${(error as Error).message}`,
        raw
      );
    }
    if (args == null || typeof args !== 'object' || Array.isArray(args)) {
      throw new ActionParseError(
        `Action "${name}" args must be a JSON object, got ${JSON.stringify(args)}`,
        raw
      );
    }
  }
  return { kind: 'action', name, args, raw };
}
