//////////////////////////////////////////////////////////////////////////////
//
// director-grammar.ts
//
// - GBNF grammar and parser for WorldDirector responses. The director
//   emits two shapes depending on why it was queried:
//
//     tick narration:
//       { "note": "..." }
//
//     conflict resolution:
//       { "note": "...",
//         "resolutions": [ { "objectId": "...", "winnerAgentId": "..."|null, "note": "..." }, ... ] }
//
//   We use one permissive grammar that accepts both — `resolutions` is
//   optional — so only one grammar needs to be compiled on the native side.
//
//////////////////////////////////////////////////////////////////////////////

import type { DirectorDecision, DirectorResolution } from './simulation-types.js';

export function getDirectorGrammar(): string {
  return DIRECTOR_GRAMMAR;
}

export function parseDirectorOutput(raw: string): DirectorDecision | null {
  const trimmed = raw.trim();
  if (trimmed.length === 0) {
    return null;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(trimmed);
  } catch {
    return null;
  }
  if (!isRecord(parsed)) {
    return null;
  }
  const note = typeof parsed.note === 'string' ? parsed.note.slice(0, 200) : '';
  const rawResolutions = Array.isArray(parsed.resolutions) ? parsed.resolutions : [];
  const resolutions: DirectorResolution[] = [];
  for (const entry of rawResolutions) {
    const resolution = coerceResolution(entry);
    if (resolution) {
      resolutions.push(resolution);
    }
  }
  return { note, resolutions };
}

function coerceResolution(value: unknown): DirectorResolution | null {
  if (!isRecord(value)) return null;
  const objectId = value.objectId;
  if (typeof objectId !== 'string' || objectId.length === 0) return null;
  let winnerAgentId: string | null;
  if (value.winnerAgentId === null) {
    winnerAgentId = null;
  } else if (typeof value.winnerAgentId === 'string' && value.winnerAgentId.length > 0) {
    winnerAgentId = value.winnerAgentId;
  } else {
    winnerAgentId = null;
  }
  const note =
    typeof value.note === 'string' && value.note.length > 0
      ? { note: value.note.slice(0, 120) }
      : {};
  return { objectId, winnerAgentId, ...note };
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

const DIRECTOR_GRAMMAR = `root ::= "{" ws "\\"note\\"" ws ":" ws note-string (ws "," ws "\\"resolutions\\"" ws ":" ws resolutions)? ws "}"

resolutions ::= "[" ws (resolution (ws "," ws resolution)*)? ws "]"

resolution ::= "{" ws "\\"objectId\\"" ws ":" ws short-string ws "," ws "\\"winnerAgentId\\"" ws ":" ws (short-string | "null") (ws "," ws "\\"note\\"" ws ":" ws short-string)? ws "}"

note-string ::= "\\"" short-char{0,200} "\\""
short-string ::= "\\"" short-char{0,48} "\\""
short-char ::= [a-zA-Z0-9_ .,!?-]

ws ::= [ \\t\\n]*
`;
