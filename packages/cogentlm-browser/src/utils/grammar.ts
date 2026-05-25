export const MAX_GRAMMAR_BYTES = 64 * 1024;

export function utf8ByteLength(value: string): number {
  return typeof TextEncoder !== 'undefined'
    ? new TextEncoder().encode(value).byteLength
    : value.length;
}

export function gbnfStringLiteral(source: string): string {
  return JSON.stringify(source);
}

export function literalAlternation(values: readonly string[]): string {
  return values.map(gbnfStringLiteral).join(' | ');
}

export function compileBracketCueGrammar(options: {
  cueRuleName: string;
  labelRuleName: string;
  labels: readonly string[];
}): string {
  return [
    `root ::= ( ${options.cueRuleName} | prose-char )+`,
    'prose-char ::= [^[]',
    `${options.cueRuleName} ::= "[" ${options.labelRuleName} "]"`,
    `${options.labelRuleName} ::= ${literalAlternation(options.labels)}`,
  ].join('\n') + '\n';
}

export function compileBracketProseGrammar(): string {
  return 'root ::= prose-char+\nprose-char ::= [^[]\n';
}

export function assertGrammarByteSize(
  grammar: string | undefined,
  options: {
    readonly label?: string;
    readonly maxBytes?: number;
    readonly createError?: (message: string) => Error;
  } = {}
): void {
  if (grammar == null) {
    return;
  }

  const maxBytes = options.maxBytes ?? MAX_GRAMMAR_BYTES;
  const byteLength = utf8ByteLength(grammar);
  if (byteLength <= maxBytes) {
    return;
  }

  const label = options.label ?? 'grammar';
  const createError = options.createError ?? ((message: string) => new Error(message));
  throw createError(`${label} exceeds maximum size of ${maxBytes} bytes (got ${byteLength}).`);
}
