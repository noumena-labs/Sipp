import { gbnfStringLiteral } from '../utils/grammar.js';

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
