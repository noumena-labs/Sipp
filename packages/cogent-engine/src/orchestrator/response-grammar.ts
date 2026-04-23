//////////////////////////////////////////////////////////////////////////////
//
// response-grammar.ts
//
// - Compiles a small JSON response schema subset into GBNF.
// - The intent is not full JSON Schema compatibility. This is a pragmatic,
//   deterministic compiler for the contracts used by `director.json`.
//
//////////////////////////////////////////////////////////////////////////////

import type { ResponseSchema } from './director-types.js';

const DEFAULT_STRING_MAX = 240;

export function compileResponseGrammar(schema: ResponseSchema): string {
  const builder = new GrammarBuilder();
  return builder.compile(schema);
}

class GrammarBuilder {
  private readonly rules: string[] = [];
  private nextId = 0;

  public compile(schema: ResponseSchema): string {
    const rootRule = this.defineSchema(schema);
    this.rules.unshift(`root ::= ${rootRule}`);
    this.rules.push('number ::= "-"? [0-9] [0-9]* ("." [0-9]+)?');
    this.rules.push('integer ::= "-"? [0-9] [0-9]*');
    this.rules.push('string-char ::= [a-zA-Z0-9_ .,!?;:/@#%&()+=*\'\-]');
    this.rules.push('ws ::= [ \\t\\n\\r]*');
    return this.rules.join('\n') + '\n';
  }

  private defineSchema(schema: ResponseSchema): string {
    const innerName = this.defineNonNullableSchema(schema);
    if (schema.type !== 'null' && schema.nullable) {
      const wrapper = this.alloc('nullable');
      this.rules.push(`${wrapper} ::= ${innerName} | "null"`);
      return wrapper;
    }
    return innerName;
  }

  private defineNonNullableSchema(schema: ResponseSchema): string {
    const ruleName = this.alloc(schema.type);
    switch (schema.type) {
      case 'null':
        this.rules.push(`${ruleName} ::= "null"`);
        return ruleName;
      case 'boolean':
        this.rules.push(`${ruleName} ::= "true" | "false"`);
        return ruleName;
      case 'number':
        this.rules.push(`${ruleName} ::= ${schema.integer ? 'integer' : 'number'}`);
        return ruleName;
      case 'string': {
        if (schema.enum && schema.enum.length > 0) {
          this.rules.push(
            `${ruleName} ::= ${schema.enum.map((value) => gbnfStringLiteral(value)).join(' | ')}`
          );
        } else {
          const max = schema.maxLength ?? DEFAULT_STRING_MAX;
          this.rules.push(`${ruleName} ::= "\\\"" string-char{0,${max}} "\\\""`);
        }
        return ruleName;
      }
      case 'array': {
        const itemRule = this.defineSchema(schema.items);
        this.rules.push(`${ruleName} ::= "[" ws (${itemRule} (ws "," ws ${itemRule})*)? ws "]"`);
        return ruleName;
      }
      case 'object': {
        const entries = Object.entries(schema.properties);
        if (entries.length === 0) {
          this.rules.push(`${ruleName} ::= "{" ws "}"`);
          return ruleName;
        }
        const fields = entries.map(([key, value]) => {
          const valueRule = this.defineSchema(value);
          return `${gbnfStringLiteral(key)} ws ":" ws ${valueRule}`;
        });
        this.rules.push(`${ruleName} ::= "{" ws ${fields.join(' ws "," ws ')} ws "}"`);
        return ruleName;
      }
    }
  }

  private alloc(prefix: string): string {
    const id = this.nextId;
    this.nextId += 1;
    return `${prefix}_${id}`;
  }
}

function gbnfStringLiteral(source: string): string {
  const escaped = source.replace(/\\/g, '\\\\').replace(/"/g, '\\"');
  return `"\\"${escaped}\\""`;
}
