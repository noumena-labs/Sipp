import { Example } from './base-example';

export const structuredOutputExample: Example = {
  id: '03-structured-output',
  title: 'Structured Output',
  description: 'Using GBNF grammar to force the model to output valid JSON.',
  run: async ({ log }) => {
    log('This example forces the model to output JSON using a GBNF grammar.', 'system');
    log('Type a description of an object (e.g., "A red car from 1995") to see it converted to JSON.', 'dim');
  },
  onUserInput: async ({ engine, log, userInput }) => {
    log(userInput, 'user');

    // Simple JSON grammar for an object with name, year, and color
    const jsonGrammar = `
      root ::= "{" space "\\"name\\":" space string "," space "\\"year\\":" space number "," space "\\"color\\":" space string "}"
      space ::= [ \\t\\n]*
      string ::= "\\"" [^\\"]* "\\""
      number ::= [0-9]+
    `.trim();

    log('Applying GBNF grammar for JSON object {name, year, color}...', 'dim');

    try {
      let fullResponse = '';
      const responseEl = log('', 'ai'); // Create persistent element for streaming

      await engine.chat([
        { role: 'user', content: `Extract data: ${userInput}` }
      ], {
        grammar: jsonGrammar,
        onTokens: (batch) => {
          fullResponse += batch.text;
          responseEl.innerText = fullResponse; // Update in real-time
        }
      });

      try {
        const parsed = JSON.parse(fullResponse);
        console.log('Successfully parsed JSON:', parsed);
        log('Valid JSON received and parsed.', 'system');
      } catch {
        log('⚠️ Received text follows grammar but failed native JSON.parse (likely due to trailing chars or whitespace).', 'error');
      }
    } catch (err) {
      log(`Error: ${err}`, 'error');
    }
  }
};
