import type { CogentClient } from '@noumena-labs/cogentlm';

export interface ExampleContext {
  client: CogentClient;
  log: (message: string, type?: 'system' | 'user' | 'ai' | 'error' | 'dim') => HTMLElement;
  userInput: string;
  inputElement: HTMLInputElement;
  media?: Uint8Array[];
}

export interface Example {
  id: string;
  title: string;
  description: string;
  run: (ctx: ExampleContext) => Promise<void>;
  onUserInput?: (ctx: ExampleContext) => Promise<void>;
}
