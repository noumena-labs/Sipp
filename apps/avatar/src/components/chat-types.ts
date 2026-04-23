export interface ChatMessage {
  readonly id: string;
  readonly role: 'user' | 'assistant';
  readonly text: string;
  readonly actions: ReadonlyArray<{
    name: string;
    label: string;
  }>;
  readonly pending?: boolean;
}
