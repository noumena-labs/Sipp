export interface ChatMessage {
  readonly id: string;
  readonly role: 'user' | 'assistant';
  readonly text: string;
  readonly actions: ReadonlyArray<{
    id: string;
    label: string;
  }>;
  readonly pending?: boolean;
}
