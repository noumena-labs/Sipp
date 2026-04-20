//////////////////////////////////////////////////////////////////////////////
//
// voice/stt-web-speech.ts
//
// - Thin adapter around the Web Speech Recognition API. Like its TTS sibling
//   this is a minimal shim so apps can import a stable interface today and
//   Phase D can replace the implementation without breaking callers.
//
//////////////////////////////////////////////////////////////////////////////

export interface SpeechToTextEvent {
  readonly transcript: string;
  readonly isFinal: boolean;
}

export interface SpeechToText {
  /** Whether the browser advertises SpeechRecognition support. */
  readonly isSupported: boolean;
  /** Begin listening. The supplied callback fires for every recognition event. */
  start(onResult: (event: SpeechToTextEvent) => void): void;
  /** Stop listening. Safe to call if not started. */
  stop(): void;
}

type SpeechRecognitionLike = {
  continuous: boolean;
  interimResults: boolean;
  onresult: ((event: { results: ArrayLike<ArrayLike<{ transcript: string }> & { isFinal?: boolean }> }) => void) | null;
  onerror: ((event: unknown) => void) | null;
  onend: (() => void) | null;
  start(): void;
  stop(): void;
};

type SpeechRecognitionCtor = new () => SpeechRecognitionLike;

/**
 * Creates a STT adapter backed by `SpeechRecognition` / `webkitSpeechRecognition`.
 * Returns a no-op adapter in environments where neither constructor exists.
 */
export function createWebSpeechSpeechToText(): SpeechToText {
  const globalObj = globalThis as typeof globalThis & {
    SpeechRecognition?: SpeechRecognitionCtor;
    webkitSpeechRecognition?: SpeechRecognitionCtor;
  };
  const Ctor = globalObj.SpeechRecognition ?? globalObj.webkitSpeechRecognition;
  const supported = typeof Ctor === 'function';
  let instance: SpeechRecognitionLike | null = null;

  return {
    isSupported: supported,
    start(onResult) {
      if (!supported || !Ctor) {
        return;
      }
      instance = new Ctor();
      instance.continuous = false;
      instance.interimResults = true;
      instance.onresult = (event) => {
        for (let index = 0; index < event.results.length; index += 1) {
          const result = event.results[index];
          const transcript = String(result[0]?.transcript ?? '');
          const isFinal = Boolean((result as { isFinal?: boolean }).isFinal);
          onResult({ transcript, isFinal });
        }
      };
      instance.onerror = () => {
        // Swallow errors for now; Phase D will route them through a proper
        // status channel so apps can surface recognition failures to users.
      };
      instance.onend = () => {
        instance = null;
      };
      instance.start();
    },
    stop() {
      instance?.stop();
      instance = null;
    },
  };
}
