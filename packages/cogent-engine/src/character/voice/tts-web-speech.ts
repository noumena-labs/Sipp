//////////////////////////////////////////////////////////////////////////////
//
// voice/tts-web-speech.ts
//
// - Thin adapter around the Web Speech Synthesis API. Phase D will flesh out
//   SSML support and lipsync hooks; for now we expose a minimal, testable
//   contract so the avatar example can import a stable surface.
//
//////////////////////////////////////////////////////////////////////////////

export interface TextToSpeech {
  /** Starts speaking the utterance. Resolves when playback completes. */
  speak(text: string, options?: { voice?: string; rate?: number; pitch?: number }): Promise<void>;
  /** Cancels any in-flight utterance. */
  stop(): void;
  /** `true` when the platform reports Speech Synthesis support. */
  readonly isSupported: boolean;
}

/**
 * Creates a TTS adapter backed by `window.speechSynthesis`. In environments
 * without Web Speech (tests, SSR) every call is a no-op and `isSupported`
 * returns false, making feature-detection a simple property read.
 */
export function createWebSpeechTextToSpeech(): TextToSpeech {
  const globalObj: typeof globalThis & {
    speechSynthesis?: SpeechSynthesis;
    SpeechSynthesisUtterance?: typeof SpeechSynthesisUtterance;
  } = globalThis as typeof globalThis & {
    speechSynthesis?: SpeechSynthesis;
    SpeechSynthesisUtterance?: typeof SpeechSynthesisUtterance;
  };
  const synthesis = globalObj.speechSynthesis;
  const UtteranceCtor = globalObj.SpeechSynthesisUtterance;
  const supported = synthesis != null && typeof UtteranceCtor === 'function';

  return {
    isSupported: supported,
    async speak(text, options = {}) {
      if (!supported || !synthesis || !UtteranceCtor) {
        return;
      }
      const utterance = new UtteranceCtor(text);
      if (options.voice) {
        const match = synthesis.getVoices().find((voice) => voice.name === options.voice);
        if (match) {
          utterance.voice = match;
        }
      }
      if (typeof options.rate === 'number') {
        utterance.rate = options.rate;
      }
      if (typeof options.pitch === 'number') {
        utterance.pitch = options.pitch;
      }
      await new Promise<void>((resolve) => {
        utterance.onend = () => resolve();
        utterance.onerror = () => resolve();
        synthesis.speak(utterance);
      });
    },
    stop() {
      if (supported && synthesis) {
        synthesis.cancel();
      }
    },
  };
}
