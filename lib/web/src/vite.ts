interface ViteServerConfig {
  readonly headers: Readonly<Record<string, string>>;
}

/** Minimal Vite configuration required by the bundled pthread WASM runtime. */
export interface SippViteConfig {
  readonly server: ViteServerConfig;
  readonly preview: ViteServerConfig;
}
/**
 * Returns the development and preview headers required for cross-origin
 * isolation. Spread the result into a Vite configuration object.
 */
export function sippViteConfig(): SippViteConfig {
  const headers = {
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Embedder-Policy': 'require-corp',
  } as const;
  return {
    server: { headers },
    preview: { headers },
  };
}
