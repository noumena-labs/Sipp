import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

type WatchEvent = 'add' | 'change' | 'unlink';

interface ViteDevServerLike {
  watcher: {
    add: (paths: string | string[]) => void;
    on: (event: WatchEvent, listener: (filePath: string) => void) => void;
    off: (event: WatchEvent, listener: (filePath: string) => void) => void;
  };
  ws: {
    send: (payload: { type: 'full-reload' }) => void;
  };
  httpServer?: {
    once: (event: 'close', listener: () => void) => void;
  } | null;
}

interface VitePluginLike {
  name: string;
  apply: 'serve';
  configureServer: (server: ViteDevServerLike) => void;
}

const appsDir = fileURLToPath(new URL('.', import.meta.url));
const repoRoot = path.resolve(appsDir, '..');
const cogentClientPackageDir = path.join(repoRoot, 'packages', 'npm');
const cogentClientArtifactDir = path.join(
  repoRoot,
  '.build',
  'artifacts',
  'npm',
  'cogentlm-browser'
);
const cogentClientSrcDir = path.join(cogentClientPackageDir, 'src');
const cogentClientWasmDir = path.join(cogentClientArtifactDir, 'dist', 'wasm');
const sourceFilePattern = /\.tsx?$/;
const wasmArtifactPattern = /cogentlm-wasm\.(?:js|wasm)$/;
const rebuildArgs = ['run', '--filter=@noumena-labs/cogentlm-browser', 'build:ts'];

function isCogentClientSourceFile(filePath: string): boolean {
  const resolvedPath = path.resolve(filePath);
  const relativePath = path.relative(cogentClientSrcDir, resolvedPath);
  return (
    relativePath !== '' &&
    !relativePath.startsWith('..') &&
    !path.isAbsolute(relativePath) &&
    sourceFilePattern.test(resolvedPath)
  );
}

function isCogentClientWasmArtifact(filePath: string): boolean {
  const resolvedPath = path.resolve(filePath);
  const relativePath = path.relative(cogentClientWasmDir, resolvedPath);
  return (
    relativePath !== '' &&
    !relativePath.startsWith('..') &&
    !path.isAbsolute(relativePath) &&
    wasmArtifactPattern.test(resolvedPath)
  );
}

function rebuildCogentClientDist(): Promise<boolean> {
  return new Promise((resolve) => {
    const childProcess = spawn('bun', rebuildArgs, {
      cwd: repoRoot,
      stdio: 'inherit',
      shell: false,
      windowsHide: true,
    });

    childProcess.once('error', (error) => {
      console.error(`[cogentlm-browser] failed to start TS rebuild: ${error.message}`);
      resolve(false);
    });

    childProcess.once('exit', (code, signal) => {
      if (signal) {
        console.error(`[cogentlm-browser] TS rebuild terminated by ${signal}`);
        resolve(false);
        return;
      }

      resolve(code === 0);
    });
  });
}

export function cogentClientDistWatch(): VitePluginLike {
  return {
    name: 'cogentlm-browser-dist-watch',
    apply: 'serve',
    configureServer(server) {
      let debounceTimer: ReturnType<typeof setTimeout> | null = null;
      let rebuildRunning = false;
      let rebuildRequested = false;
      let stopped = false;

      const runRebuild = async () => {
        if (rebuildRunning) {
          rebuildRequested = true;
          return;
        }

        rebuildRunning = true;

        do {
          rebuildRequested = false;
          console.info('[cogentlm-browser] rebuilding TS dist...');
          const rebuildSucceeded = await rebuildCogentClientDist();

          if (stopped) {
            break;
          }

          if (rebuildSucceeded) {
            console.info('[cogentlm-browser] TS dist rebuilt; reloading app.');
            server.ws.send({ type: 'full-reload' });
          } else {
            console.error('[cogentlm-browser] TS dist rebuild failed; keeping current app session.');
          }
        } while (rebuildRequested && !stopped);

        rebuildRunning = false;
      };

      const scheduleRebuild = (filePath: string) => {
        if (!isCogentClientSourceFile(filePath)) {
          return;
        }

        if (debounceTimer) {
          clearTimeout(debounceTimer);
        }

        debounceTimer = setTimeout(() => {
          debounceTimer = null;
          void runRebuild();
        }, 150);
      };

      let wasmReloadTimer: ReturnType<typeof setTimeout> | null = null;
      const scheduleWasmReload = (filePath: string) => {
        if (!isCogentClientWasmArtifact(filePath)) {
          return;
        }

        if (wasmReloadTimer) {
          clearTimeout(wasmReloadTimer);
        }

        wasmReloadTimer = setTimeout(() => {
          wasmReloadTimer = null;
          console.info('[cogentlm-browser] wasm runtime rebuilt; reloading app.');
          server.ws.send({ type: 'full-reload' });
        }, 150);
      };

      const handleAdd = (filePath: string) => {
        scheduleRebuild(filePath);
        scheduleWasmReload(filePath);
      };
      const handleChange = (filePath: string) => {
        scheduleRebuild(filePath);
        scheduleWasmReload(filePath);
      };
      const handleUnlink = (filePath: string) => {
        scheduleRebuild(filePath);
        scheduleWasmReload(filePath);
      };

      server.watcher.add([cogentClientSrcDir, cogentClientWasmDir]);
      server.watcher.on('add', handleAdd);
      server.watcher.on('change', handleChange);
      server.watcher.on('unlink', handleUnlink);
      console.info('[cogentlm-browser] watching packages/npm/src and .build artifact wasm.');

      server.httpServer?.once('close', () => {
        stopped = true;

        if (debounceTimer) {
          clearTimeout(debounceTimer);
          debounceTimer = null;
        }
        if (wasmReloadTimer) {
          clearTimeout(wasmReloadTimer);
          wasmReloadTimer = null;
        }

        server.watcher.off('add', handleAdd);
        server.watcher.off('change', handleChange);
        server.watcher.off('unlink', handleUnlink);
      });
    },
  };
}
