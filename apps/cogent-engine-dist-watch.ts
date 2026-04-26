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
const cogentEngineSrcDir = path.join(repoRoot, 'packages', 'cogent-engine', 'src');
const sourceFilePattern = /\.tsx?$/;
const rebuildArgs = ['run', '--filter=@noumena-labs/cogent-engine', 'build:ts'];

function isCogentEngineSourceFile(filePath: string): boolean {
  const resolvedPath = path.resolve(filePath);
  const relativePath = path.relative(cogentEngineSrcDir, resolvedPath);
  return (
    relativePath !== '' &&
    !relativePath.startsWith('..') &&
    !path.isAbsolute(relativePath) &&
    sourceFilePattern.test(resolvedPath)
  );
}

function rebuildCogentEngineDist(): Promise<boolean> {
  return new Promise((resolve) => {
    const childProcess = spawn('bun', rebuildArgs, {
      cwd: repoRoot,
      stdio: 'inherit',
      shell: false,
      windowsHide: true,
    });

    childProcess.once('error', (error) => {
      console.error(`[cogent-engine] failed to start TS rebuild: ${error.message}`);
      resolve(false);
    });

    childProcess.once('exit', (code, signal) => {
      if (signal) {
        console.error(`[cogent-engine] TS rebuild terminated by ${signal}`);
        resolve(false);
        return;
      }

      resolve(code === 0);
    });
  });
}

export function cogentEngineDistWatch(): VitePluginLike {
  return {
    name: 'cogent-engine-dist-watch',
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
          console.info('[cogent-engine] rebuilding TS dist...');
          const rebuildSucceeded = await rebuildCogentEngineDist();

          if (stopped) {
            break;
          }

          if (rebuildSucceeded) {
            console.info('[cogent-engine] TS dist rebuilt; reloading app.');
            server.ws.send({ type: 'full-reload' });
          } else {
            console.error('[cogent-engine] TS dist rebuild failed; keeping current app session.');
          }
        } while (rebuildRequested && !stopped);

        rebuildRunning = false;
      };

      const scheduleRebuild = (filePath: string) => {
        if (!isCogentEngineSourceFile(filePath)) {
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

      const handleAdd = (filePath: string) => scheduleRebuild(filePath);
      const handleChange = (filePath: string) => scheduleRebuild(filePath);
      const handleUnlink = (filePath: string) => scheduleRebuild(filePath);

      server.watcher.add(cogentEngineSrcDir);
      server.watcher.on('add', handleAdd);
      server.watcher.on('change', handleChange);
      server.watcher.on('unlink', handleUnlink);
      console.info('[cogent-engine] watching packages/cogent-engine/src for TS dist rebuilds.');

      server.httpServer?.once('close', () => {
        stopped = true;

        if (debounceTimer) {
          clearTimeout(debounceTimer);
          debounceTimer = null;
        }

        server.watcher.off('add', handleAdd);
        server.watcher.off('change', handleChange);
        server.watcher.off('unlink', handleUnlink);
      });
    },
  };
}
