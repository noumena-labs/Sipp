//////////////////////////////////////////////////////////////////////////////
//
// App.tsx
//
// - Top-level simulation app. Wires:
//     - a single shared CogentEngine, loaded from a user-pasted .gguf URL
//     - a WorldDirector and four SimulationAgents, all sharing that engine
//     - a WorldOrchestrator that drives the tick loop
//     - a SimulationCanvas that mirrors tick-end snapshots into three.js
//     - a side panel with transport, event log, and an agent inspector
//
//////////////////////////////////////////////////////////////////////////////

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { CogentEngine, getBundledRuntimeUrls } from 'cogent-engine';
import {
  createSimulationAgentFromConfigUrl,
  SimulationBus,
  WorldDirector,
  WorldOrchestrator,
  type SimulationAgentState,
  type SimulationEvent,
  type WorldSnapshot,
} from 'cogent-engine/orchestrator';
import { SimulationCanvas } from './components/SimulationCanvas';
import { ControlsPanel } from './components/ControlsPanel';
import { EventLog, type EventLogEntry } from './components/EventLog';
import { AgentInspector } from './components/AgentInspector';
import { COURTYARD_AGENTS, COURTYARD_SCENARIO } from './scenarios/courtyard-snack';

interface LoadedHarness {
  readonly engine: CogentEngine;
  readonly orchestrator: WorldOrchestrator;
}

export default function App() {
  const bus = useMemo(() => new SimulationBus(), []);
  const [modelUrl, setModelUrl] = useState('');
  const [status, setStatus] = useState('Idle. Paste a .gguf URL and press Load.');
  const [busy, setBusy] = useState(false);
  const [harness, setHarness] = useState<LoadedHarness | null>(null);
  const [running, setRunning] = useState(false);
  const [tickHz, setTickHz] = useState(1.5);
  const [snapshot, setSnapshot] = useState<WorldSnapshot | null>(null);
  const [events, setEvents] = useState<EventLogEntry[]>([]);
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);
  const eventIdRef = useRef(0);

  const pushEvent = useCallback((entry: Omit<EventLogEntry, 'id'>) => {
    const id = ++eventIdRef.current;
    setEvents((prev) => [...prev.slice(-199), { ...entry, id }]);
  }, []);

  // Subscribe to bus once.
  useEffect(() => {
    const off = bus.onAny((event: SimulationEvent) => {
      switch (event.kind) {
        case 'tick-end':
          setSnapshot(event.snapshot);
          break;
        case 'agent-query-start':
          pushEvent({ tick: event.tick, kind: 'query', text: `querying ${event.agentId}…` });
          break;
        case 'agent-intent':
          pushEvent({
            tick: event.tick,
            kind: 'intent',
            text: `${event.agentId} -> ${event.intent.kind}`,
          });
          break;
        case 'director-conflict':
          pushEvent({
            tick: event.tick,
            kind: 'conflict',
            text: `conflict: ${event.conflicts.map((c) => `${c.objectId} contested by [${c.contenderAgentIds.join(', ')}]`).join('; ')}`,
          });
          break;
        case 'director-decision': {
          const parts: string[] = [];
          if (event.decision.note) parts.push(event.decision.note);
          for (const r of event.decision.resolutions) {
            parts.push(`${r.objectId} -> ${r.winnerAgentId ?? 'none'}`);
          }
          pushEvent({ tick: event.tick, kind: 'decision', text: parts.join(' | ') });
          break;
        }
        case 'world-note':
          pushEvent({ tick: event.tick, kind: 'note', text: event.note });
          break;
      }
    });
    return () => off();
  }, [bus, pushEvent]);

  // Keep orchestrator tick rate synced with slider.
  useEffect(() => {
    harness?.orchestrator.setTickHz(tickHz);
  }, [harness, tickHz]);

  const loadHarness = useCallback(
    async (url: string): Promise<void> => {
      setBusy(true);
      setModelUrl(url);
      try {
        if (harness) {
          await harness.orchestrator.dispose();
          harness.engine.close();
          setHarness(null);
          setRunning(false);
        }
        setStatus('Initialising engine…');
        const engine = new CogentEngine({ ...getBundledRuntimeUrls() });
        await engine.initModule();

        setStatus('Downloading model…');
        const modelPath = await engine.loadModelFromUrl(url, 'model.gguf', (pct) =>
          setStatus(`Downloading model… ${Math.floor(pct)}%`)
        );

        setStatus('Initialising inference runtime…');
        await engine.initEngine(modelPath, {
          sampling: {
            temperature: 0.5,
            topP: 0.9,
            topK: 40,
            minP: 0.05,
            repeatPenalty: 1.05,
          },
        });

        setStatus('Building orchestrator…');
        const director = new WorldDirector('courtyard', engine);
        const orchestrator = new WorldOrchestrator(director, {
          id: 'courtyard',
          bus,
          bounds: COURTYARD_SCENARIO.bounds,
          tickHz,
          directorCadenceTicks: 10,
          initialDirectorNote: COURTYARD_SCENARIO.directorNote ?? null,
        });
        for (const seed of COURTYARD_SCENARIO.objects) {
          orchestrator.upsertObject(seed);
        }

        setStatus('Loading agent personas…');
        for (const assignment of COURTYARD_AGENTS) {
          const { agent } = await createSimulationAgentFromConfigUrl({
            agentId: assignment.agentId,
            configUrl: assignment.characterUrl,
            engine,
          });
          const seed = COURTYARD_SCENARIO.agents.find((a) => a.id === assignment.agentId);
          if (!seed) {
            throw new Error(`no scenario seed for ${assignment.agentId}`);
          }
          orchestrator.addAgent(agent, seed);
        }

        setSnapshot(orchestrator.getSnapshot());
        setHarness({ engine, orchestrator });
        setStatus('Ready. Press Start.');
      } catch (error) {
        console.error(error);
        setStatus(`Load failed: ${(error as Error).message}`);
      } finally {
        setBusy(false);
      }
    },
    [bus, harness, tickHz]
  );

  const handleStart = (): void => {
    if (!harness) return;
    harness.orchestrator.start();
    setRunning(true);
    setStatus('Running.');
  };

  const handlePause = (): void => {
    if (!harness) return;
    harness.orchestrator.pause();
    setRunning(false);
    setStatus('Paused.');
  };

  const handleStep = async (): Promise<void> => {
    if (!harness) return;
    if (running) {
      harness.orchestrator.pause();
      setRunning(false);
    }
    setStatus('Stepping…');
    await harness.orchestrator.step();
    setStatus('Stepped.');
  };

  const handleReset = async (): Promise<void> => {
    if (!harness) return;
    setBusy(true);
    try {
      await harness.orchestrator.dispose();
      harness.engine.close();
      setHarness(null);
      setRunning(false);
      setSnapshot(null);
      setEvents([]);
      setSelectedAgentId(null);
      setStatus('Reset. Press Load to rebuild.');
    } finally {
      setBusy(false);
    }
  };

  const agents: readonly SimulationAgentState[] = snapshot?.agents ?? [];

  return (
    <div className="sim-app">
      <SimulationCanvas
        bus={bus}
        bounds={COURTYARD_SCENARIO.bounds ?? { halfExtent: 8 }}
        highlightedAgentId={selectedAgentId}
      />

      <div className="sim-overlay sim-top-left">
        <ControlsPanel
          modelUrl={modelUrl}
          onLoad={loadHarness}
          onStart={handleStart}
          onPause={handlePause}
          onStep={handleStep}
          onReset={handleReset}
          tickHz={tickHz}
          onTickHzChange={setTickHz}
          status={status}
          busy={busy}
          loaded={harness != null}
          running={running}
          tick={snapshot?.tick ?? 0}
        />
      </div>

      <div className="sim-overlay sim-top-right">
        <AgentInspector
          agents={agents}
          selectedAgentId={selectedAgentId}
          onSelect={setSelectedAgentId}
        />
      </div>

      <div className="sim-overlay sim-bottom">
        <EventLog entries={events} />
      </div>

      {snapshot?.directorNote ? (
        <div className="sim-overlay sim-top-center">
          <div className="director-note glass-panel">
            <span className="panel-eyebrow">Director</span>
            <span>{snapshot.directorNote}</span>
          </div>
        </div>
      ) : null}
    </div>
  );
}
