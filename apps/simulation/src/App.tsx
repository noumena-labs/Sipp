//////////////////////////////////////////////////////////////////////////////
//
// App.tsx
//
// - Top-level simulation app. Wires:
//     - a single shared CogentEngine, loaded from a user-pasted .gguf URL
//     - a DirectorRuntime from `director.json`
//     - four CharacterAgent-backed chooser adapters
//     - an app-local SimulationRuntime that owns the world loop
//     - a SimulationCanvas that mirrors tick-end snapshots into three.js
//     - a side panel with transport, event log, and an agent inspector
//
//////////////////////////////////////////////////////////////////////////////

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { CogentEngine, getBundledRuntimeUrls } from 'cogent-engine';
import { createDirectorFromConfigUrl } from 'cogent-engine/orchestrator';
import { SimulationCanvas } from './components/SimulationCanvas';
import { ControlsPanel } from './components/ControlsPanel';
import { EventLog, type EventLogEntry } from './components/EventLog';
import { AgentInspector } from './components/AgentInspector';
import { Scoreboard } from './components/Scoreboard';
import { COURTYARD_AGENTS, COURTYARD_SCENARIO } from './scenarios/courtyard-snack.js';
import { SimulationBus, type SimulationEvent } from './runtime/bus.js';
import {
  createSimulationAgentChooserFromConfigUrl,
} from './runtime/agent-chooser.js';
import { SimulationRuntime } from './runtime/simulation-runtime.js';
import type {
  DirectorDecision,
  DirectorResolution,
  SimulationAgentState,
  SimulationGameEvent,
  WorldConflict,
  WorldSnapshot,
} from './runtime/types.js';

interface LoadedHarness {
  readonly engine: CogentEngine;
  readonly runtime: SimulationRuntime;
}

interface DirectorPanelState {
  readonly mode: 'idle' | 'ruling' | 'update';
  readonly tick: number | null;
  readonly headline: string;
  readonly detail: string | null;
  readonly note: string | null;
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
  const [eventLogCollapsed, setEventLogCollapsed] = useState(true);
  const eventIdRef = useRef(0);
  const snapshotRef = useRef<WorldSnapshot | null>(null);
  const [directorState, setDirectorState] = useState<DirectorPanelState>(() =>
    createInitialDirectorState(COURTYARD_SCENARIO.directorNote ?? null)
  );

  const pushEvent = useCallback((entry: Omit<EventLogEntry, 'id'>) => {
    const id = ++eventIdRef.current;
    setEvents((prev) => [...prev.slice(-199), { ...entry, id }]);
  }, []);

  // Subscribe to bus once.
  useEffect(() => {
    const off = bus.onAny((event: SimulationEvent) => {
      switch (event.kind) {
        case 'tick-end':
          snapshotRef.current = event.snapshot;
          setSnapshot(event.snapshot);
          if (event.snapshot.directorNote) {
            setDirectorState((prev) => (
              prev.note === event.snapshot.directorNote
                ? prev
                : { ...prev, note: event.snapshot.directorNote }
            ));
          }
          break;
        case 'agent-query-start':
          pushEvent({
            tick: event.tick,
            kind: 'query',
            text: `${nameOf(event.agentId, snapshotRef.current)} is thinking...`,
          });
          break;
        case 'agent-query-end':
          if (event.errorMessage) {
            pushEvent({
              tick: event.tick,
              kind: 'note',
              text: `${nameOf(event.agentId, snapshotRef.current)} hits a decision error: ${event.errorMessage}`,
            });
          } else if (event.cancelled) {
            pushEvent({
              tick: event.tick,
              kind: 'query',
              text: `${nameOf(event.agentId, snapshotRef.current)}'s turn gets interrupted.`,
            });
          }
          break;
        case 'agent-intent':
          pushEvent({
            tick: event.tick,
            kind: 'intent',
            text: `${nameOf(event.agentId, snapshotRef.current)} decides to ${event.goal.label}.`,
          });
          break;
        case 'director-conflict': {
          const detail = event.conflicts
            .map((conflict) => describeConflict(conflict, snapshotRef.current))
            .join('; ');
          pushEvent({
            tick: event.tick,
            kind: 'referee',
            text: `Director reviews: ${detail}`,
          });
          setDirectorState({
            mode: 'ruling',
            tick: event.tick,
            headline: 'Adjudicating a contested play',
            detail,
            note: 'Reviewing the scramble before making the call.',
          });
          break;
        }
        case 'director-decision': {
          const summary = describeDecision(event.decision, snapshotRef.current);
          if (event.decision.resolutions.length > 0) {
            pushEvent({ tick: event.tick, kind: 'decision', text: summary });
          }
          setDirectorState((prev) => {
            const note = event.decision.note || prev.note;
            return {
              mode: 'update',
              tick: event.tick,
              headline: event.decision.resolutions.length > 0 ? 'Ruling delivered' : 'Director update',
              detail: event.decision.resolutions.length > 0 ? summary : note,
              note,
            };
          });
          break;
        }
        case 'world-note':
          pushEvent({ tick: event.tick, kind: 'note', text: `Director: ${event.note}` });
          setDirectorState((prev) => ({
            ...prev,
            tick: event.tick,
            note: event.note,
            ...(prev.mode === 'idle' ? { headline: 'Director update', detail: event.note } : {}),
          }));
          break;
        case 'game-event': {
          const text = describeGameEvent(event.event, snapshotRef.current);
          if (!text) break;
          pushEvent({
            tick: event.tick,
            kind: event.event.kind === 'fallback' ? 'note' : 'game',
            text,
          });
          break;
        }
      }
    });
    return () => off();
  }, [bus, pushEvent]);

  useEffect(() => {
    if (!running || !harness) {
      return;
    }
    const delayMs = 1000 / tickHz;
    let disposed = false;
    let timeoutId: ReturnType<typeof setTimeout> | null = null;

    const loop = async (): Promise<void> => {
      if (disposed) return;
      await harness.runtime.step(1 / tickHz);
      await harness.runtime.waitForIdle();
      if (disposed) return;
      timeoutId = setTimeout(() => {
        void loop();
      }, delayMs);
    };

    timeoutId = setTimeout(() => {
      void loop();
    }, 0);

    return () => {
      disposed = true;
      if (timeoutId !== null) {
        clearTimeout(timeoutId);
      }
    };
  }, [harness, running, tickHz]);

  const loadHarness = useCallback(
    async (url: string): Promise<void> => {
      setBusy(true);
      setModelUrl(url);
      eventIdRef.current = 0;
      setEvents([]);
      setSelectedAgentId(null);
      snapshotRef.current = null;
      setSnapshot(null);
      setDirectorState(createInitialDirectorState(COURTYARD_SCENARIO.directorNote ?? null));
      try {
        if (harness) {
          await harness.runtime.dispose();
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

        setStatus('Loading director config…');
        const { director } = await createDirectorFromConfigUrl({
          configUrl: COURTYARD_SCENARIO.directorConfigUrl,
          engine,
          runtimeOptions: { maxOutputTokens: 96 },
        });

        setStatus('Building simulation runtime…');
        const runtime = new SimulationRuntime(director, {
          id: COURTYARD_SCENARIO.id,
          bus,
          bounds: COURTYARD_SCENARIO.bounds,
          game: COURTYARD_SCENARIO.game,
          directorCadenceTicks: COURTYARD_SCENARIO.directorCadenceTicks,
          initialDirectorNote: COURTYARD_SCENARIO.directorNote ?? null,
          resolveRefereeQuery: COURTYARD_SCENARIO.resolveRefereeQuery,
          narrateQuery: COURTYARD_SCENARIO.narrateQuery,
        });
        for (const seed of COURTYARD_SCENARIO.objects) {
          runtime.upsertObject(seed);
        }

        setStatus('Loading agent personas…');
        for (const assignment of COURTYARD_AGENTS) {
          const { agent } = await createSimulationAgentChooserFromConfigUrl({
            agentId: assignment.agentId,
            configUrl: assignment.characterUrl,
            engine,
          });
          const seed = COURTYARD_SCENARIO.agents.find((a) => a.id === assignment.agentId);
          if (!seed) {
            throw new Error(`no scenario seed for ${assignment.agentId}`);
          }
          runtime.addAgent(agent, seed);
        }

        const initialSnapshot = runtime.getSnapshot();
        snapshotRef.current = initialSnapshot;
        setSnapshot(initialSnapshot);
        setDirectorState(createInitialDirectorState(initialSnapshot.directorNote));
        setHarness({ engine, runtime });
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
    setRunning(true);
    setStatus('Running.');
  };

  const handlePause = (): void => {
    if (!harness) return;
    setRunning(false);
    setStatus('Paused.');
  };

  const handleStep = async (): Promise<void> => {
    if (!harness) return;
    if (running) {
      setRunning(false);
    }
    setStatus('Stepping…');
    await harness.runtime.step(1 / tickHz);
    await harness.runtime.waitForIdle();
    setStatus('Stepped.');
  };

  const handleReset = async (): Promise<void> => {
    if (!harness) return;
    setBusy(true);
    try {
      await harness.runtime.dispose();
      harness.engine.close();
      setHarness(null);
      setRunning(false);
      snapshotRef.current = null;
      setSnapshot(null);
      setEvents([]);
      setSelectedAgentId(null);
      eventIdRef.current = 0;
      setDirectorState(createInitialDirectorState(COURTYARD_SCENARIO.directorNote ?? null));
      setStatus('Reset. Press Load to rebuild.');
    } finally {
      setBusy(false);
    }
  };

  const agents: readonly SimulationAgentState[] = snapshot?.agents ?? [];
  const scoreboardStatus = directorState.mode === 'ruling'
    ? 'director adjudicating a call'
    : 'race in progress';
  const directorDetail = directorState.detail?.trim() || null;
  const directorNote = directorState.note?.trim() || null;
  const showDirectorNote = directorNote != null && directorNote !== directorDetail;

  return (
    <div className="sim-app">
      <SimulationCanvas
        bus={bus}
        bounds={COURTYARD_SCENARIO.bounds ?? { halfExtent: 8 }}
        highlightedAgentId={selectedAgentId}
        snapshot={snapshot}
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
        <EventLog
          entries={events}
          collapsed={eventLogCollapsed}
          onToggle={() => setEventLogCollapsed((value) => !value)}
        />
      </div>

      {snapshot ? (
        <div className="sim-overlay sim-top-center sim-center-stack">
          <Scoreboard snapshot={snapshot} metaText={scoreboardStatus} />
          <div className={`director-note glass-panel director-${directorState.mode}`}>
            <div className="director-note-head">
              <div className="director-note-main">
                <span className="panel-eyebrow">Director</span>
                <span className="director-headline">{directorState.headline}</span>
              </div>
              <span className={`director-pill director-pill-${directorState.mode}`}>
                {directorState.mode === 'ruling' ? 'Adjudicating' : 'Live'}
              </span>
            </div>
            {directorDetail ? <span className="director-detail">{directorDetail}</span> : null}
            {showDirectorNote ? <span className="director-quote">{directorNote}</span> : null}
          </div>
        </div>
      ) : null}
    </div>
  );
}

function describeGameEvent(
  event: SimulationGameEvent,
  snapshot: WorldSnapshot | null
): string | null {
  switch (event.kind) {
    case 'pickup':
      return `${nameOf(event.agentId, snapshot)} grabs the banana.`;
    case 'delivery':
      return `${nameOf(event.agentId, snapshot)} scores by delivering the banana!`;
    case 'respawn':
      return `The banana pops back into play at (${event.position.x.toFixed(1)}, ${event.position.z.toFixed(1)}).`;
    case 'drop':
      return `${nameOf(event.agentId, snapshot)} drops the banana${event.cause === 'forced' ? ' after a bump' : ''}.`;
    case 'forced_drop':
      if (event.outcome === 'drop') {
        return `${nameOf(event.attackerAgentId, snapshot)} bumps ${nameOf(event.targetAgentId, snapshot)} and the banana flies loose.`;
      }
      if (event.outcome === 'attacker_fumbles') {
        return `${nameOf(event.attackerAgentId, snapshot)} fumbles the bump on ${nameOf(event.targetAgentId, snapshot)}.`;
      }
      return `${nameOf(event.targetAgentId, snapshot)} keeps hold through the bump.`;
    case 'fallback':
      if (event.message.startsWith('Director is ruling on ')) {
        return null;
      }
      return event.message;
  }
}

function nameOf(agentId: string, snapshot: WorldSnapshot | null): string {
  return snapshot?.agents.find((agent) => agent.id === agentId)?.name
    ?? COURTYARD_SCENARIO.agents.find((agent) => agent.id === agentId)?.name
    ?? agentId;
}

function objectLabelOf(objectId: string, snapshot: WorldSnapshot | null): string {
  return snapshot?.objects.find((object) => object.id === objectId)?.label ?? objectId;
}

function describeConflict(conflict: WorldConflict, snapshot: WorldSnapshot | null): string {
  if (conflict.kind === 'contested_object') {
    return `${objectLabelOf(conflict.objectId, snapshot)} contested by ${conflict.contenderAgentIds
      .map((agentId) => nameOf(agentId, snapshot))
      .join(', ')}`;
  }
  return `${nameOf(conflict.attackerAgentId, snapshot)} bumping ${nameOf(conflict.targetAgentId, snapshot)}`;
}

function describeDecision(decision: DirectorDecision, snapshot: WorldSnapshot | null): string {
  const summaries = decision.resolutions.map((resolution) => describeResolution(resolution, snapshot));
  if (summaries.length > 0) {
    return summaries.join(' ');
  }
  return decision.note || 'Director updates the scene.';
}

function describeResolution(resolution: DirectorResolution, snapshot: WorldSnapshot | null): string {
  const parsed = parseConflictKey(resolution.conflictId);
  if (parsed?.kind === 'pickup') {
    const objectLabel = objectLabelOf(parsed.objectId, snapshot);
    if (resolution.outcome === 'pickup' && resolution.winnerAgentId) {
      return `${nameOf(resolution.winnerAgentId, snapshot)} wins the ${objectLabel}.`;
    }
    return `The referee waves off the ${objectLabel} scramble.`;
  }
  if (parsed?.kind === 'forced_drop') {
    const attacker = nameOf(parsed.attackerAgentId, snapshot);
    const target = nameOf(parsed.targetAgentId, snapshot);
    switch (resolution.outcome) {
      case 'drop':
        return `${attacker} knocks the banana loose from ${target}.`;
      case 'hold':
        return `${target} hangs on through ${attacker}'s bump.`;
      case 'attacker_fumbles':
        return `${attacker} mistimes the bump on ${target}.`;
      default:
        return `${attacker} and ${target} get a house-rule ruling.`;
    }
  }
  return resolution.note ?? `${resolution.conflictId}: ${resolution.outcome}`;
}

function parseConflictKey(
  conflictId: string
): { kind: 'pickup'; objectId: string } | { kind: 'forced_drop'; attackerAgentId: string; targetAgentId: string } | null {
  const [prefix, first, second] = conflictId.split(':');
  if (prefix === 'pickup' && first) {
    return { kind: 'pickup', objectId: first };
  }
  if (prefix === 'drop' && first && second) {
    return { kind: 'forced_drop', attackerAgentId: first, targetAgentId: second };
  }
  return null;
}

function createInitialDirectorState(note: string | null): DirectorPanelState {
  return {
    mode: 'idle',
    tick: null,
    headline: 'Watching the field',
    detail: 'No ruling in progress.',
    note,
  };
}
