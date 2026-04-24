//////////////////////////////////////////////////////////////////////////////
//
// App.tsx
//
// - Top-level simulation app. Wires:
//     - a single shared CogentEngine, loaded from the configured .gguf URL
//     - a DirectorRuntime from `director.json`
//     - four CharacterAgent-backed chooser adapters
//     - an app-local SimulationRuntime that owns the world loop
//     - a SimulationCanvas that mirrors tick-end snapshots into three.js
//     - a side panel with transport, event log, and an agent inspector
//
//////////////////////////////////////////////////////////////////////////////

import { useCallback, useEffect, useMemo, useRef, useState, useSyncExternalStore } from 'react';
import { CogentEngine, getBundledRuntimeUrls } from '@noumena-labs/cogent-engine';
import { createDirectorFromConfigUrl } from '@noumena-labs/cogent-engine/orchestrator';
import { BrainActivityHud } from './components/BrainActivityHud';
import { BrainTraceDrawer } from './components/BrainTraceDrawer';
import { SimulationCanvas } from './components/SimulationCanvas';
import { ControlsPanel } from './components/ControlsPanel';
import { StartPanel, type StartScenarioSettings } from './components/StartPanel';
import { EventLog, type EventLogEntry } from './components/EventLog';
import { AgentInspector } from './components/AgentInspector';
import { Scoreboard } from './components/Scoreboard';
import { TutorialOverlay, type TutorialCardPlacement } from './components/TutorialOverlay';
import { COURTYARD_AGENTS, COURTYARD_SCENARIO, createCourtyardScenario } from './scenarios/courtyard-snack.js';
import { SimulationBus, type SimulationEvent } from './runtime/bus.js';
import {
  BrainActivityStore,
  type BrainDefinition,
} from './runtime/brain-activity-store.js';
import {
  createSimulationAgentChooserFromConfigUrl,
} from './runtime/agent-chooser.js';
import { createTracedBrainEngine } from './runtime/traced-engine.js';
import { SimulationRuntime } from './runtime/simulation-runtime.js';
import type {
  DirectorDecision,
  DirectorResolution,
  ScenarioSeed,
  SimulationAgentState,
  SimulationGameEvent,
  WorldConflict,
  WorldSnapshot,
} from './runtime/types.js';
import type { HoveredSceneObject } from './scene/world-binding.js';

interface LoadedHarness {
  readonly engine: CogentEngine;
  readonly runtime: SimulationRuntime;
  readonly scenario: ScenarioSeed;
}

interface DirectorPanelState {
  readonly mode: 'idle' | 'ruling' | 'update';
  readonly tick: number | null;
  readonly headline: string;
  readonly detail: string | null;
  readonly note: string | null;
}

interface TutorialStep {
  readonly id: string;
  readonly target: string | null;
  readonly placement: TutorialCardPlacement;
  readonly title: string;
  readonly body: string;
}

const BRAIN_DEFINITIONS: readonly BrainDefinition[] = [
  ...COURTYARD_AGENTS.map((agent) => ({
    id: agent.agentId,
    label: agent.name,
    kind: 'agent' as const,
    accentColor: agent.color,
  })),
  {
    id: 'director',
    label: 'Director',
    kind: 'director' as const,
    accentColor: '#ffd166',
  },
];

const DEFAULT_MODEL_URL = 'https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF/resolve/main/LFM2.5-350M-Q8_0.gguf';
const SIMULATION_STEP_SECONDS = 0.15;
const SIMULATION_STEP_DELAY_MS = SIMULATION_STEP_SECONDS * 1000;
const DEFAULT_SCENARIO_SETTINGS: StartScenarioSettings = {
  obstaclesEnabled: true,
  obstacleTarget: 12,
  batsEnabled: true,
  batTarget: 1,
  iceCubesEnabled: true,
  iceCubeTarget: 1,
};
const TUTORIAL_STORAGE_KEY = 'simulation.tutorial.dismissed';
const TUTORIAL_STEPS: readonly TutorialStep[] = [
  {
    id: 'overview',
    target: null,
    placement: 'center',
    title: 'What this simulation is',
    body: 'Banana Dash is the proof of concept. Five local LLM-powered brains operate live in a shared environment with real-time decisions, arbitration, and observability. This is not a chatbot demo. It shows how local inference can power proactive systems that monitor context, adapt behavior, surface help before the user asks, and enable hybrid products where long-running local LLM observers work alongside intelligent cloud models.',
  },
  {
    id: 'controls',
    target: 'controls',
    placement: 'right',
    title: 'Control panel',
    body: 'Use Start to begin the live match, Pause to stop the loop, Step to advance one simulation slice at a time, and Reset to rebuild the scenario from scratch.',
  },
  {
    id: 'brain-hud',
    target: 'brain-hud',
    placement: 'top',
    title: 'LLM query visualizer',
    body: 'This panel shows the live query traffic for all five brains. You can see which brain is active, how many queries have run, how long they take, and inspect the latest streamed output.',
  },
  {
    id: 'event-log',
    target: 'event-log',
    placement: 'top',
    title: 'Event log',
    body: 'The event log is the readable play-by-play. It summarizes agent choices, director rulings, score events, and runtime issues as the match unfolds.',
  },
  {
    id: 'director',
    target: 'director',
    placement: 'bottom',
    title: 'Director panel',
    body: 'The director is the fifth brain. It narrates the scene and steps in when the simulation needs a ruling on a contested interaction or house-rule decision.',
  },
  {
    id: 'scoreboard',
    target: 'scoreboard',
    placement: 'bottom',
    title: 'Scoreboard',
    body: 'The scoreboard tracks who is holding the banana and how many deliveries each agent has converted into points.',
  },
  {
    id: 'agents',
    target: 'agents',
    placement: 'left',
    title: 'Agents panel',
    body: 'This panel shows each agent\'s live state: position, current activity, goal, executor intent, held items, and status. Click an agent to highlight them in the scene.',
  },
  {
    id: 'goal',
    target: 'game-board',
    placement: 'bottom',
    title: 'Game board and scoring goal',
    body: 'This is the game board. Agents score by carrying the banana into the glowing home base scoring ring. The tutorial highlights that home base while this step is active.',
  },
];

export default function App() {
  const bus = useMemo(() => new SimulationBus(), []);
  const brainStore = useMemo(() => new BrainActivityStore(BRAIN_DEFINITIONS), []);
  const appRef = useRef<HTMLDivElement | null>(null);
  const inspectorRef = useRef<HTMLDivElement | null>(null);
  const [modelUrl, setModelUrl] = useState(DEFAULT_MODEL_URL);
  const [scenarioSettings, setScenarioSettings] = useState<StartScenarioSettings>(DEFAULT_SCENARIO_SETTINGS);
  const [status, setStatus] = useState('Idle. Press Load to initialize the model.');
  const [busy, setBusy] = useState(false);
  const [harness, setHarness] = useState<LoadedHarness | null>(null);
  const [activeScenario, setActiveScenario] = useState<ScenarioSeed | null>(null);
  const [running, setRunning] = useState(false);
  const [snapshot, setSnapshot] = useState<WorldSnapshot | null>(null);
  const [events, setEvents] = useState<EventLogEntry[]>([]);
  const [selectedAgentId, setSelectedAgentId] = useState<string | null>(null);
  const [selectedBrainId, setSelectedBrainId] = useState<string | null>(null);
  const [hasSimulationStarted, setHasSimulationStarted] = useState(false);
  const [brainHudExpanded, setBrainHudExpanded] = useState(false);
  const [hoveredObject, setHoveredObject] = useState<HoveredSceneObject | null>(null);
  const [eventLogCollapsed, setEventLogCollapsed] = useState(true);
  const [tutorialOpen, setTutorialOpen] = useState(false);
  const [tutorialStepIndex, setTutorialStepIndex] = useState(0);
  const [tutorialDismissed, setTutorialDismissed] = useState(() => readTutorialDismissedPreference());
  const eventIdRef = useRef(0);
  const snapshotRef = useRef<WorldSnapshot | null>(null);
  const criticalIssueRef = useRef(false);
  const [directorState, setDirectorState] = useState<DirectorPanelState>(() =>
    createInitialDirectorState(COURTYARD_SCENARIO.directorNote ?? null)
  );
  const brainActivity = useSyncExternalStore(
    brainStore.subscribe,
    brainStore.getSnapshot,
    brainStore.getSnapshot
  );

  const pushEvent = useCallback((entry: Omit<EventLogEntry, 'id'>) => {
    const id = ++eventIdRef.current;
    setEvents((prev) => [...prev.slice(-199), { ...entry, id }]);
  }, []);

  const handleExpandBrainHud = useCallback((): void => {
    setBrainHudExpanded(true);
  }, []);

  const handleCollapseBrainHud = useCallback((): void => {
    setBrainHudExpanded(false);
    setSelectedBrainId(null);
  }, []);

  const handleSelectBrain = useCallback((brainId: string): void => {
    setSelectedBrainId((prev) => prev === brainId ? null : brainId);
  }, []);

  const openTutorial = useCallback((stepIndex = 0): void => {
    setHoveredObject(null);
    setSelectedAgentId(null);
    setSelectedBrainId(null);
    setTutorialStepIndex(stepIndex);
    setTutorialOpen(true);
  }, []);

  const closeTutorial = useCallback((): void => {
    setTutorialOpen(false);
  }, []);

  const handleTutorialNext = useCallback((): void => {
    if (tutorialStepIndex >= TUTORIAL_STEPS.length - 1) {
      setTutorialOpen(false);
      return;
    }
    setTutorialStepIndex((prev) => prev + 1);
  }, [tutorialStepIndex]);

  const handleTutorialPreferenceChange = useCallback((value: boolean): void => {
    setTutorialDismissed(value);
  }, []);

  const resetSimulationUi = useCallback((): void => {
    eventIdRef.current = 0;
    criticalIssueRef.current = false;
    snapshotRef.current = null;
    setSnapshot(null);
    setEvents([]);
    brainStore.reset();
    setHasSimulationStarted(false);
    setBrainHudExpanded(false);
    setTutorialOpen(false);
    setTutorialStepIndex(0);
    setSelectedBrainId(null);
    setSelectedAgentId(null);
    setHoveredObject(null);
    setEventLogCollapsed(true);
    setDirectorState(createInitialDirectorState(COURTYARD_SCENARIO.directorNote ?? null));
  }, [brainStore]);

  // Subscribe to bus once.
  useEffect(() => {
    const off = bus.onAny((event: SimulationEvent) => {
      switch (event.kind) {
        case 'tick-end':
        case 'world-sync':
          brainStore.setCurrentTick(event.tick);
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
          } else if (event.queryStatus === 'aborted') {
            pushEvent({
              tick: event.tick,
              kind: 'query',
              text: `${nameOf(event.agentId, snapshotRef.current)}'s turn gets interrupted.`,
            });
          }
          if (event.queryStatus !== 'ok') {
            brainStore.reviseLatestQuery(event.agentId, {
              status: mapSimulationQueryStatus(event.queryStatus),
              errorMessage: event.errorMessage,
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
        case 'runtime-error': {
          const text = `${event.severity === 'critical' ? 'Critical' : 'Warning'} ${event.source} task issue: ${event.message}`;
          const targetBrainId = event.source === 'agent' ? event.agentId ?? null : 'director';
          if (targetBrainId) {
            brainStore.reviseLatestQuery(targetBrainId, {
              status: classifyRuntimeIssueStatus(event.message),
              errorMessage: event.message,
            });
          }
          if (event.severity === 'critical') {
            criticalIssueRef.current = true;
            setRunning(false);
            setStatus(`Paused: ${event.message}`);
            setDirectorState((prev) => ({
              ...prev,
              mode: 'update',
              tick: event.tick,
              headline: 'Runtime query error',
              detail: event.message,
            }));
          }
          pushEvent({
            tick: event.tick,
            kind: event.severity === 'critical' ? 'error' : 'note',
            text,
          });
          break;
        }
      }
    });
    return () => off();
  }, [brainStore, bus, pushEvent]);

  useEffect(() => {
    window.localStorage.setItem(TUTORIAL_STORAGE_KEY, tutorialDismissed ? '1' : '0');
  }, [tutorialDismissed]);

  useEffect(() => {
    if (!tutorialOpen) {
      return;
    }
    setHoveredObject(null);
    const activeStep = TUTORIAL_STEPS[tutorialStepIndex];
    if (activeStep?.id === 'brain-hud') {
      setBrainHudExpanded(true);
    }
  }, [tutorialOpen, tutorialStepIndex]);

  useEffect(() => {
    if (!running || !harness) {
      return;
    }
    let disposed = false;
    let timeoutId: ReturnType<typeof setTimeout> | null = null;

    const loop = async (): Promise<void> => {
      if (disposed) return;
      await harness.runtime.step(SIMULATION_STEP_SECONDS);
      await harness.runtime.waitForIdle();
      if (criticalIssueRef.current) return;
      if (disposed) return;
      timeoutId = setTimeout(() => {
        void loop();
      }, SIMULATION_STEP_DELAY_MS);
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
  }, [harness, running]);

  const loadHarness = useCallback(
    async (url: string): Promise<void> => {
      setBusy(true);
      setModelUrl(url);
      setRunning(false);
      try {
        if (harness) {
          setHarness(null);
          await harness.runtime.dispose();
          harness.engine.close();
        }
        const scenario = createCourtyardScenario({
          obstacles: {
            enabled: scenarioSettings.obstaclesEnabled,
            target: scenarioSettings.obstacleTarget,
          },
          bats: {
            enabled: scenarioSettings.batsEnabled,
            target: scenarioSettings.batTarget,
          },
          iceCubes: {
            enabled: scenarioSettings.iceCubesEnabled,
            target: scenarioSettings.iceCubeTarget,
          },
        });
        setActiveScenario(scenario);
        resetSimulationUi();
        setStatus('Initialising engine…');
        const engine = new CogentEngine({ ...getBundledRuntimeUrls() });
        await engine.initModule();

        setStatus('Downloading model…');
        const modelPath = await engine.loadModelFromUrl(url, 'model.gguf', (pct) =>
          setStatus(`Downloading model… ${Math.floor(pct)}%`)
        );

        setStatus('Initialising inference runtime…');
        await engine.initEngine(modelPath, {
          enableRuntimeObservability: true,
          sampling: {
            temperature: 0.5,
            topP: 0.9,
            topK: 40,
            minP: 0.05,
            repeatPenalty: 1.05,
          },
        });

        setStatus('Loading director config…');
        const directorBrain = BRAIN_DEFINITIONS.find((brain) => brain.id === 'director');
        if (!directorBrain) {
          throw new Error('director brain definition is missing');
        }
        const { director } = await createDirectorFromConfigUrl({
          configUrl: scenario.directorConfigUrl,
          engine: createTracedBrainEngine(engine, brainStore, directorBrain),
          runtimeOptions: { maxOutputTokens: 96 },
        });

        setStatus('Building simulation runtime…');
        const runtime = new SimulationRuntime(director, {
          id: scenario.id,
          bus,
          bounds: scenario.bounds,
          game: scenario.game,
          directorCadenceTicks: scenario.directorCadenceTicks,
          initialDirectorNote: scenario.directorNote ?? null,
          resolveRefereeTask: scenario.resolveRefereeTask,
          narrateTask: scenario.narrateTask,
          refereeTimeoutMs: scenario.refereeTimeoutMs,
          narrationTimeoutMs: scenario.narrationTimeoutMs,
          agentQueryTimeoutMs: scenario.agentQueryTimeoutMs,
        });
        for (const seed of scenario.objects) {
          runtime.upsertObject(seed);
        }

        setStatus('Loading agent personas…');
        for (const assignment of COURTYARD_AGENTS) {
          const brain = BRAIN_DEFINITIONS.find((entry) => entry.id === assignment.agentId);
          if (!brain) {
            throw new Error(`brain definition missing for ${assignment.agentId}`);
          }
          const { agent } = await createSimulationAgentChooserFromConfigUrl({
            agentId: assignment.agentId,
            configUrl: assignment.characterUrl,
            engine: createTracedBrainEngine(engine, brainStore, brain),
          });
          const seed = scenario.agents.find((a) => a.id === assignment.agentId);
          if (!seed) {
            throw new Error(`no scenario seed for ${assignment.agentId}`);
          }
          runtime.addAgent(agent, seed);
        }

        const initialSnapshot = runtime.getSnapshot();
        snapshotRef.current = initialSnapshot;
        setSnapshot(initialSnapshot);
        setDirectorState(createInitialDirectorState(initialSnapshot.directorNote));
        setHarness({ engine, runtime, scenario });
        setBrainHudExpanded(true);
        if (!tutorialDismissed) {
          setTutorialStepIndex(0);
          setTutorialOpen(true);
        }
        setStatus('Ready. Press Start.');
      } catch (error) {
        console.error(error);
        setActiveScenario(null);
        setStatus(`Load failed: ${(error as Error).message}`);
      } finally {
        setBusy(false);
      }
    },
    [brainStore, bus, harness, resetSimulationUi, scenarioSettings, tutorialDismissed]
  );

  const handleStart = (): void => {
    if (!harness) return;
    criticalIssueRef.current = false;
    setHasSimulationStarted(true);
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
    criticalIssueRef.current = false;
    setHasSimulationStarted(true);
    setStatus('Stepping…');
    await harness.runtime.step(SIMULATION_STEP_SECONDS);
    await harness.runtime.waitForIdle();
    if (criticalIssueRef.current) return;
    setStatus('Stepped.');
  };

  const handleReset = async (): Promise<void> => {
    if (!harness) return;
    setBusy(true);
    setRunning(false);
    setStatus('Resetting…');
    const currentHarness = harness;
    setHarness(null);
    try {
      await currentHarness.runtime.dispose();
      currentHarness.engine.close();
      setActiveScenario(null);
      resetSimulationUi();
      setStatus('Reset. Press Load to rebuild.');
    } finally {
      setBusy(false);
    }
  };

  const agents: readonly SimulationAgentState[] = snapshot?.agents ?? [];
  const scenario = activeScenario ?? harness?.scenario ?? COURTYARD_SCENARIO;
  const scoreboardStatus = directorState.mode === 'ruling'
    ? 'director adjudicating a call'
    : 'race in progress';
  const directorDetail = directorState.detail?.trim() || null;
  const directorNote = directorState.note?.trim() || null;
  const showDirectorNote = directorNote != null && directorNote !== directorDetail;
  const highlightStart = harness != null && !busy && !running && !hasSimulationStarted;
  const tutorialStep = tutorialOpen ? TUTORIAL_STEPS[tutorialStepIndex] ?? null : null;
  const activeTutorialTarget = tutorialStep?.target ?? null;
  const highlightedObjectId = tutorialStep?.id === 'goal'
    ? scenario.game.goalObjectId
    : hoveredObject?.id ?? null;
  const topCenterFocused = activeTutorialTarget === 'scoreboard' || activeTutorialTarget === 'director';
  const tutorialTargetClass = (targetId: string, baseClass = ''): string => {
    const activeClass = activeTutorialTarget === targetId ? ' tutorial-focus' : '';
    return `${baseClass}${activeClass}`.trim();
  };

  useEffect(() => {
    const app = appRef.current;
    if (!app) {
      return;
    }

    const handlePointerDown = (event: PointerEvent): void => {
      if (selectedAgentId == null) {
        return;
      }
      const target = event.target;
      if (!(target instanceof Node)) {
        return;
      }
      if (inspectorRef.current?.contains(target)) {
        return;
      }
      setSelectedAgentId(null);
    };

    app.addEventListener('pointerdown', handlePointerDown);
    return () => app.removeEventListener('pointerdown', handlePointerDown);
  }, [selectedAgentId]);

  return (
    <div ref={appRef} className="sim-app">
      <div className={tutorialTargetClass('game-board', 'sim-board')} data-tutorial-target="game-board">
        <SimulationCanvas
          bus={bus}
          bounds={scenario.bounds ?? { halfExtent: 8 }}
          highlightedAgentId={selectedAgentId}
          highlightedObjectId={highlightedObjectId}
          onBackgroundClick={() => setSelectedAgentId(null)}
          onHoverObject={setHoveredObject}
          snapshot={snapshot}
        />
      </div>

      {harness ? (
        <div className={tutorialTargetClass('controls', 'sim-overlay sim-top-left')} data-tutorial-target="controls">
          <ControlsPanel
            onStart={handleStart}
            onPause={handlePause}
            onStep={handleStep}
            onReset={handleReset}
            onOpenTutorial={() => openTutorial(0)}
            status={status}
            running={running}
            tick={snapshot?.tick ?? 0}
            highlightStart={highlightStart}
          />
        </div>
      ) : (
        <div className="sim-overlay sim-start">
          <StartPanel
            modelUrl={modelUrl}
            onModelUrlChange={setModelUrl}
            scenarioSettings={scenarioSettings}
            onScenarioSettingsChange={setScenarioSettings}
            onLoad={loadHarness}
            status={status}
            busy={busy}
          />
        </div>
      )}

      {harness ? (
        <>
          <div ref={inspectorRef} className={tutorialTargetClass('agents', 'sim-overlay sim-top-right')} data-tutorial-target="agents">
            <AgentInspector
              agents={agents}
              bananaObjectId={snapshot?.game.bananaObjectId}
              tick={snapshot?.tick ?? 0}
              selectedAgentId={selectedAgentId}
              onSelect={setSelectedAgentId}
            />
          </div>

          <div className={tutorialTargetClass('event-log', 'sim-overlay sim-bottom')} data-tutorial-target="event-log">
            <EventLog
              entries={events}
              collapsed={eventLogCollapsed}
              onToggle={() => setEventLogCollapsed((value) => !value)}
            />
          </div>
        </>
      ) : null}

      {harness ? (
        <div className={tutorialTargetClass('brain-hud', 'sim-overlay sim-bottom-left')} data-tutorial-target="brain-hud">
          <BrainActivityHud
            activity={brainActivity}
            expanded={brainHudExpanded}
            selectedBrainId={selectedBrainId}
            onExpand={handleExpandBrainHud}
            onCollapse={handleCollapseBrainHud}
            onSelectBrain={handleSelectBrain}
          />
        </div>
      ) : null}

      {harness && snapshot ? (
        <div className={`sim-overlay sim-top-center sim-center-stack${topCenterFocused ? ' tutorial-focus-host' : ''}`}>
          <div className={tutorialTargetClass('scoreboard')} data-tutorial-target="scoreboard">
            <Scoreboard snapshot={snapshot} metaText={scoreboardStatus} />
          </div>
          <div className={tutorialTargetClass('director')} data-tutorial-target="director">
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
        </div>
      ) : null}

      {harness ? (
        <BrainTraceDrawer
          activity={brainActivity}
          selectedBrainId={selectedBrainId}
          onClose={() => setSelectedBrainId(null)}
        />
      ) : null}

      {harness && hoveredObject && !tutorialOpen ? (
        <div className="sim-overlay sim-bottom-right">
          <div className="hover-card glass-panel">
            <span className="panel-eyebrow">Inspecting</span>
            <span className="hover-card-title">{hoveredObject.label}</span>
            <span className="hover-card-copy">{hoveredObject.description}</span>
          </div>
        </div>
      ) : null}

      {tutorialStep ? (
        <TutorialOverlay
          open={tutorialOpen}
          title={tutorialStep.title}
          body={tutorialStep.body}
          target={tutorialStep.target}
          placement={tutorialStep.placement}
          stepIndex={tutorialStepIndex}
          stepCount={TUTORIAL_STEPS.length}
          neverShowAgain={tutorialDismissed}
          onNext={handleTutorialNext}
          onDismiss={closeTutorial}
          onNeverShowAgainChange={handleTutorialPreferenceChange}
        />
      ) : null}
    </div>
  );
}

function readTutorialDismissedPreference(): boolean {
  return window.localStorage.getItem(TUTORIAL_STORAGE_KEY) === '1';
}

function describeGameEvent(
  event: SimulationGameEvent,
  snapshot: WorldSnapshot | null
): string | null {
  switch (event.kind) {
    case 'pickup':
      return `${nameOf(event.agentId, snapshot)} grabs the ${objectLabelOf(event.objectId, snapshot)}.`;
    case 'delivery':
      return `${nameOf(event.agentId, snapshot)} scores by delivering the banana!`;
    case 'respawn':
      return `The ${objectLabelOf(event.objectId, snapshot)} pops back into play at (${event.position.x.toFixed(1)}, ${event.position.z.toFixed(1)}).`;
    case 'drop':
      return `${nameOf(event.agentId, snapshot)} drops the banana${event.cause === 'bump' ? ' after a bump' : event.cause === 'bat' ? ' after a bonk' : event.cause === 'ice' ? ' after freezing up' : ''}.`;
    case 'forced_drop':
      if (event.outcome === 'drop') {
        return `${nameOf(event.attackerAgentId, snapshot)} bumps ${nameOf(event.targetAgentId, snapshot)} and the banana flies loose.`;
      }
      if (event.outcome === 'attacker_fumbles') {
        return `${nameOf(event.attackerAgentId, snapshot)} fumbles the bump on ${nameOf(event.targetAgentId, snapshot)}.`;
      }
      return `${nameOf(event.targetAgentId, snapshot)} keeps hold through the bump.`;
    case 'bump_whiff':
      return `${nameOf(event.attackerAgentId, snapshot)} lunges at ${nameOf(event.targetAgentId, snapshot)} and whiffs the bump.`;
    case 'push':
      return `${nameOf(event.agentId, snapshot)} pushes ${nameOf(event.targetAgentId, snapshot)} away.`;
    case 'power_up_throw':
      return `${nameOf(event.agentId, snapshot)} hurls an ice cube at ${nameOf(event.targetAgentId, snapshot)}.`;
    case 'bat_swing': {
      const hitNames = event.hits.map((hit) => nameOf(hit.agentId, snapshot));
      return hitNames.length > 0
        ? `${nameOf(event.agentId, snapshot)} swings the bat and sends ${hitNames.join(', ')} flying.`
        : `${nameOf(event.agentId, snapshot)} swings the bat through open air.`;
    }
    case 'power_up_use':
      return `${nameOf(event.targetAgentId, snapshot)} gets encased in ice after ${nameOf(event.agentId, snapshot)}'s throw lands.`;
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

function mapSimulationQueryStatus(
  status: import('./runtime/bus.js').AgentQueryEndEvent['queryStatus']
): 'cancelled' | 'timed_out' | 'failed' {
  switch (status) {
    case 'aborted':
      return 'cancelled';
    case 'timed_out':
      return 'timed_out';
    case 'failed':
    case 'invalid_response':
      return 'failed';
    case 'ok':
      return 'failed';
  }
}

function classifyRuntimeIssueStatus(message: string): 'timed_out' | 'failed' {
  return message.toLowerCase().includes('timed out') ? 'timed_out' : 'failed';
}
