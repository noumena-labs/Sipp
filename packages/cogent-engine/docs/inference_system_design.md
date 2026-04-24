# CogentEngine Codebase Visualization

This document presents a comprehensive set of visualizations for the CogentEngine architecture using Mermaid diagrams. The visualizations are structured from the high-level system boundary down to the nuanced function calls of the native inference scheduler.

---

## 1. High-Level System Architecture

`CogentEngine` is a facade that provides a unified interface for model lifecycle management (`CogentModelManager`), observability, and inference execution. `CogentEngine.create()` selects the underlying implementation based on the environment and configuration. When running in a browser environment with `Worker` support (and no explicit `executionMode` override), it delegates to a `WorkerModelServiceClient`. On Node.js or when `executionMode: 'main-thread'` is set, it uses `ModelService` directly with a `MainThreadEngineRuntime`. Runtime asset URLs are resolved lazily inside the chosen execution context. If callers do not provide explicit `{ moduleUrl, wasmUrl }`, the runtime falls back to the package-bundled assets from inside that context.

All inference-critical logic — WASM module management, native bridge calls, and request tracking — lives inside `MainThreadEngineRuntime`. The worker-backed path instantiates the same runtime class inside a Web Worker, communicates through a structured `postMessage` protocol (defined in `WorkerRequestMessage`/`WorkerResponseMessage`), and exposes the same `ModelLifecycleService` interface.

```mermaid
graph TD
    subgraph UserSpace ["User Application"]
        App["Application Code"]
    end

    subgraph SDK ["CogentEngine SDK (src/)"]
        CE["CogentEngine<br/>(Public Facade)"]
        
        subgraph Managers ["Public Managers"]
            MM["CogentModelManager<br/>(Models facade)"]
            OM["RuntimeObservability<br/>(Observability facade)"]
        end

        subgraph ServiceLayer ["Model Lifecycle Service"]
            WorkerClient["WorkerModelServiceClient<br/>(Main thread proxy)"]
            Service["ModelService<br/>(Registry & Asset management)"]
        end

        subgraph CoreRuntimes ["Engine Runtimes"]
            MainRT["MainThreadEngineRuntime<br/>(WASM & Scheduler controller)"]
        end

        subgraph WorkerThread ["Web Worker Thread"]
            Entry["model-service-entry.ts<br/>(Worker message dispatcher)"]
            WorkerService["ModelService (inside worker)"]
            InnerMain["MainThreadEngineRuntime (inside worker)"]
        end
    end

    subgraph Persistence ["Persistence Layer (OPFS)"]
        Registry["ModelRegistryStore<br/>(registry.json)"]
        Assets["AssetStore<br/>(Raw downloaded files)"]
    end

    subgraph NativeBridge ["WASM Bridge & Exports"]
        Bridge["WasmBridge<br/>(Emscripten Abstraction)"]
        WasmExports["CE_Init / CE_Close<br/>CE_StartTextRequest / CE_CancelRequest<br/>CE_RunSchedulerBurst / CE_DrainRuntimeEvents"]
    end

    subgraph Native ["Native Inference Runtime (C++)"]
        IR["InferenceRuntime"]
        RQ["RequestQueue"]
        SlotSched["SlotScheduler"]
        Session["SessionStore + PrefixStateCache"]
    end

    App --> CE
    CE --> MM
    CE --> OM
    MM --> |executionMode=worker| WorkerClient
    MM --> |executionMode=main-thread| Service
    OM --> WorkerClient
    OM --> Service

    WorkerClient <--> |"postMessage protocol"| Entry
    Entry --> WorkerService
    WorkerService --> InnerMain
    
    Service --> Registry
    Service --> Assets
    Service --> MainRT

    InnerMain --> Bridge
    MainRT --> Bridge
    Bridge --> WasmExports

    WasmExports --> IR
    IR --> RQ
    IR --> SlotSched
    IR --> Session
```

---

## 2. Model Asset Management & Loading

`ModelService` completely manages model and projector assets. It downloads remote assets, caches them in an OPFS-backed `AssetStore`, and tracks installed models plus projector pairing state in a `ModelRegistryStore`. Installed model ids identify the persisted base-model entry, not a temporary runtime mount. Successful projector pairings are stored on that entry and reused later; unresolved pairing scans are cached against the current projector inventory revision and retried only after the installed projector set changes. The execution path translates a user's `ModelSource` into installed asset records and an internal runtime bundle descriptor before handing it to the runtime. Low-level bundle descriptors, mount paths, queue internals, and native scheduler details are not public API.

```mermaid
flowchart TD
    Load["engine.models.load(source)"]

    Load --> ServiceLoad["ModelService.load(source)"]
    
    subgraph StoragePipeline ["Asset Storage Pipeline"]
        Resolve["Resolve Remote Metadata<br/>(AssetStore.resolveRemoteMetadata)"]
        Download["Download & Stream to OPFS<br/>(AssetStore.downloadRemote)"]
        CacheHit{"Asset exists<br/>in Registry?"}
    end

    subgraph Preparation ["Pairing & Classification"]
        Classify["Classify Asset<br/>(PairingValidator.classify)"]
        Plan["Resolve Pairing Plan<br/>(PairingValidator.resolve)"]
        Upsert["Create ModelEntry<br/>(Registry.write)"]
    end

    ServiceLoad --> CacheHit
    CacheHit --> |No| Resolve
    Resolve --> Download
    Download --> Classify
    CacheHit --> |Yes| Classify
    
    Classify --> Plan
    Plan --> Upsert
    
    Upsert --> RunStage["MainThreadEngineRuntime.stageModelBundle()"]
    RunStage --> RunLoad["MainThreadEngineRuntime.loadRuntimeModel()"]
    RunLoad --> CEInit["bridge.loadRuntimeModel() → CE_Init/CE_InitWithMultimodal"]
```

In **worker mode**, `WorkerModelServiceClient` proxies `load` via a `models-load` message. The worker performs the fetch, OPFS writes, and `CE_Init` locally. `load-progress` messages are sent back to the main thread to drive UI progress bars.

---

## 3. Request Lifecycle & Burst Scheduling

Execution uses a **native-owned scheduling model** driven by a thin TypeScript browser event-loop pump. The TypeScript `QueuedRequestScheduler` does not own scheduling policy; it only drives the native engine in bounded bursts via `CE_RunSchedulerBurst` (or `CE_RunSchedulerBurstWithDeadline`), drains runtime events, forwards token callbacks, handles aborts, and yields for browser responsiveness.

```mermaid
sequenceDiagram
    participant App as Application
    participant MS as ModelService
    participant RT as MainThreadEngineRuntime
    participant Sched as QueuedRequestScheduler
    participant Tracker as RequestTracker
    participant Bridge as WasmBridge
    participant Wasm as WASM CE_ API

    App->>MS: engine.query(input, options)
    MS->>RT: enqueueQuery(session, prompt, options)
    
    RT->>Bridge: startTextRequest() OR startMediaRequest()
    Bridge->>Wasm: CE_StartTextRequest / CE_StartMediaRequest
    Wasm-->>Bridge: requestId
    Bridge-->>RT: requestId

    RT->>Tracker: track(requestId)
    RT->>Sched: track(requestId)
    Note over Sched: ensureRunning() starts the async pump loop
    
    RT-->>MS: requestId
    MS->>RT: awaitQuery(requestId, options)
    
    loop Async Pump Loop (QueuedRequestPump)
        Note over Sched: Calculates limits based on requests waiting for 1st token vs interactive streaming

        Sched->>Bridge: runSchedulerProgress()
        Bridge->>Wasm: CE_RunSchedulerBurst / CE_RunSchedulerBurstWithDeadline
        Wasm-->>Bridge: stepResult (PROGRESSED, WAITING, TERMINAL)

        Sched->>Bridge: drainRuntimeEvents()
        Bridge->>Wasm: CE_DrainRuntimeEvents()
        Wasm-->>Bridge: Array of TOKEN and TERMINAL events

        loop For each drained TOKEN event
            Sched->>Sched: bufferTokenPiece()
        end
        
        Sched->>Sched: flushAllQueuedTokenPieces() → Calls onToken callbacks

        loop For each drained TERMINAL event
            Sched->>Bridge: takeCompletedResponse(requestId)
            Bridge->>Wasm: CE_CopyCompletedRequestOutput / CE_GetCompletedRequestRuntimeObservability
            Bridge->>Wasm: CE_ConsumeCompletedRequest()
            Bridge-->>Sched: GenerateResponse (Text + Metrics)
            Sched->>Tracker: resolve(requestId, response)
            Sched->>RT: finalizeRequest()
        end
        
        Note over Sched: If streak ≥ idleStreakBeforeYield, yield via setTimeout
    end
    
    Tracker-->>RT: Resolved GenerateResponse
    RT-->>MS: GenerateResponse
    MS-->>App: string (outputText)
```

---

## 4. Worker Execution Path

When `executionMode` is `worker`, all asset processing, OPFS I/O, and inference happen in a background thread.

```mermaid
sequenceDiagram
    participant App as Application
    participant WMSC as WorkerModelServiceClient<br/>(Main Thread)
    participant Worker as Worker Thread<br/>(model-service-entry.ts)
    participant MS as ModelService (Worker)
    participant RT as MainThreadEngineRuntime (Worker)

    App->>WMSC: query(input)
    WMSC->>Worker: postMessage({kind:'query', callId, input})
    Worker->>MS: query(input)
    MS->>RT: enqueueQuery()
    RT->>RT: Starts queued pump loop inside worker
    
    loop During execution
        RT->>RT: flushAllQueuedTokenPieces()
        RT->>Worker: callback onToken(text)
        Worker->>WMSC: postMessage({kind:'token', callId, text})
        WMSC->>App: onToken(text)
    end
    
    MS-->>Worker: String Result
    Worker->>WMSC: postMessage({kind:'resolve', callId, value})
    WMSC-->>App: result
```

---

## 5. Observability Subsystem

CogentEngine exposes an opt-in, lifecycle-boundary observability pipeline through `EngineObservability`. This system aggregates timing, throughput, backend profiling, and lifecycle transitions without changing the public `query()` return type.

```mermaid
graph TD
    subgraph Senders
        RT["MainThreadEngineRuntime<br/>(Collects Bridge / Wasm Metrics)"]
        MS["ModelService<br/>(Emits state transitions & query boundaries)"]
    end

    subgraph Core ["ObservabilityController"]
        State["Maintains ObservabilitySnapshot<br/>(mode, state, model, query, runtime, profile)"]
        Emitter["emit() / update() / ingest()"]
    end

    subgraph Receivers
        App["App Subscription<br/>(engine.observability.subscribe)"]
    end
    
    subgraph Worker ["Worker Mode"]
        WorkerEvents["Worker Controller Emits"] --> |"postMessage(kind: 'observability-event')"| Proxy["WorkerModelServiceClient"]
        Proxy --> Core
    end

    RT --> MS
    MS --> Emitter
    Emitter --> State
    Emitter --> App
```

| Layer | Type | Handled By |
|---|---|---|
| **State Tracking** | Enum states (`idle`, `loading`, `querying`, etc.) | `ObservabilityController` |
| **RuntimeMetrics** | Detailed engine timings (TTFT, ITL, Tokens/sec, prefix cache stats) | Read from `CE_GetRuntimeObservability` memory struct |
| **BackendProfiling** | Raw llama.cpp device profiling JSON | `CE_GetBackendObservabilityJson` |
| **QueryMetrics** | High-level session success/failure timings | Built by `ModelService` during `query()` wrapper |

---

## 6. RequestScheduler & Prefix Caching (Native)

The native side manages `SlotScheduler` (which routes requests to batches) and `PrefixStateCache` (for system prompt or persistent context reuse).

```mermaid
graph TD
    subgraph RequestQueue ["Request Queue (request_queue.cpp)"]
        Pending["Pending GenerateRequests"]
    end

    subgraph SlotScheduler ["SlotScheduler (slot_scheduler.cpp)"]
        Assign["Assign Requests to Slots"]
        BuildBudget["BuildTickBudget() <br/> Decides Decode/Prefill ratio"]
    end

    subgraph BatchPlanner ["BatchPlanner (batch_planner.cpp)"]
        RoutePolicy["BuildPolicyBatch()"]
        Chunk["Chunk Contexts (adaptivePrefillChunking)"]
    end

    subgraph PrefixCache ["PrefixStateCache (prefix_state_cache.cpp)"]
        ExactMatch["Compute Rolling Token Hash"]
        Restore["Restore Session (KV Shift/Copy)"]
    end

    Pending --> Assign
    Assign --> ExactMatch
    ExactMatch --> |Hit| Restore
    Assign --> BuildBudget
    BuildBudget --> RoutePolicy
    RoutePolicy --> Chunk
    Chunk --> LlamaDecode["llama_decode()"]
```
