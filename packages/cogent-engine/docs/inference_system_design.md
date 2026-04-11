# CogentEngine Codebase Visualization

This document presents a comprehensive set of visualizations for the CogentEngine architecture using Mermaid diagrams. The visualizations are structured from the high-level system boundary down to the nuanced function calls of the native inference scheduler.

---

## 1. High-Level System Architecture

`CogentEngine` is a facade that selects between two `EngineRuntime` implementations at construction time. When running in a browser environment with `Worker` support (and no explicit `executionMode` override), it defaults to `WorkerEngineRuntime`. On Node.js or when `executionMode: 'main-thread'` is set, it uses `MainThreadEngineRuntime` directly.

All inference-critical logic — request tracking, state machine management, WASM calls — lives inside `MainThreadEngineRuntime`. The worker-backed path transfers this class into a `Worker` thread via `WorkerEntryState`, communicates through a structured `postMessage` protocol, and re-exposes the same `EngineRuntime` interface to the caller through `WorkerEngineRuntime`.

```mermaid
graph TD
    subgraph UserSpace ["User Application"]
        App["Application Code"]
    end

    subgraph SDK ["CogentEngine SDK (src/)"]
        CE["CogentEngine<br/>(Public Facade)"]

        subgraph Runtimes ["EngineRuntime Implementations"]
            WorkerRT["WorkerEngineRuntime<br/>(Main thread proxy)"]
            MainRT["MainThreadEngineRuntime<br/>(Engine controller)"]
        end

        subgraph InternalCore ["Internal Engine Core"]
            Scheduler["QueuedRequestScheduler<br/>(Burst & adaptive policy)"]
            Tracker["RequestTracker&lt;GenerateResponse&gt;<br/>(Promise lifecycle)"]
            Bridge["WasmBridge<br/>(Native call abstraction)"]
        end

        subgraph WorkerThread ["Web Worker Thread"]
            Entry["engine-runtime-worker-entry.ts<br/>(Worker message dispatcher)"]
            EntryState["WorkerEntryState<br/>(Worker-side engine + pump)"]
            InnerMain["MainThreadEngineRuntime<br/>(Inside worker, pump mode=external)"]
        end
    end

    subgraph WASM ["WASM Export Layer"]
        WasmExports["CE_Init / CE_Close<br/>CE_EnqueuePrompt / CE_CancelQueuedRequest<br/>CE_RunSchedulerBurst / CE_RunSchedulerBurstWithDeadline<br/>CE_DrainRuntimeEvents / CE_DrainCompletedRequestIds<br/>CE_GetCompletedRequestStatus / CE_CopyCompletedRequestOutput<br/>CE_GetRuntimeObservability / CE_GetBackendObservabilityJson"]
    end

    subgraph Native ["Native Inference Runtime (C++)"]
        IR["InferenceRuntime"]
        RQ["RequestQueue"]
        SlotSched["SlotScheduler"]
        Batch["BatchPlanner"]
        Session["SessionStore + PrefixStateCache"]
        Llama["llama.cpp"]
    end

    App --> CE
    CE -->|"executionMode=worker (default in browser)"| WorkerRT
    CE -->|"executionMode=main-thread"| MainRT

    WorkerRT <-->|"postMessage protocol<br/>WorkerRequestMessage / WorkerResponseMessage"| Entry
    Entry --> EntryState
    EntryState --> InnerMain

    MainRT --> Scheduler
    InnerMain --> Scheduler

    Scheduler --> Tracker
    Scheduler --> Bridge
    Bridge --> WasmExports

    WasmExports --> IR
    IR --> RQ
    IR --> SlotSched
    IR --> Session
    SlotSched --> Batch
    Batch --> Llama
```

---

## 2. Execution Mode Selection & Initialization

`CogentEngine` selects its runtime backend at construction time based on environment detection. Initialization then proceeds in two distinct phases: **module loading** (fetching & instantiating the WASM binary) and **engine initialization** (calling `CE_Init` with inference configuration).

```mermaid
flowchart TD
    New["new CogentEngine(config)"]

    New --> CheckMode{config.executionMode?}
    CheckMode -->|"'main-thread'"| UseMain["Use MainThreadEngineRuntime"]
    CheckMode -->|"'worker'"| UseWorker["Use WorkerEngineRuntime"]
    CheckMode -->|"'auto' or omitted"| AutoDetect{Browser with Worker support?}
    AutoDetect -->|Yes| UseWorker
    AutoDetect -->|No| UseMain

    UseMain --> InitModule1["initModule()<br/>Dynamic import() of moduleUrl<br/>Instantiate Emscripten EngineModule<br/>locateFile() redirects .wasm → wasmUrl"]
    InitModule1 --> InitEngine1["initEngine(modelPath, config)<br/>normalizeInitConfig()<br/>bridge.initEngine() → CE_Init(20 params)<br/>Stores runtimeObservabilityEnabled flag"]

    UseWorker --> SpawnWorker["ensureWorkerInitialized()<br/>new Worker(workerUrl)<br/>postMessage: init-module"]
    SpawnWorker --> WorkerInit["Worker receives init-module<br/>WorkerEntryState.initModule()<br/>new MainThreadEngineRuntime(config)<br/>Sets pump mode = 'external'<br/>Calls runtime.initModule()"]
    WorkerInit --> InitEngine2["initEngine(modelPath, config)<br/>postMessage: init-engine<br/>Worker calls runtime.initEngine()<br/>Sets runtimeObservabilityEnabled"]
```

---

## 3. Model Loading

`MainThreadEngineRuntime` delegates all model loading to `MainThreadModelLoader`, which writes the model data into the WASM module's virtual filesystem (MEMFS). Several source types are supported, all transparently writing to OPFS for persistent caching where available. In worker mode, file/stream data is transferred from the main thread to the worker thread via structured ArrayBuffer chunks.

```mermaid
flowchart LR
    subgraph Sources ["Model Sources"]
        URL["loadModelFromUrl()"]
        File["loadModelFromFile()"]
        Shards["loadModelFromFileShards()"]
        URLs["loadModelFromUrls()"]
        Stream["loadModelFromReadableStream()"]
        Buffer["loadModelFromBuffer()"]
    end

    subgraph Cache ["Persistence Layer"]
        OPFS["FileSystemStorage (OPFS)"]
        BrowserCache["BrowserModelCache<br/>(persistent model cache key)"]
    end

    subgraph MEMFS ["WASM Virtual FS (MEMFS)"]
        ModelFile["model.gguf (or named file)<br/>Mounted into Emscripten FS"]
    end

    URL --> BrowserCache
    File --> BrowserCache
    Shards --> BrowserCache
    URLs --> BrowserCache
    Stream --> MEMFS
    Buffer --> MEMFS

    BrowserCache --> OPFS
    OPFS --> ModelFile

    ModelFile -->|"modelPath returned<br/>to initEngine()"| Init["CE_Init(modelPath, ...)"]
```

In **worker mode**, `loadModelFromUrl`, `loadModelFromFile`, and `loadModelFromFileShards` send a `load-model-url/file/file-shards` message; `loadModelFromReadableStream` sends chunked `load-model-stream-chunk` messages with backpressure ack (`load-stream-ack`) before the final `load-model-stream-end`. The worker executes the actual download and MEMFS write inside the `WorkerEntryState`.

---

## 4. Request Lifecycle & Execution Call Flow

The execution path has been redesigned around a **burst scheduling model**. Rather than a host-native round-trip per token, the TypeScript scheduler issues a single `CE_RunSchedulerBurst` (or `CE_RunSchedulerBurstWithDeadline`) call that runs many native ticks in one WASM re-entry. Tokens and terminal signals are then batch-collected via `CE_DrainRuntimeEvents`. Requests are tracked entirely in TypeScript through `RequestTracker`; the native side manages the `RequestQueue`.

```mermaid
sequenceDiagram
    participant App as Application
    participant RT as MainThreadEngineRuntime
    participant Sched as QueuedRequestScheduler
    participant Tracker as RequestTracker
    participant Bridge as WasmBridge
    participant Wasm as WASM CE_ API
    participant IR as InferenceRuntime (C++)

    App->>RT: queuePrompt(contextKey, promptText, options)
    RT->>RT: shouldUseNativeRuntimeEvents(bridge, onToken)<br/>Decides: 'runtime-events' vs 'callbacks'

    alt Token transport = runtime-events (preferred)
        RT->>Bridge: enqueuePrompt(contextKey, text, maxTokens, callbackPtr=0)
    else Token transport = callbacks (fallback)
        RT->>Bridge: registerTokenCallback(onToken)
        RT->>Bridge: enqueuePrompt(contextKey, text, maxTokens, callbackPtr)
    end

    Bridge->>Wasm: CE_EnqueuePrompt(contextKey, text, maxTokens, ptr)
    Wasm->>IR: EnqueueRequest()
    IR-->>Wasm: requestId
    Wasm-->>Bridge: requestId
    Bridge-->>RT: requestId

    RT->>Tracker: track(requestId) → TrackedRequest{promise, ...}
    RT->>Sched: track(requestId)
    Note over Sched: If pump mode='internal',<br/>ensureRunning() starts pump loop

    RT-->>App: Promise [requestId tracked internally]

    par Optional Cancellation
        App->>RT: cancelQueuedRequest(requestId)
        RT->>Bridge: cancelQueuedRequest(requestId)
        Bridge->>Wasm: CE_CancelQueuedRequest(requestId)
        Wasm->>IR: Sets request.cancel_requested = true
        RT->>Sched: settleCompletedRequestIfPresent(bridge, id)
    end

    loop Async Pump Loop — runQueuedRequestPumpLoop()
        Note over Sched: Burst parameters are adaptive:<br/>awaitingFirstToken → tight limits (8 ticks, 1 token)<br/>interactiveStreaming → medium limits (16 ticks, 8 tokens, 80ms deadline)<br/>backgroundBurst → loose limits (64 ticks, 32 tokens)

        Sched->>Bridge: runSchedulerProgress(maxTicks, maxCompleted, maxTokens, [deadline])
        Bridge->>Wasm: CE_RunSchedulerBurstWithDeadline(...) OR CE_RunSchedulerBurst(...)
        Wasm->>IR: Runs N internal batch ticks

        loop Each tick inside native burst
            IR->>IR: SlotScheduler.Tick() → check cancel_requested
            IR->>IR: BatchPlanner.BuildPolicyBatch()
            IR->>IR: llama_decode(batch)
            IR->>IR: llama_sampler_sample() → tokens
            IR->>IR: Push token & terminal events to native event queue
        end

        Wasm-->>Bridge: WasmSchedulerProgressResult{stepResult, completedResponseCount}

        Bridge->>Wasm: CE_DrainRuntimeEvents(eventBuf, count, textBuf, textSize, resultPtr)
        Wasm-->>Bridge: Array of RuntimeEvents (TOKEN | TERMINAL) + decoded text

        loop For each TOKEN event
            Bridge->>Sched: bufferTokenPiece(requestId, token)
            Sched->>Sched: queuedPromptTokenBuffers.push(token)
        end

        Sched->>Sched: flushAllQueuedTokenPieces() → calls onToken callbacks<br/>Removes from requestsAwaitingFirstToken set

        loop For each TERMINAL event (or CE_DrainCompletedRequestIds fallback)
            Sched->>Bridge: getCompletedRequestStatus(requestId)
            Bridge->>Wasm: CE_GetCompletedRequestStatus(requestId)
            Wasm-->>Bridge: status (COMPLETED | CANCELLED | FAILED)
            Bridge->>Wasm: CE_CopyCompletedRequestOutput / CE_CopyCompletedRequestError
            Bridge->>Wasm: CE_GetCompletedRequestRuntimeObservability
            Bridge->>Wasm: CE_ConsumeCompletedRequest(requestId)
            Bridge-->>Sched: GenerateResponse{outputText, cancelled, failed, observability}
            Sched->>Tracker: resolve(requestId, response)
            Sched->>RT: finalizeRequest(bridge, requestId)
            RT->>Bridge: unregisterCallback(callbackPtr)
            Note over Tracker: TrackedRequest.settled = true<br/>Promise resolves for awaiter
        end

        alt stepResult = INVALID or FATAL_NO_PROGRESS
            Sched->>Tracker: rejectAll(error)
        end
    end

    App->>RT: runQueuedRequest(requestId)
    RT->>Tracker: get(requestId) → tracked.promise
    Tracker-->>RT: resolved GenerateResponse
    RT-->>App: GenerateResponse
```

---

## 5. Worker Execution Path & Message Protocol

When using `WorkerEngineRuntime`, all inference execution is offloaded to a dedicated `Worker` thread. The main thread sends typed `WorkerRequestMessage` objects and receives typed `WorkerResponseMessage` objects. Tokens are streamed as `token` messages using a coalescing buffer inside `WorkerEntryState` to minimize cross-thread IPC overhead.

```mermaid
sequenceDiagram
    participant App as Application
    participant WRT as WorkerEngineRuntime<br/>(Main Thread)
    participant Worker as Worker Thread<br/>(engine-runtime-worker-entry.ts)
    participant ES as WorkerEntryState
    participant InnerRT as MainThreadEngineRuntime<br/>(pump='external')

    App->>WRT: queuePrompt(contextKey, text, options)
    WRT->>Worker: postMessage({kind:'queue-prompt', callId, contextKey, text, options})
    Worker->>ES: handleQueuePrompt(message)
    ES->>InnerRT: queuePrompt(contextKey, text, {onToken, signal})
    Note over ES: onToken → bufferTokenPiece(requestId, token)<br/>Coalesces up to bufferedTokenLimit tokens OR flushIntervalMs timeout
    InnerRT-->>ES: requestId
    ES->>ES: markRequestRunning(requestId)
    ES->>ES: ensureSchedulerPumpRunning() → starts runQueuedRequestPumpLoop
    Worker-->>WRT: postMessage({kind:'resolve', callId, value: requestId})
    WRT-->>App: requestId (tracked in WorkerEngineRuntime.tracker)

    loop Scheduler pump (external mode, worker thread)
        ES->>InnerRT: scheduler.pumpOnce() → runSchedulerProgress + drainRuntimeEvents
        ES->>ES: emitSettledQueuedRequests(runtime)
        ES->>ES: flushBufferedTokens(requestId)
        ES->>Worker: postMessage({kind:'token', requestId, text, bufferedTokenCount})
        Worker->>WRT: WorkerResponseMessage token
        WRT->>WRT: queuedTokenCallbacks.get(requestId)(text)
        WRT->>App: onToken(text) callback fires
    end

    ES->>ES: Settlement handler fires on request completion
    alt Success
        ES->>Worker: postMessage({kind:'request-complete', requestId, result})
        Worker->>WRT: WorkerResponseMessage request-complete
        WRT->>WRT: settleQueuedRequestCompletion(requestId, result)
        WRT->>WRT: tracker.resolve(requestId, result)
    else Failure / Callback error
        ES->>Worker: postMessage({kind:'request-failed', requestId, message, errorName})
        Worker->>WRT: WorkerResponseMessage request-failed
        WRT->>WRT: rejectQueuedRequestCompletion(requestId, error)
    end

    App->>WRT: runQueuedRequest(requestId)
    WRT->>WRT: tracker.get(requestId).promise
    WRT-->>App: GenerateResponse
```

---

## 6. QueuedRequestScheduler — Adaptive Burst Policy

The `QueuedRequestScheduler` controls how aggressively the native burst loop runs each pump step. It maintains two internal sets to classify active requests by their progress state, and uses those sets to select appropriate burst limits:

```mermaid
flowchart TD
    Tick["pumpQueuedRequestsStep()"]

    Tick --> Check1{requestsAwaitingFirstToken.size > 0?}

    Check1 -->|Yes — First-token latency mode| FTMode["Burst limits:<br/>maxTicks=8, maxEmittedTokens=1<br/>No deadline constraint<br/>Goal: minimize TTFT"]

    Check1 -->|No| Check2{interactiveStreamingRequests.size > 0?}

    Check2 -->|Yes — Interactive streaming mode| ISMode["Burst limits:<br/>maxTicks=16, maxEmittedTokens=8<br/>maxDurationUs=80,000µs<br/>CE_RunSchedulerBurstWithDeadline called"]

    Check2 -->|No — Background throughput mode| BTMode["Burst limits:<br/>maxTicks=64, maxEmittedTokens=32<br/>No deadline constraint<br/>CE_RunSchedulerBurst called"]

    FTMode --> RunBurst["runSchedulerProgress(bridge)"]
    ISMode --> RunBurst
    BTMode --> RunBurst

    RunBurst --> Drain["drainRuntimeEvents()"]
    Drain --> FlushTokens["flushAllQueuedTokenPieces()<br/>Pop from queuedPromptTokenBuffers<br/>Call onToken() callbacks"]
    FlushTokens --> UpdateSets["Remove from requestsAwaitingFirstToken<br/>if token was flushed for that request"]
    UpdateSets --> SettleTerminal["settleCompletedQueuedRequest() per terminal event"]

    SettleTerminal --> Check3{stepResult?}
    Check3 -->|PROGRESSED or TERMINAL| Continue["Continue loop"]
    Check3 -->|WAITING, no settled| IncrStreak["waitingStreak++<br/>If streak ≥ idleStreakBeforeYield → yield via setTimeout(0)"]
    Check3 -->|INVALID or FATAL| Reject["rejectPendingQueuedRequests(error)"]

    subgraph Responsiveness ["Responsiveness Yield Logic"]
        ShouldYield{"shouldYieldForResponsiveness(burstTickCount)?<br/>(burstTickCount ≥ 128 AND window+document defined)"}
        ShouldYield -->|Yes| Yield["await waitForNextSchedulerStep()<br/>setTimeout(0) — yield to event loop"]
        ShouldYield -->|No| Loop["Continue synchronous burst"]
    end
```

---

## 7. Token Transport Selection

The `MainThreadEngineRuntime` selects the token delivery mechanism per-request at enqueue time. The preferred path is `runtime-events` (zero-allocation batch drain); the fallback is a native C function pointer callback per token.

```mermaid
flowchart TD
    QP["queuePrompt(contextKey, text, options)"]
    QP --> HasCallback{onToken callback provided?}

    HasCallback -->|No| NoTransport["activeTokenTransport = 'none'<br/>callbackPtr = 0<br/>No token streaming"]

    HasCallback -->|Yes| CheckPref{config.debugTokenTransport?}

    CheckPref -->|"'runtime-events'"| ForceRE["Require CE_DrainRuntimeEvents support<br/>Throw if unavailable<br/>activeTokenTransport = 'runtime-events'<br/>callbackPtr = 0"]

    CheckPref -->|"'auto' (default)"| ProbeRE["bridge.supportsRuntimeEventDrain()<br/>(probe CE_DrainRuntimeEvents with null args)"]

    ProbeRE -->|Available| UseRE["activeTokenTransport = 'runtime-events'<br/>callbackPtr = 0<br/>Tokens collected via drainRuntimeEvents()"]

    ProbeRE -->|Not available| UseCB["activeTokenTransport = 'callbacks'<br/>bridge.registerTokenCallback(onToken)<br/>callbackPtr registered in WASM function table<br/>Native calls into JS per token via Emscripten addFunction()"]
```

---

## 8. RequestTracker — Promise Lifecycle Management

`RequestTracker<TResult>` is a generic bookkeeping class shared by both `MainThreadEngineRuntime` (tracking `GenerateResponse`) and `WorkerEngineRuntime` (tracking `WorkerRunQueuedRequestResult`). It manages deferred promises, abort signal listeners, and memory cleanup for request completions.

```mermaid
stateDiagram-v2
    [*] --> Tracked : tracker.track(requestId)<br/>Creates deferred Promise<br/>Adds to activeRuns set

    Tracked --> Settled_Resolved : tracker.resolve(id, result)<br/>settled=true, settlementState='resolved'

    Tracked --> Settled_Rejected : tracker.reject(id, error)<br/>settled=true, settlementState='rejected'

    Settled_Resolved --> Consumed : runQueuedRequest() awaits promise<br/>tracked.consumed = true<br/>tracked.waiterCount++

    Consumed --> Cleaned : tracker.cleanupIfConsumed(id)<br/>waiterCount=0 AND consumed=true<br/>Delete from completions map

    Settled_Rejected --> Cleaned : Same cleanup path

    Tracked --> Aborted : AbortSignal fires<br/>Attached via tracker.attachSignal()<br/>Calls cancelQueuedRequest()

    Aborted --> Settled_Rejected : Cancel propagates through bridge

    Settled_Resolved --> Finalized : tracker.finalize(id)<br/>releaseSignal()<br/>activeRuns.delete(id)

    Settled_Rejected --> Finalized : Same path

    Finalized --> Cleaned : cleanupIfConsumed() or deleteCompletion=true
    Cleaned --> [*]
```

---

## 9. Scheduler & Batching Pipeline

The native `SlotScheduler` multiplexes multiple concurrent requests into a single `llama_batch` per tick. The scheduler policy (configurable at init time via `schedulerPolicy: 'latency-first' | 'balanced' | 'throughput-first'`) controls the relative weight given to prefill vs decode phases in each tick budget.

```mermaid
graph TD
    subgraph RequestQueue ["Request Queue"]
        Pending["Pending Requests"]
    end

    subgraph SlotScheduler ["SlotScheduler (slot_scheduler.cpp)"]
        Assign["Assign Requests to Slots"]
        BuildBudget["BuildTickBudget() <br/> Decides Decode/Prefill ratio<br/>Based on SchedulerPolicyMode"]
        SelectPrefill["SelectPrefillReadySlots()"]
        SelectDecode["SelectDecodeReadySlots()"]
    end

    subgraph BatchPlanner ["BatchPlanner (batch_planner.cpp)"]
        RoutePolicy["BuildPolicyBatch()"]
        Chunk["Chunk Contexts (prefillChunkSize / adaptivePrefillChunking)"]
        AddTokens["LlamaBatchBuilder::Add()"]
    end

    Pending --> Assign
    Assign --> BuildBudget
    BuildBudget --> SelectPrefill
    BuildBudget --> SelectDecode

    SelectPrefill --> RoutePolicy
    SelectDecode --> RoutePolicy

    RoutePolicy --> Chunk
    Chunk --> AddTokens
    AddTokens --> |Outputs| BatchOut["llama_batch (To llama_decode)"]
```

---

## 10. Context Session and Cache Flow

The `SessionStore` manages KV cache reuse across turns. When a slot needs context space (e.g. rotation due to max context length), the runtime attempts to restore previously saved KV state via `llama_kv_cache_seq_cp`, avoiding full re-prefill.

```mermaid
graph LR
    Slot["SlotState"] --> |Needs Context Space| IR["InferenceRuntime::EnsureContextSpace()"]
    
    subgraph SessionStore ["Session Store"]
        CheckCache["Check if cached session exists"]
        Restore["Restore Session (KV Shift/Copy)"]
        Evict["Evict LRU Sessions<br/>(maxCachedSessions limit)"]
    end

    IR --> CheckCache
    CheckCache --> |Mismatch| Evict
    CheckCache --> |Partial Match| Restore
    
    Restore --> LlamaKV["llama_kv_cache_seq_rm / seq_cp"]
```

---

## 11. Prefix Caching Architecture

To eliminate redundant prompt computations (such as system prompts or repeated context chunks), the runtime employs a dedicated `PrefixStateCache`. It uses an Exact Hash Bucketing algorithm optimized for `O(1)` memory lookups while remaining token-accurate. The interval granularity is configurable via `prefixCacheIntervalTokens`; the maximum number of stored entries is controlled by `maxPrefixCacheEntries`.

### Exact Lookup and Restore Flow
When a new sequence is requested, the scheduler attempts to locate an exact historical token sequence. Fast retrieval is achieved through hashed candidate lengths combined with a strict full-token equality verification.

```mermaid
flowchart TD
    Start([Incoming Prompt Tokens]) --> Candidate[Build candidate prefix lengths<br/>Longest to Shortest]
    
    Candidate --> Loop{For each candidate length}
    
    Loop --> |Length L| Hash[Compute rolling hash over exact token IDs]
    Hash --> Key["Form Lookup Key:<br/>[Model Fingerprint + Prefix Count + Hash]"]
    
    Key --> Bucket[(Prefix Cache Buckets)]
    
    Bucket --> Found{Bucket Found?}
    Found --> |No| Loop
    
    Found --> |Yes| Verify[Verify Exact Token Equality]
    Verify --> Match{Exact Match?}
    
    Match --> |No| Loop
    
    Match --> |Yes| Rank["Rank Matches:<br/>1. Same Context Key<br/>2. Highest Retention Priority<br/>3. Most Recently Used (LRU)"]
    Rank --> Restore([Restore Best Exact Match])
```

### Prefix Cache Store Policy
When a sequence crosses a cacheable interval boundary (dictated by `PrefixCachePolicy` and `prefixCacheIntervalTokens`), the runtime stores the sequence to accelerate future permutations. `retainedPrefixTokens` controls how many leading context tokens are always preserved during eviction.

```mermaid
flowchart TD
    Start([Sequence reaches cache boundary]) --> Read[Read sequence state bytes via llama_state_seq_*]
    
    Read --> Assemble["Assemble Storage Entry:<br/>- Model Fingerprint<br/>- Context Key<br/>- Prefix Count<br/>- Token Hash<br/>- Exact Token IDs<br/>- Serialized KV Bytes<br/>- Retention Meta"]
    
    Assemble --> CheckExisting{Does identical entry exist?}
    
    CheckExisting --> |Yes| Replace[Replace existing entry]
    CheckExisting --> |No| Bucket[Store and Rebuild Lookup Buckets]
    
    Replace --> CheckLimit
    Bucket --> CheckLimit{maxPrefixCacheEntries Exceeded?}
    
    CheckLimit --> |Yes| Evict[Evict Oldest/Lowest Priority Entry<br/>Based on LRU + retainedPrefixTokens]
    CheckLimit --> |No| End([Entry Cached])
    
    Evict --> End
```

---

## 12. Observability Architecture

CogentEngine exposes three layers of observability, all opt-in:

| Layer | Type | Enabled by |
|---|---|---|
| **RuntimeObservability** | Per-request + aggregate metrics (timing, token counts, cache hits) | `enableRuntimeObservability: true` in `InferenceInitConfig` |
| **BackendObservability** | Raw llama.cpp backend profiling data (JSON) | `enableBackendProfiling: true` in `InferenceInitConfig` |
| **TransportObservability** | Cross-thread token delivery metrics (flush counts, coalescing stats, active transport mode) | Always collected, not gated |

```mermaid
graph LR
    subgraph Native ["Native Side"]
        RO["RuntimeObservabilityMetrics<br/>(per request + aggregate)<br/>Populated in each batch tick"]
        BO["Backend Profiling JSON<br/>CE_GetBackendObservabilityJson()"]
    end

    subgraph Bridge ["WasmBridge"]
        ReadObs["readRuntimeObservability()<br/>CE_GetRuntimeObservability(metricsPtr)<br/>Reads 9 doubles + 13 ints from HEAP"]
        ReadReqObs["readCompletedRequestRuntimeObservability(id)<br/>CE_GetCompletedRequestRuntimeObservability(id, ptr)"]
        ReadBackend["getBackendObservabilityJson()<br/>CE_GetBackendObservabilityJson() → pointer<br/>UTF8ToString() → JSON string<br/>CE_FreeString()"]
    end

    subgraph Output ["DetailedRuntimeObservabilityMetrics (derived)"]
        Fields["totalMs, promptEvalMs, decodeEvalMs, sampleMs<br/>queueDelayMs, ttftMs, meanItlMs, tailItlMs, e2elMs<br/>inputTokenCount, outputTokenCount, decodeEvalCount<br/>lcpReuseTokens, prefixCacheHitCount, prefixCacheStoreCount<br/>+ derived: tokensPerSecond"]
    end

    subgraph Transport ["TransportObservability"]
        TFields["executionMode, workerBacked<br/>activeTokenTransport, tokenCallbackRegistrationCount<br/>nativeCallbackTokenCount, runtimeEventDrainCount<br/>runtimeEventTokenCount, runtimeEventTextBytes<br/>flushCount, coalescedTokenCount (worker only)"]
    end

    RO --> ReadObs --> Output
    RO --> ReadReqObs --> Output
    BO --> ReadBackend

    Note1["Included in GenerateResponse.runtimeObservability<br/>after each completed request"]
    Output --> Note1

    Note2["Returned by getTransportObservability()<br/>on both runtime implementations"]
    Transport --> Note2
```
