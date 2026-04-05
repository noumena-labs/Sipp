# CogentEngine Codebase Visualization

This document presents a comprehensive set of visualizations for the CogentEngine architecture using Mermaid diagrams. The visualizations are structured from the high-level system boundary down to the nuanced function calls of the native inference scheduler. 

## 1. High-Level System Architecture

This diagram illustrates how the frontend components (the TypeScript SDK and Web Worker) interface with the WebAssembly compilation of the C++ Native Runtime. 

```mermaid
graph TD
    subgraph Frontend ["TypeScript SDK (src/)"]
        CE["CogentEngine Facade"]
        subgraph RuntimeStrategy ["Execution Backends"]
            Worker["WorkerEngineRuntime"]
            Main["MainThreadEngineRuntime"]
        end
        Submit["submitPrompt()"]
        Poll["runQueuedRequest()"]
        Cancel["cancelQueuedRequest()"]
    end

    subgraph WASM ["WASM Export Layer (native/api/)"]
        WasmAPI["CE_EnqueuePrompt"]
        WasmRun["CE_RunQueuedRequestJson"]
        WasmCancel["CE_CancelQueuedRequest"]
        BridgeAPI["CE_EnqueuePromptQuery"]
        BridgeRun["CE_RunQueuedRequestJsonString"]
        BridgeCancel["CE_CancelQueuedPromptQuery"]
    end

    subgraph NativeCore ["Inference Runtime (native/runtime/)"]
        IR["InferenceRuntime<br/>(Core Orchestrator)"]
        RQ["RequestQueue"]
        Scheduler["SlotScheduler & BatchPlanner"]
        Session["SessionStore & PrefixStateCache"]
        Llama["llama.cpp Backend"]
    end

    CE --> Worker
    CE --> Main
    Worker --> Submit
    Worker --> Poll
    Worker --> Cancel

    Submit -- "Pointer Payload" --> WasmAPI
    Poll -- "Request ID" --> WasmRun
    Cancel -- "Request ID" --> WasmCancel

    WasmAPI --> BridgeAPI
    WasmRun --> BridgeRun
    WasmCancel --> BridgeCancel

    BridgeAPI --> |1. EnqueueRequest| IR
    BridgeRun --> |2. RunUntilRequestCompletes| IR
    BridgeCancel --> |3. CancelRequest| IR

    IR --> RQ
    IR --> Scheduler
    IR --> Session
    Scheduler --> Llama
```

---

## 2. Request Lifecycle & Execution Call Flow

When a prompt is enqueued, it doesn't execute immediately. It enters the `RequestQueue` and is polled by the TypeScript layer (typically via Asyncify or Web Workers polling the `generate` function). The engine tick then schedules, plans, and executes the batch.

```mermaid
sequenceDiagram
    participant TS as TypeScript (CogentEngine)
    participant Wasm as engine_bridge.cpp
    participant IR as InferenceRuntime
    participant Queue as RequestQueue
    participant Sched as SlotScheduler
    participant Batch as BatchPlanner
    participant Llama as llama.cpp

    TS->>Wasm: CE_EnqueuePromptQuery(prompt)
    Wasm->>IR: EnqueueRequest()
    IR->>Queue: Push(GenerateRequest)
    Queue-->>IR: request_id
    IR-->>Wasm: request_id
    Wasm-->>TS: request_id

    par Async Cancellation (Optional)
        TS->>Wasm: CE_CancelQueuedRequest(id)
        Wasm->>Queue: Cancel(id) / set cancel_requested
    end

    loop Polling Loop until Complete or Cancelled
        TS->>Wasm: CE_RunQueuedRequestJson(id)
        Wasm->>IR: RunUntilRequestCompletes(id)

        IR->>Queue: TryPopCompletedResponse(id)
        alt Response not ready
            IR->>IR: RunPolicyBatchTickLocked()
            
            IR->>Sched: Tick() / Select Slots
            Sched->>Sched: Check request->cancel_requested
            Sched-->>IR: Ready Slots (Decode/Prefill)
            
            IR->>Batch: BuildPolicyBatch()
            Batch-->>IR: SharedBatchPlan
            
            IR->>Llama: llama_decode(batch)
            
            IR->>Llama: llama_sampler_sample()
            Llama-->>IR: New Token
            
            IR->>Queue: Update Request status <br/> & attach RuntimeObservabilityMetrics
        end
        IR-->>Wasm: GenerateResponse (JSON)
        Wasm-->>TS: Parsed Object
    end
```

---

## 3. Scheduler & Batching Pipeline

CogentEngine supports scheduling with an ITL/Throughput policy mode (`SchedulerPolicyMode::LatencyFirst`, `Balanced`, etc.). This diagram explains how multiple slot sequences are multiplexed into a single `llama_batch` per tick.

```mermaid
graph TD
    subgraph RequestQueue ["Request Queue"]
        Pending["Pending Requests"]
    end

    subgraph SlotScheduler ["SlotScheduler (slot_scheduler.cpp)"]
        Assign["Assign Requests to Slots"]
        BuildBudget["BuildTickBudget() <br/> Decides Decode/Prefill ratio"]
        SelectPrefill["SelectPrefillReadySlots()"]
        SelectDecode["SelectDecodeReadySlots()"]
    end

    subgraph BatchPlanner ["BatchPlanner (batch_planner.cpp)"]
        RoutePolicy["BuildPolicyBatch()"]
        Chunk["Chunk Contexts (max_chunk_size)"]
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

## 4. Context session and Cache flow

The `SessionStore` is responsible for Shift-KV (managing context windows without invalidating everything). 

```mermaid
graph LR
    Slot["SlotState"] --> |Needs Context Space| IR["InferenceRuntime::EnsureContextSpace()"]
    
    subgraph SessionStore ["Session Store"]
        CheckCache["Check if cached session exists"]
        Restore["Restore Session (KV Shift/Copy)"]
        Evict["Evict LRU Sessions"]
    end

    IR --> CheckCache
    CheckCache --> |Mismatch| Evict
    CheckCache --> |Partial Match| Restore
    
    Restore --> LlamaKV["llama_kv_cache_seq_rm / seq_cp"]
```

---

## 5. Prefix Caching Architecture

To eliminate redundant prompt computations (such as system prompts or repeated context chunks), the runtime employs a dedicated `PrefixStateCache`. It uses an Exact Hash Bucketing algorithm optimized for `O(1)` memory lookups while remaining token-accurate.

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
When a sequence crosses a cacheable interval boundary (dictated by `PrefixCachePolicy`), the runtime stores the sequence to accelerate future permutations.

```mermaid
flowchart TD
    Start([Sequence reaches cache boundary]) --> Read[Read sequence state bytes via llama_state_seq_*]
    
    Read --> Assemble["Assemble Storage Entry:<br/>- Model Fingerprint<br/>- Context Key<br/>- Prefix Count<br/>- Token Hash<br/>- Exact Token IDs<br/>- Serialized Bytes<br/>- Retention Meta"]
    
    Assemble --> CheckExisting{Does identical entry exist?}
    
    CheckExisting --> |Yes| Replace[Replace existing entry]
    CheckExisting --> |No| Bucket[Store and Rebuild Lookup Buckets]
    
    Replace --> CheckLimit
    Bucket --> CheckLimit{Memory/Entry Limit Exceeded?}
    
    CheckLimit --> |Yes| Evict[Evict Oldest/Lowest Priority Entry<br/>Based on LRU]
    CheckLimit --> |No| End([Entry Cached])
    
    Evict --> End
```
