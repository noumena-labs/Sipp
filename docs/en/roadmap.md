# Sipp Technical Roadmap

This document outlines the engineering milestones and long-term research initiatives for Sipp Core, Sipp Gateway, and Sipp Platform.

Sipp is built around three core ideas: maximizing privacy-preserving inference, low latency interactions, and high-performance compute across the edge and cloud.

The current core library has a powerful WebGPU backend for running models in-browser, as well as bare-metal GPU support for CUDA or Vulkan when running on device or server. We see the future of AI as hybrid, with edge-native AI processing and cloud-based AI processing working together seamlessly. 

## Research 1: Sipp Core: The Local Runtime Library

Sipp Core is built to be a high-performance power house for running inference locally, either on bare-metal GPU/NPU or via WebGPU for browser-based applications. It is built on a foundation of llama.cpp with a custom C++ and Rust runtime layer. 

### Key Initiatives

- **Edge-Native Local RAG & Memory Optimization:** Integrate an in-memory, zero-dependency vector database (compiled directly to WASM) into the client SDK. This enables developers to run fully local vector searches, embed conversational state, and execute document retrievals with zero external API dependencies or cloud database costs.
    
- **Full-Spectrum Client Support (Apps, Web, and Games):** Sipp currently supports browser through WebGPU and desktop through CUDA and Vulkan backends. Our next phase will be to expand backend support for hardware accelerated inference across web, desktop and mobile devices. This includes:
    
    - **Desktop & Mobile Wrappers:** Expand native compilation targets for Electron and Tauri apps, exposing direct access to **NVIDIA CUDA**, Metal, and Vulkan.
        
    - **Gaming Runtimes:** Lightweight SDK integration frameworks for **Unreal Engine** and **Unity** to support local, low-latency AI agents inside application loops.
        
- **Cross-Site & Cross-App Persistence Caching:** Standard browser sandboxing isolates cache stores to individual web origins. We seek to solve this by building a lightweight, local **background desktop daemon** built in Rust. This daemon serves as a centralized, secure model registry mirror. If a user visits an Electron app or a website utilizing a specific model, the local runtime fetches it instantly from the daemon's cache instead of re-downloading gigabytes of weights.
    
- **Client-Side Local Contextual Routing:** There may be times where running a query locally may not produce good enough results, in which re-routing to a cloud or provider model is needed. However, when this should happen or how a query could be split apart is unknown. We beleive a solution is in a hyper-lightweight, client-side small language model (sLLM) that makes those decisions dynamically, we see two applications:
    
    1. **PII/PPI Stripping and Masking:** A local model intercepts text inputs to detect and strip Protected Personal Information (PPI) or Personally Identifiable Information (PII), replacing sensitive entities with secure local tokenized hashes before any cloud handoff occurs.
        
    2. **Contextual Query Splitting:** The local engine analyzes incoming chats to determine what components can be handled instantly on the edge (e.g., immediate structural formatting, basic data verification) vs. what must be escalated to the cloud, dynamically stitching cloud completions back into the interface as they stream down.
        

## Research 2: The Gateway Server (The Orchestration & Interception Layer)

The open-source Gateway Server serves as an autonomous "API Fortress" that acts as a secure, high-performance middleware layer between client networks and cloud endpoints.

```
                    ┌──────────────────────────────┐
                    │ Client Submits Prompt to GW  │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │ Preemptive Middle-Layer Cache│
                    │   (Vector & KV Intercept)    │
                    └──────────────┬───────────────┘
                                   │
              ┌────────────────────┴────────────────────┐
              ▼ (Cache Hit / Guided Path)              ▼ (Cache Miss)
  ┌───────────────────────┐                 ┌───────────────────────┐
  │ Route to Endpoint X   │                 │ Route to Endpoint Y   │
  │ (Low-cost/Fast Stream)│                 │ (Deep Processing/MoE) │
  └───────────────────────┘                 └───────────────────────┘
```

### Key Initiatives

- **Gateway-Level Vector Memory & RAG Interception:** The gateway implements an internal, stateful vector index layer to handle server-side memory optimization. It caches semantic embeddings of historical document fragments and prior system queries. When a client submits a prompt, the gateway performs a preemptive vector evaluation to determine if a relevant context context match exists, entirely bypassing the need to repeatedly re-fetch or re-encode massive RAG documents from central cloud instances.
    
- **Preemptive Middle-Layer Caching:** In tandem with vector storage, the gateway features a stateful intermediate cache layer designed to intercept incoming requests _before_ they hit large upstream models. If a cached structural completion matches the incoming footprint, the gateway can reroute traffic conditionally (e.g., _"If cache footprint exists, route to fast Endpoint X; if not, route to reasoning Endpoint Y"_).
    
- **Persistent Admin Control Dashboard:** Expand on the gateway dashboard and admin UI to visualize active routes, manage cryptographic client application identities, view live input/output token allocation metrics, and manually map model fallback rules and more.
    
- **Token-Aware Traffic Shaping:** Implements token-bucket rate limiters directly inside the networking wrapper to monitor and throttle users based on their exact token throughput footprint, protecting downstream clusters from malicious execution loops or unexpected API bills.
    
