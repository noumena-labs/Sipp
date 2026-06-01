# CogentLM Agent Context Hub

This directory contains specialized system blueprints, build workflows, and Standard Operating Procedures (SOPs) to help you complete tasks in this polyglot monorepo efficiently.

## Directory Structure

- **/system**: Architecture, structural boundaries, and API interfaces.
  - Read [system/architecture.md](system/architecture.md) to understand the crate organization.
  - Read [system/native-interfaces.md](system/native-interfaces.md) to understand the Rust/C++/Wasm/Node/Python bridge layers.
- **/build**: Building, dependencies, and local development.
  - Read [build/instructions.md](build/instructions.md) before compiling or running command-line tooling.
- **/SOPs**: Step-by-step procedures for specific actions.
  - Check here for dynamic learnings and debugging guides (e.g., handling FFI memory leaks, packaging releases).
- **/task**: Local workspace task state.
