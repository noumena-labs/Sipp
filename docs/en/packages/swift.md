# Swift Package

The Swift package exposes Sipp to native macOS and iOS applications. It supports local
GGUF model registration, named local endpoints, query, chat, embedding,
streaming tokens, cancellation, typed errors, and sandbox-friendly file access.

Swift support is still in development and is not published as a remote SwiftPM
package yet. Today, build it from a source checkout on macOS.

See the [Library API Overview](../api) for the shared `add`, `query`, `chat`,
and `embed` contracts.

## Install From Source

Swift builds require macOS, Xcode, CMake, Ninja, and the Apple Rust targets:

```bash
cargo xtask toolchain install rust-apple
cargo xtask doctor --target swift
cargo xtask build swift
```

The build has no backend selector or runtime fallback. Each XCFramework slice
has one backend chosen for its platform and architecture:

| Platform | Architectures | Backend |
| --- | --- | --- |
| macOS | arm64 | Metal |
| macOS | x86_64 | CPU |
| iOS device | arm64 | Metal |
| iOS Simulator | arm64, x86_64 | CPU |

SwiftPM selects the matching slice when it links the application. Metal-capable
Apple silicon devices therefore use Metal, while Intel macOS and simulator
builds use CPU code. A Metal initialization failure is reported as an error; it
does not silently rerun the request on CPU.

The build writes a complete local Swift package to:

```text
.build/artifacts/swift/package
```

Point your application package dependency at that directory while developing
against an untagged checkout. A remote SwiftPM URL will be documented after the
versioned `SippCore.xcframework` archive is published.

## Use It For

- Native macOS and iOS applications that need local GGUF inference.
- Sandboxed apps that receive model files from a system file picker.
- Query, chat, embedding, token streaming, and cancellation.
- Deterministic Metal or CPU execution selected by the linked platform slice.

Gateway, direct provider, Objective-C, and other Apple-platform APIs are
outside the initial Swift package scope.

## Local Chat

```swift
import Sipp

let client = try SippClient()
let model = try await client.models.add([modelURL])
try await client.add("local", model: model)

let run = client.chat(
    messages: [
        ChatMessage(role: .system, content: "Answer concisely."),
        ChatMessage(role: .user, content: "Explain on-device inference."),
    ],
    endpoint: "local",
    options: TextOptions(maxTokens: 64)
)

for await batch in run.tokens {
    print(batch.text, terminator: "")
}

let response = try await run.response
print(response.text)
```

`TextRun.tokens` is a consumptive `AsyncSequence`. `TextRun.response` is cached,
so multiple awaits observe the same terminal result.

## Query And Embedding

Use `query` for raw prompts:

```swift
let run = client.query(
    "Explain Sipp in one sentence.",
    endpoint: "local",
    options: TextOptions(maxTokens: 64)
)

print(try await run.response.text)
```

Use `embed` with a model that supports embeddings:

```swift
let run = client.embed(
    "Text to embed",
    endpoint: "local",
    local: LocalEmbedOptions(normalize: true)
)

let embedding = try await run.response
print(embedding.values.prefix(8))
```

Call `cancel()` on any active run to request native cancellation:

```swift
run.cancel()
```

Cancelling a Swift task that is awaiting tokens or a response also requests
cancellation.

## Models And Endpoints

Create one client per application storage root:

```swift
let client = try SippClient()
```

When `storageRoot` is omitted, Sipp uses the user's Application Support
directory. To use a custom location, pass a file URL:

```swift
let client = try SippClient(storageRoot: appStorageURL)
```

Register local file URLs or HTTP(S) model sources:

```swift
let model = try await client.models.add([modelURL])
let models = try await client.models.list()
try await client.models.remove(model)
```

Endpoints are caller-owned strings:

```swift
try await client.add("local", model: model)
try await client.remove("local")
```

Removing a model fails while an endpoint still uses it. Remove the endpoint
first, then remove the model.

## App Sandbox

For sandboxed macOS apps, pass the `URL` returned by a system file picker
directly to `client.models.add`. Do not convert it to a path before Sipp
receives it.

Files already inside the application container remain zero-copy and are stored
relative to that container, so their registration survives container
relocation. Files outside the container use security-scoped bookmarks. Sipp
resolves those bookmarks when an endpoint first activates the registered model,
not while the client is being initialized.

The sandboxed app needs these entitlements for local model files:

```xml
<key>com.apple.security.app-sandbox</key>
<true/>
<key>com.apple.security.files.bookmarks.app-scope</key>
<true/>
<key>com.apple.security.files.user-selected.read-only</key>
<true/>
```

Add `com.apple.security.network.client` only when registering HTTP(S) model
sources.

## Examples

`cargo xtask build swift` builds a command-line example and a sandboxed SwiftUI
example for macOS, plus a SwiftUI iOS example:

```bash
.build/artifacts/swift/examples/SippCLI chat /path/to/model.gguf \
  "Explain on-device inference"
open .build/artifacts/swift/examples/SippSandbox.app

cargo xtask run examples ios --simulator SIMULATOR-UDID \
  --model /path/to/model.gguf
```

The CLI supports `query`, `chat`, `embed`, and `cancel`. The SwiftUI example
uses a system importer, streams chat tokens, runs embeddings, shows typed
errors, and supports cancellation. See
[`examples/swift`](../../../examples/swift/README.md).

## Troubleshooting

- Run `cargo xtask doctor --target swift` before building.
- If an Apple Rust target is missing, run
  `cargo xtask toolchain install rust-apple`.
- If `SippCore.xcframework` is missing, depend on the staged package under
  `.build/artifacts/swift/package`, not `lib/swift`.
- If an external sandboxed model file is unavailable when its endpoint is
  activated, keep the same signed app identity and inspect the typed bookmark
  error. App-container model files do not depend on bookmarks.
- If embedding fails, use a GGUF/runtime with embedding support.
- If Metal is unavailable, the package uses the core CPU behavior.

## Related Docs

- [Examples And Demos](../examples-demos.md)
- [Runtime Options](../reference/runtime-options.md)
- [Source Builds](../maintainers/source-builds.md)
