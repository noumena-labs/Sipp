# Sipp for Swift

This directory owns the handwritten public Swift package. The generated
UniFFI declarations live in the non-product `SippCoreBindings` target and are
an implementation detail.

The internal bridge covers model/local-endpoint lifecycle plus query, chat,
embedding, token, response, statistics, and cancellation contracts. Remote
model acquisition uses a dedicated native executor; local token and response
polling stays on the calling async task. The public target adds Swift semantics:
Foundation URLs, Application Support storage, Swift values and errors,
`AsyncSequence` token batches, cached terminal responses, task cancellation,
and persistent security-scoped model-file access.
Build and test the staged package with:

```bash
cargo xtask build swift
# Equivalent cataloged interface gate:
cargo xtask test unit suite swift-package
```

The command writes a complete local package to
`.build/artifacts/swift/package`. The artifact directory also contains the
external consumer fixture, `SippCore.abi.txt`, the distributable
`SippCore.xcframework.zip`, and its `.sha256` checksum. The source directory
intentionally does not commit generated bindings or native build artifacts.
The same gate builds `.build/artifacts/swift/examples/SippCLI`, assembles
`.build/artifacts/swift/examples/SippSandbox.app`, and ad hoc signs the app
with the committed App Sandbox entitlement.

The XCFramework contains Metal slices for macOS arm64 and iOS device arm64,
plus CPU slices for macOS x86_64 and both iOS Simulator architectures. There is
no runtime backend selection or fallback. The build fails unless both generated
binding passes are byte-identical, each native slice and universal archive
contain exactly their declared architectures, the generated and exported Sipp
FFI symbol sets match exactly, the external consumer links only Apple system and
Swift runtime libraries, and SwiftPM's archive checksum matches an independent
SHA-256 calculation. `cargo xtask clean` removes all of these generated outputs
through the shared artifact root.

The public API is:

```swift
import Sipp

let client = try SippClient()
let model = try await client.models.add([modelURL])
try await client.add("chat", model: model)

let run = client.chat(
    messages: [ChatMessage(role: .user, content: "Explain on-device inference")],
    endpoint: "chat"
)

for await batch in run.tokens {
    print(batch.text, terminator: "")
}

let response = try await run.response
```

Sandbox-owned model files are registered as zero-copy paths relative to the
application container, so a container relocation does not invalidate them.
External files use versioned security-scoped bookmarks stored beside the native
model registry; Sipp resolves a model's bookmarks only when that model is
activated and then retains access until successful model removal. The signed
sandbox relaunch test still requires macOS and must pass before release. The
complete gates are tracked in the [Swift package plan](../../docs/en/packages/swift.md).

Runnable command-line and SwiftUI examples live under
[`examples/swift`](../../examples/swift/README.md). Remote SwiftPM installation
is intentionally undocumented until a tagged release contains a manifest with
the checksum of the actual published XCFramework archive.
