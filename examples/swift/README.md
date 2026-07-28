# Swift Examples

The Swift examples consume only the public `Sipp` product from the staged
Swift package. Build the XCFramework, package, examples, and signed sandbox app
on macOS:

```bash
cargo xtask build swift
```

The package contains a Metal arm64 macOS slice and a CPU x86_64 macOS slice.
SwiftPM links the slice matching the host; the examples do not accept a backend
flag and the runtime does not fall back between backends.

## Command Line

The command-line example requires an operation, a local GGUF path, and input:

```bash
.build/artifacts/swift/examples/SippCLI query \
  /path/to/model.gguf "Explain on-device inference"

.build/artifacts/swift/examples/SippCLI chat \
  /path/to/model.gguf "Explain on-device inference"
```

Supported operations are `query`, `chat`, `embed`, and `cancel`. Query and chat
show terminal responses, chat streams token batches, embed prints a vector
preview, and cancel verifies the typed cancellation error.

Embedding requires a model that supports embedding generation.

## Sandboxed SwiftUI

Open the ad-hoc-signed app bundle:

```bash
open .build/artifacts/swift/examples/SippSandbox.app
```

Choose a GGUF with the system file importer, select query, chat, or embedding,
and run the request. The app retains the selected file's security-scoped access
through the public Sipp package and exposes an explicit cancellation button.

The committed entitlements grant the App Sandbox, app-scoped bookmark use, and
user-selected read-only file access. The example does not request broad
filesystem or network access.

## iOS Simulator

List the installed simulator identifiers:

```bash
xcrun simctl list devices available
```

Build, install, and launch the iOS example through xtask:

```bash
cargo xtask run examples ios \
  --simulator CBEBA5E8-F06A-4539-A842-A3561907311F \
  --model /path/to/model.gguf
```

The command builds the unified Swift package, boots the selected simulator,
installs `SippIOS.app`, copies the optional model into the application's
Documents directory, and launches it. In the app, choose the model from
**On My iPhone → Sipp iOS**. Pass `--no-build` to reuse the current app artifact.

The Simulator links the CPU slice. To test the Metal device slice, open
`ios/SippIOS.xcodeproj` in Xcode, select a connected iPhone or iPad, configure
the signing team, and run the `SippIOS` scheme.
