# Sipp for Swift

This repository is the generated Swift Package Manager distribution channel
for [Sipp](https://github.com/noumena-labs/Sipp). Do not edit the package
sources here: each tagged snapshot is produced and tested by Sipp's release
workflow from one recorded source commit.

## Add The Package

In Xcode, select **File → Add Package Dependencies**, then enter:

```text
https://github.com/noumena-labs/Sipp-Swift.git
```

Select a tagged release and add the `Sipp` product to the macOS or iOS target.
For a `Package.swift` dependency, use the same URL and the desired tagged
version.

```swift
import Sipp

let client = try SippClient()
```

The package downloads a versioned `SippCore.xcframework.zip` release asset and
verifies the checksum committed in `Package.swift`. It does not require Rust,
CMake, or a Sipp source checkout.

Development prereleases are intended for integration testing and should be
pinned to an exact version. Stable releases follow Sipp's shared package
version.

## Source And Issues

`SIPP_SOURCE_REVISION` records the Sipp repository commit, package version, and
binary checksum used for the current snapshot. Report issues and contribute
changes in the [Sipp repository](https://github.com/noumena-labs/Sipp).
