import Foundation

let securityBookmarkRegistryVersion: UInt32 = 1
let securityBookmarkRegistryFileName = "security-scoped-bookmarks.plist"

#if os(macOS)
private let bookmarkCreationOptions: URL.BookmarkCreationOptions = [
    .withSecurityScope,
    .securityScopeAllowOnlyReadAccess,
]
private let bookmarkResolutionOptions: URL.BookmarkResolutionOptions = [
    .withSecurityScope,
    .withoutUI,
]
let platformIsSandboxed =
    ProcessInfo.processInfo.environment["APP_SANDBOX_CONTAINER_ID"] != nil
#else
private let bookmarkCreationOptions: URL.BookmarkCreationOptions = []
private let bookmarkResolutionOptions: URL.BookmarkResolutionOptions = []
let platformIsSandboxed = true
#endif

func sandboxLocalSourceRoot(
    isSandboxed: Bool = platformIsSandboxed
) -> URL? {
    guard isSandboxed else {
        return nil
    }
    return URL(fileURLWithPath: NSHomeDirectory(), isDirectory: true)
}

func isFileURL(_ candidate: URL, containedBy root: URL) -> Bool {
    guard candidate.isFileURL, root.isFileURL else {
        return false
    }
    let candidateComponents = candidate.standardizedFileURL
        .resolvingSymlinksInPath()
        .pathComponents
    let rootComponents = root.standardizedFileURL
        .resolvingSymlinksInPath()
        .pathComponents
    guard candidateComponents.count >= rootComponents.count else {
        return false
    }
    return zip(rootComponents, candidateComponents).allSatisfy { root, candidate in
        root == candidate
    }
}

struct StoredModelBookmarks: Codable, Equatable, Sendable {
    let bookmarks: [Data]
}

struct StoredSecurityBookmarks: Codable, Equatable, Sendable {
    let version: UInt32
    var models: [String: StoredModelBookmarks]

    static var empty: Self {
        Self(version: securityBookmarkRegistryVersion, models: [:])
    }
}

struct SecurityBookmarkFile: Sendable {
    let url: URL

    init(storageRoot: URL) {
        url = storageRoot.appendingPathComponent(
            securityBookmarkRegistryFileName,
            isDirectory: false
        )
    }

    func load() throws -> StoredSecurityBookmarks {
        guard FileManager.default.fileExists(atPath: url.path) else {
            return .empty
        }
        let registry: StoredSecurityBookmarks
        do {
            let data = try Data(contentsOf: url)
            registry = try PropertyListDecoder().decode(StoredSecurityBookmarks.self, from: data)
        } catch {
            throw SippError.bookmarkStore(
                message: "Failed to read security-scoped bookmarks: \(error.localizedDescription)"
            )
        }
        guard registry.version == securityBookmarkRegistryVersion else {
            throw SippError.bookmarkVersion(
                found: registry.version,
                expected: securityBookmarkRegistryVersion
            )
        }
        return registry
    }

    func write(_ registry: StoredSecurityBookmarks) throws {
        do {
            let encoder = PropertyListEncoder()
            encoder.outputFormat = .binary
            let data = try encoder.encode(registry)
            try data.write(to: url, options: .atomic)
        } catch {
            throw SippError.bookmarkStore(
                message: "Failed to persist security-scoped bookmarks: \(error.localizedDescription)"
            )
        }
    }
}

final class ActiveSecurityScope: Sendable {
    private let url: URL

    init?(_ url: URL) {
        guard url.startAccessingSecurityScopedResource() else {
            return nil
        }
        self.url = url
    }

    deinit {
        url.stopAccessingSecurityScopedResource()
    }
}

struct AcquiredSecurityScopes: Sendable {
    let bookmarks: [Data]
    let active: [ActiveSecurityScope]

    static func acquire(
        for sources: [URL],
        localSourceRoot: URL? = sandboxLocalSourceRoot(),
        isSandboxed: Bool = platformIsSandboxed
    ) throws -> Self {
        guard isSandboxed else {
            return Self(bookmarks: [], active: [])
        }

        var bookmarks = [Data]()
        var active = [ActiveSecurityScope]()
        for source in sources where source.isFileURL {
            if let localSourceRoot, isFileURL(source, containedBy: localSourceRoot) {
                continue
            }
            guard let scope = ActiveSecurityScope(source) else {
                throw SippError.securityScopedAccess(
                    url: source,
                    message: "The selected file's security scope could not be activated"
                )
            }
            active.append(scope)
            do {
                bookmarks.append(
                    try source.bookmarkData(
                        options: bookmarkCreationOptions,
                        includingResourceValuesForKeys: nil,
                        relativeTo: nil
                    )
                )
            } catch {
                throw SippError.securityScopedAccess(
                    url: source,
                    message: error.localizedDescription
                )
            }
        }
        return Self(bookmarks: bookmarks, active: active)
    }
}

struct SecurityScopeRestoration: Sendable {
    let file: SecurityBookmarkFile
    let registry: StoredSecurityBookmarks

    static func load(storageRoot: URL) throws -> Self {
        let file = SecurityBookmarkFile(storageRoot: storageRoot)
        return Self(file: file, registry: try file.load())
    }
}

actor SecurityScopeManager {
    private let file: SecurityBookmarkFile
    private var registry: StoredSecurityBookmarks
    private var activeByModel: [String: [ActiveSecurityScope]]

    init(restoration: SecurityScopeRestoration) {
        file = restoration.file
        registry = restoration.registry
        activeByModel = [:]
    }

    func activate(modelID: String) throws {
        guard activeByModel[modelID] == nil,
              let stored = registry.models[modelID]
        else {
            return
        }

        var resolvedBookmarks = [Data]()
        var active = [ActiveSecurityScope]()
        var renewedBookmarks = false
        for bookmark in stored.bookmarks {
            var isStale = false
            let url: URL
            do {
                url = try URL(
                    resolvingBookmarkData: bookmark,
                    options: bookmarkResolutionOptions,
                    relativeTo: nil,
                    bookmarkDataIsStale: &isStale
                )
            } catch {
                throw SippError.bookmarkStore(
                    message: "Failed to resolve bookmarks for model \(modelID): \(error.localizedDescription)"
                )
            }

            guard let scope = ActiveSecurityScope(url) else {
                throw SippError.securityScopedAccess(
                    url: url,
                    message: "The persisted security scope could not be activated"
                )
            }
            active.append(scope)
            if isStale {
                do {
                    resolvedBookmarks.append(
                        try url.bookmarkData(
                            options: bookmarkCreationOptions,
                            includingResourceValuesForKeys: nil,
                            relativeTo: nil
                        )
                    )
                } catch {
                    throw SippError.securityScopedAccess(
                        url: url,
                        message: error.localizedDescription
                    )
                }
                renewedBookmarks = true
            } else {
                resolvedBookmarks.append(bookmark)
            }
        }

        if renewedBookmarks {
            var next = registry
            next.models[modelID] = StoredModelBookmarks(bookmarks: resolvedBookmarks)
            try file.write(next)
            registry = next
        }
        activeByModel[modelID] = active
    }

    func register(_ acquisition: AcquiredSecurityScopes, for modelID: String) throws {
        var next = registry
        if acquisition.bookmarks.isEmpty {
            next.models.removeValue(forKey: modelID)
        } else {
            next.models[modelID] = StoredModelBookmarks(bookmarks: acquisition.bookmarks)
        }
        if next != registry {
            try file.write(next)
            registry = next
        }
        if acquisition.active.isEmpty {
            activeByModel.removeValue(forKey: modelID)
        } else {
            activeByModel[modelID] = acquisition.active
        }
    }

    func remove(modelID: String) throws {
        if registry.models[modelID] != nil {
            var next = registry
            next.models.removeValue(forKey: modelID)
            try file.write(next)
            registry = next
        }
        activeByModel.removeValue(forKey: modelID)
    }
}
