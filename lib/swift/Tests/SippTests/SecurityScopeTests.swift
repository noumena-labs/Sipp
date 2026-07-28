import Foundation
import XCTest
@testable import Sipp

@MainActor
final class SecurityScopeTests: XCTestCase {
    func testBookmarkRegistryRoundTripsByModelID() throws {
        let root = try temporaryRoot()
        let file = SecurityBookmarkFile(storageRoot: root)
        let registry = StoredSecurityBookmarks(
            version: securityBookmarkRegistryVersion,
            models: [
                "model-id": StoredModelBookmarks(bookmarks: [Data([1, 2, 3])]),
            ]
        )

        try file.write(registry)

        XCTAssertEqual(try file.load(), registry)
    }

    func testBookmarkRegistryRejectsUnknownVersion() throws {
        let root = try temporaryRoot()
        let file = SecurityBookmarkFile(storageRoot: root)
        let registry = StoredSecurityBookmarks(version: 2, models: [:])
        let encoder = PropertyListEncoder()
        encoder.outputFormat = .binary
        try encoder.encode(registry).write(to: file.url, options: .atomic)

        XCTAssertThrowsError(try file.load()) { error in
            XCTAssertEqual(error as? SippError, .bookmarkVersion(found: 2, expected: 1))
        }
    }

    func testRemoteSourcesDoNotCreateSecurityBookmarks() throws {
        let source = try XCTUnwrap(URL(string: "https://models.example/model.gguf"))

        let acquisition = try AcquiredSecurityScopes.acquire(
            for: [source],
            isSandboxed: true
        )

        XCTAssertTrue(acquisition.bookmarks.isEmpty)
        XCTAssertTrue(acquisition.active.isEmpty)
    }

    func testOrdinaryFileURLsDoNotCreateSecurityBookmarks() throws {
        let root = try temporaryRoot()
        let source = root.appendingPathComponent("model.gguf")
        try Data([1, 2, 3]).write(to: source)

        let acquisition = try AcquiredSecurityScopes.acquire(
            for: [source],
            isSandboxed: false
        )

        XCTAssertTrue(acquisition.bookmarks.isEmpty)
        XCTAssertTrue(acquisition.active.isEmpty)
    }

    func testContainerFileURLsDoNotCreateSecurityBookmarks() throws {
        let root = try temporaryRoot()
        let source = root.appendingPathComponent("Documents/model.gguf")
        try FileManager.default.createDirectory(
            at: source.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try Data([1, 2, 3]).write(to: source)

        let acquisition = try AcquiredSecurityScopes.acquire(
            for: [source],
            localSourceRoot: root,
            isSandboxed: true
        )

        XCTAssertTrue(acquisition.bookmarks.isEmpty)
        XCTAssertTrue(acquisition.active.isEmpty)
    }

    func testRestorationDoesNotResolveBookmarksDuringClientInitialization() throws {
        let root = try temporaryRoot()
        let file = SecurityBookmarkFile(storageRoot: root)
        let registry = StoredSecurityBookmarks(
            version: securityBookmarkRegistryVersion,
            models: [
                "model-id": StoredModelBookmarks(bookmarks: [Data([1, 2, 3])]),
            ]
        )
        try file.write(registry)

        let restoration = try SecurityScopeRestoration.load(storageRoot: root)

        XCTAssertEqual(restoration.registry, registry)
    }

    func testManagerCommitsAndRemovesModelBookmarks() async throws {
        let root = try temporaryRoot()
        let file = SecurityBookmarkFile(storageRoot: root)
        let restoration = SecurityScopeRestoration(
            file: file,
            registry: .empty
        )
        let manager = SecurityScopeManager(restoration: restoration)
        let acquisition = AcquiredSecurityScopes(
            bookmarks: [Data([4, 5, 6])],
            active: []
        )

        try await manager.register(acquisition, for: "model-id")
        XCTAssertEqual(
            try file.load().models["model-id"],
            StoredModelBookmarks(bookmarks: [Data([4, 5, 6])])
        )

        try await manager.remove(modelID: "model-id")
        XCTAssertTrue(try file.load().models.isEmpty)
    }

    private func temporaryRoot() throws -> URL {
        let fileManager = FileManager.default
        let root = fileManager.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try fileManager.createDirectory(at: root, withIntermediateDirectories: false)
        addTeardownBlock {
            try fileManager.removeItem(at: root)
        }
        return root
    }
}
