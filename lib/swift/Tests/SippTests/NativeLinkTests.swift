import Foundation
import XCTest
import Sipp

final class NativeLinkTests: XCTestCase {
    func testPublicClientInitializesAgainstNativeCore() throws {
        let fileManager = FileManager.default
        let root = fileManager.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try fileManager.createDirectory(at: root, withIntermediateDirectories: false)
        addTeardownBlock {
            try fileManager.removeItem(at: root)
        }

        _ = try SippClient(storageRoot: root)
    }
}
