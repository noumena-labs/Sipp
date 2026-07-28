import Foundation
import Sipp

@main
struct SippConsumer {
    static func main() throws {
        let fileManager = FileManager.default
        let storageRoot = fileManager.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try fileManager.createDirectory(
            at: storageRoot,
            withIntermediateDirectories: false
        )

        do {
            _ = try SippClient(storageRoot: storageRoot)
        }

        try fileManager.removeItem(at: storageRoot)
    }
}
