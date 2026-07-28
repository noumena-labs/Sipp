import Foundation
import SippCoreBindings

func callBridge<Value>(_ operation: () throws -> Value) throws -> Value {
    do {
        return try operation()
    } catch let error as FfiError {
        throw SippError(error)
    }
}

func callBridge<Value>(_ operation: () async throws -> Value) async throws -> Value {
    do {
        return try await operation()
    } catch let error as FfiError {
        throw SippError(error)
    }
}

func nativeStoragePath(_ url: URL) throws -> String {
    guard url.isFileURL else {
        throw SippError.invalidURL(url)
    }
    return url.path
}

func nativeModelSource(_ url: URL) throws -> String {
    if url.isFileURL {
        return url.path
    }
    switch url.scheme?.lowercased() {
    case "http", "https":
        return url.absoluteString
    default:
        throw SippError.invalidURL(url)
    }
}

func defaultStorageRoot(fileManager: FileManager = .default) throws -> URL {
    do {
        return try fileManager.url(
            for: .applicationSupportDirectory,
            in: .userDomainMask,
            appropriateFor: nil,
            create: true
        ).appendingPathComponent("Sipp", isDirectory: true)
    } catch {
        throw SippError.storageLocation(message: error.localizedDescription)
    }
}
