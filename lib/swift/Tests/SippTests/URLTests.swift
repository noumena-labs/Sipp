import Foundation
import XCTest
@testable import Sipp

final class URLTests: XCTestCase {
    func testFileURLsCrossTheBridgeAsNativePaths() throws {
        let url = URL(fileURLWithPath: "/tmp/model.gguf")

        XCTAssertEqual(try nativeStoragePath(url), url.path)
        XCTAssertEqual(try nativeModelSource(url), url.path)
    }

    func testHTTPModelSourcesRemainURLs() throws {
        let http = try XCTUnwrap(URL(string: "http://models.example/model.gguf"))
        let https = try XCTUnwrap(URL(string: "https://models.example/model.gguf"))

        XCTAssertEqual(try nativeModelSource(http), http.absoluteString)
        XCTAssertEqual(try nativeModelSource(https), https.absoluteString)
    }

    func testUnsupportedSourceSchemeIsTyped() throws {
        let url = try XCTUnwrap(URL(string: "ftp://models.example/model.gguf"))

        XCTAssertThrowsError(try nativeModelSource(url)) { error in
            XCTAssertEqual(error as? SippError, .invalidURL(url))
        }
    }

    func testNonFileStorageRootIsTyped() throws {
        let url = try XCTUnwrap(URL(string: "https://storage.example/sipp"))

        XCTAssertThrowsError(try nativeStoragePath(url)) { error in
            XCTAssertEqual(error as? SippError, .invalidURL(url))
        }
    }

    func testDefaultStorageRootUsesApplicationSupport() throws {
        let root = try defaultStorageRoot()

        XCTAssertTrue(root.isFileURL)
        XCTAssertEqual(root.lastPathComponent, "Sipp")
        XCTAssertEqual(root.deletingLastPathComponent().lastPathComponent, "Application Support")
    }
}
