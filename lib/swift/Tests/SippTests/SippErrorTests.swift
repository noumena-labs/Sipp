import Foundation
import SippCoreBindings
import XCTest
@testable import Sipp

final class SippErrorTests: XCTestCase {
    func testEveryNativeErrorCategoryHasAStableSwiftCase() {
        XCTAssertEqual(
            SippError(.ModelLifecycle(code: "in-use", message: "busy", status: 409, retryAfterMs: 2)),
            .modelLifecycle(
                code: "in-use",
                message: "busy",
                status: 409,
                retryAfterMilliseconds: 2
            )
        )
        XCTAssertEqual(
            SippError(.InvalidArgument(message: "invalid")),
            .invalidArgument(message: "invalid")
        )
        XCTAssertEqual(
            SippError(.UnsupportedOperation(message: "unsupported")),
            .unsupportedOperation(message: "unsupported")
        )
        XCTAssertEqual(
            SippError(
                .Endpoint(
                    kind: "transport",
                    status: 503,
                    code: "unavailable",
                    message: "offline",
                    requestId: "request"
                )
            ),
            .endpoint(
                kind: "transport",
                status: 503,
                code: "unavailable",
                message: "offline",
                requestID: "request"
            )
        )
        XCTAssertEqual(
            SippError(.EndpointSelection(message: "missing")),
            .endpointSelection(message: "missing")
        )
        XCTAssertEqual(
            SippError(.Cancelled(reason: .deadlineExceeded)),
            .cancelled(reason: .deadlineExceeded)
        )
        XCTAssertEqual(
            SippError(.InvalidState(message: "consumed")),
            .invalidState(message: "consumed")
        )
        XCTAssertEqual(
            SippError(.Runtime(message: "runtime")),
            .runtime(message: "runtime")
        )
    }
}
