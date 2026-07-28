import Foundation
import SippCoreBindings

/// A stable failure reported by the Sipp Swift API.
public enum SippError: Error, Equatable, LocalizedError, Sendable {
    /// Model registration, storage, or lifecycle failed.
    case modelLifecycle(
        code: String,
        message: String,
        status: UInt16?,
        retryAfterMilliseconds: UInt64?
    )
    /// A request contained invalid input.
    case invalidArgument(message: String)
    /// The selected endpoint or runtime does not support the operation.
    case unsupportedOperation(message: String)
    /// An inference endpoint returned a structured failure.
    case endpoint(
        kind: String,
        status: UInt16?,
        code: String?,
        message: String,
        requestID: String?
    )
    /// Sipp could not select an endpoint for the request.
    case endpointSelection(message: String)
    /// An in-flight run was cancelled.
    case cancelled(reason: CancellationReason)
    /// An operation was attempted after a one-shot native value was consumed.
    case invalidState(message: String)
    /// The native runtime failed.
    case runtime(message: String)
    /// A URL cannot be used for the requested operation.
    case invalidURL(URL)
    /// The Application Support storage location could not be opened.
    case storageLocation(message: String)
    /// Security-scoped access or bookmark creation failed for a model URL.
    case securityScopedAccess(url: URL, message: String)
    /// The persistent security-scoped bookmark registry could not be used.
    case bookmarkStore(message: String)
    /// The bookmark registry uses an unsupported schema version.
    case bookmarkVersion(found: UInt32, expected: UInt32)
    /// Native registration succeeded but its bookmark transaction could not commit.
    case bookmarkTransaction(commit: String, rollback: String)

    public var errorDescription: String? {
        switch self {
        case let .modelLifecycle(_, message, _, _),
            let .invalidArgument(message),
            let .unsupportedOperation(message),
            let .endpoint(_, _, _, message, _),
            let .endpointSelection(message),
            let .invalidState(message),
            let .runtime(message),
            let .storageLocation(message),
            let .bookmarkStore(message):
            message
        case let .cancelled(reason):
            "Sipp run cancelled: \(reason.description)"
        case let .invalidURL(url):
            "Unsupported Sipp URL: \(url.absoluteString)"
        case let .securityScopedAccess(url, message):
            "Security-scoped access failed for \(url.lastPathComponent): \(message)"
        case let .bookmarkVersion(found, expected):
            "Unsupported security bookmark version \(found); expected \(expected)"
        case let .bookmarkTransaction(commit, rollback):
            "Bookmark commit failed (\(commit)); native rollback failed (\(rollback))"
        }
    }
}

/// Why an in-flight Sipp operation stopped.
public enum CancellationReason: Equatable, Sendable {
    case callerCancelled
    case clientDisconnected
    case serverShutdown
    case deadlineExceeded

    fileprivate var description: String {
        switch self {
        case .callerCancelled:
            "caller cancelled"
        case .clientDisconnected:
            "client disconnected"
        case .serverShutdown:
            "server shutdown"
        case .deadlineExceeded:
            "deadline exceeded"
        }
    }
}

extension SippError {
    init(_ error: FfiError) {
        switch error {
        case let .ModelLifecycle(code, message, status, retryAfterMs):
            self = .modelLifecycle(
                code: code,
                message: message,
                status: status,
                retryAfterMilliseconds: retryAfterMs
            )
        case let .InvalidArgument(message):
            self = .invalidArgument(message: message)
        case let .UnsupportedOperation(message):
            self = .unsupportedOperation(message: message)
        case let .Endpoint(kind, status, code, message, requestId):
            self = .endpoint(
                kind: kind,
                status: status,
                code: code,
                message: message,
                requestID: requestId
            )
        case let .EndpointSelection(message):
            self = .endpointSelection(message: message)
        case let .Cancelled(reason):
            self = .cancelled(reason: CancellationReason(reason))
        case let .InvalidState(message):
            self = .invalidState(message: message)
        case let .Runtime(message):
            self = .runtime(message: message)
        }
    }
}

extension CancellationReason {
    init(_ reason: FfiCancellationReason) {
        switch reason {
        case .callerCancelled:
            self = .callerCancelled
        case .clientDisconnected:
            self = .clientDisconnected
        case .serverShutdown:
            self = .serverShutdown
        case .deadlineExceeded:
            self = .deadlineExceeded
        }
    }
}
