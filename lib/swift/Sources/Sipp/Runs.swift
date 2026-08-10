import SippCoreBindings

actor RunCancellation<Run: Sendable> {
    private let cancelRun: @Sendable (Run) -> Void
    private var run: Run?
    private var cancellationRequested = false

    init(cancelRun: @escaping @Sendable (Run) -> Void) {
        self.cancelRun = cancelRun
    }

    func install(_ run: Run) {
        self.run = run
        if cancellationRequested {
            cancelRun(run)
        }
    }

    func cancel() {
        cancellationRequested = true
        if let run {
            cancelRun(run)
        }
    }
}

final class RunLifetime<Run: Sendable>: Sendable {
    let cancellation: RunCancellation<Run>
    /// Retains model-file access until the final public run handle is released.
    private let securityScopes: SecurityScopeManager

    init(
        cancellation: RunCancellation<Run>,
        securityScopes: SecurityScopeManager
    ) {
        self.cancellation = cancellation
        self.securityScopes = securityScopes
    }

    deinit {
        let cancellation = self.cancellation
        Task {
            await cancellation.cancel()
        }
    }
}

actor TextTokenCursor {
    private let run: Task<FfiTextRun, Never>
    private let lifetime: RunLifetime<FfiTextRun>

    init(
        run: Task<FfiTextRun, Never>,
        lifetime: RunLifetime<FfiTextRun>
    ) {
        self.run = run
        self.lifetime = lifetime
    }

    func next() async -> TokenBatch? {
        let runTask = self.run
        let cancellation = lifetime.cancellation
        return await withTaskCancellationHandler {
            let bridgeRun = await runTask.value
            return await bridgeRun.nextToken().map(TokenBatch.init)
        } onCancel: {
            Task {
                await cancellation.cancel()
            }
        }
    }
}

/// An active query or chat operation.
public final class TextRun: Sendable {
    private let lifetime: RunLifetime<FfiTextRun>
    private let responseTask: Task<TextResponse, Error>

    /// The ordered, consumptive token stream for this run.
    public let tokens: TokenSequence

    init(
        securityScopes: SecurityScopeManager,
        start: @escaping @Sendable () async -> FfiTextRun
    ) {
        let cancellation = RunCancellation<FfiTextRun> { run in
            run.cancel()
        }
        let lifetime = RunLifetime(
            cancellation: cancellation,
            securityScopes: securityScopes
        )
        let runTask = Task<FfiTextRun, Never> {
            let bridgeRun = await start()
            await cancellation.install(bridgeRun)
            return bridgeRun
        }
        self.lifetime = lifetime
        tokens = TokenSequence(
            cursor: TextTokenCursor(
                run: runTask,
                lifetime: lifetime
            )
        )
        responseTask = Task<TextResponse, Error> {
            defer { _ = securityScopes }
            let bridgeRun = await runTask.value
            let response = try await callBridge {
                try await bridgeRun.takeResponse()
            }
            return TextResponse(response)
        }
    }

    /// The cached terminal response. Every await observes the same result.
    public var response: TextResponse {
        get async throws {
            let cancellation = lifetime.cancellation
            return try await withTaskCancellationHandler {
                try await responseTask.value
            } onCancel: {
                Task {
                    await cancellation.cancel()
                }
            }
        }
    }

    /// Requests cancellation through the native Sipp cancellation handle.
    public func cancel() {
        let cancellation = lifetime.cancellation
        Task {
            await cancellation.cancel()
        }
    }
}

/// The token batches emitted by a `TextRun`.
public struct TokenSequence: AsyncSequence, Sendable {
    public typealias Element = TokenBatch

    private let cursor: TextTokenCursor

    init(cursor: TextTokenCursor) {
        self.cursor = cursor
    }

    public func makeAsyncIterator() -> Iterator {
        Iterator(cursor: cursor)
    }

    public struct Iterator: AsyncIteratorProtocol {
        private let cursor: TextTokenCursor

        init(cursor: TextTokenCursor) {
            self.cursor = cursor
        }

        public mutating func next() async -> TokenBatch? {
            await cursor.next()
        }
    }
}

/// An active embedding operation.
public final class EmbeddingRun: Sendable {
    private let lifetime: RunLifetime<FfiEmbeddingRun>
    private let responseTask: Task<EmbeddingResponse, Error>

    init(
        securityScopes: SecurityScopeManager,
        start: @escaping @Sendable () async -> FfiEmbeddingRun
    ) {
        let cancellation = RunCancellation<FfiEmbeddingRun> { run in
            run.cancel()
        }
        let lifetime = RunLifetime(
            cancellation: cancellation,
            securityScopes: securityScopes
        )
        let runTask = Task<FfiEmbeddingRun, Never> {
            let bridgeRun = await start()
            await cancellation.install(bridgeRun)
            return bridgeRun
        }
        self.lifetime = lifetime
        responseTask = Task<EmbeddingResponse, Error> {
            defer { _ = securityScopes }
            let bridgeRun = await runTask.value
            let response = try await callBridge {
                try await bridgeRun.takeResponse()
            }
            return EmbeddingResponse(response)
        }
    }

    /// The cached terminal response. Every await observes the same result.
    public var response: EmbeddingResponse {
        get async throws {
            let cancellation = lifetime.cancellation
            return try await withTaskCancellationHandler {
                try await responseTask.value
            } onCancel: {
                Task {
                    await cancellation.cancel()
                }
            }
        }
    }

    /// Requests cancellation through the native Sipp cancellation handle.
    public func cancel() {
        let cancellation = lifetime.cancellation
        Task {
            await cancellation.cancel()
        }
    }
}

/// An active speech-synthesis operation.
public final class AudioRun: Sendable {
    private let lifetime: RunLifetime<FfiAudioRun>
    private let responseTask: Task<AudioResponse, Error>

    init(
        securityScopes: SecurityScopeManager,
        start: @escaping @Sendable () async -> FfiAudioRun
    ) {
        let cancellation = RunCancellation<FfiAudioRun> { run in
            run.cancel()
        }
        let lifetime = RunLifetime(
            cancellation: cancellation,
            securityScopes: securityScopes
        )
        let runTask = Task<FfiAudioRun, Never> {
            let bridgeRun = await start()
            await cancellation.install(bridgeRun)
            return bridgeRun
        }
        self.lifetime = lifetime
        responseTask = Task<AudioResponse, Error> {
            defer { _ = securityScopes }
            let bridgeRun = await runTask.value
            let response = try await callBridge {
                try await bridgeRun.takeResponse()
            }
            return AudioResponse(response)
        }
    }

    /// The cached terminal WAV response. Every await observes the same result.
    public var response: AudioResponse {
        get async throws {
            let cancellation = lifetime.cancellation
            return try await withTaskCancellationHandler {
                try await responseTask.value
            } onCancel: {
                Task {
                    await cancellation.cancel()
                }
            }
        }
    }

    /// Requests cancellation through the native Sipp cancellation handle.
    public func cancel() {
        let cancellation = lifetime.cancellation
        Task {
            await cancellation.cancel()
        }
    }
}
