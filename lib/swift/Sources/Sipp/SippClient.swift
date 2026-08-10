import Foundation
import SippCoreBindings

actor LifecycleOperationQueue {
    private var active = false
    private var waiters = [CheckedContinuation<Void, Never>]()
    private var nextWaiterIndex = 0

    func perform<Value: Sendable>(
        _ operation: @Sendable () async throws -> Value
    ) async throws -> Value {
        await acquire()
        defer { release() }
        return try await operation()
    }

    private func acquire() async {
        if !active {
            active = true
            return
        }
        await withCheckedContinuation { continuation in
            waiters.append(continuation)
        }
    }

    private func release() {
        if nextWaiterIndex == waiters.count {
            active = false
            waiters.removeAll(keepingCapacity: true)
            nextWaiterIndex = 0
        } else {
            let waiter = waiters[nextWaiterIndex]
            nextWaiterIndex += 1
            waiter.resume()
        }
    }
}

/// A client for on-device Sipp model and inference operations.
public final class SippClient: Sendable {
    private let bridge: FfiSippClient
    private let lifecycleOperations: LifecycleOperationQueue
    private let securityScopes: SecurityScopeManager

    /// The model store owned by this client.
    public let models: ModelStore

    /// Opens a client at an explicit file URL or at the user's Application
    /// Support directory when `storageRoot` is `nil`.
    public init(storageRoot: URL? = nil) throws {
        let root: URL
        if let storageRoot {
            root = storageRoot
        } else {
            root = try defaultStorageRoot()
        }
        let path = try nativeStoragePath(root)
        let localSourceRoot = sandboxLocalSourceRoot()
        let restoration = try SecurityScopeRestoration.load(storageRoot: root)
        let bridge = try callBridge {
            try FfiSippClient(
                storageRoot: path,
                localSourceRoot: try localSourceRoot.map(nativeStoragePath)
            )
        }
        let securityScopes = SecurityScopeManager(restoration: restoration)
        let lifecycleOperations = LifecycleOperationQueue()
        self.bridge = bridge
        self.lifecycleOperations = lifecycleOperations
        self.securityScopes = securityScopes
        models = ModelStore(
            bridge: bridge.models(),
            lifecycleOperations: lifecycleOperations,
            securityScopes: securityScopes,
            localSourceRoot: localSourceRoot
        )
    }

    /// Registers or replaces a named endpoint.
    public func add(_ id: String, _ endpoint: Endpoint) async throws -> EndpointRef {
        try await lifecycleOperations.perform {
            try await securityScopes.activate(modelID: endpoint.model.id)
            return try await callBridge {
                EndpointRef(try await self.bridge.add(id: id, endpoint: endpoint.bridgeValue))
            }
        }
    }

    /// Removes a registered endpoint.
    public func remove(_ id: String) async throws {
        try await lifecycleOperations.perform {
            try await callBridge {
                try await self.bridge.remove(id: id)
            }
        }
    }

    /// Starts raw-prompt text generation.
    public func query(
        _ prompt: String,
        endpoint: EndpointRef? = nil,
        requestID: String? = nil,
        options: TextOptions = TextOptions(),
        local: LocalTextOptions = LocalTextOptions()
    ) -> TextRun {
        let request = FfiQueryRequest(
            requestId: requestID,
            endpoint: endpoint?.id,
            prompt: prompt,
            options: options.bridgeValue,
            local: local.bridgeValue,
            emitTokens: true
        )
        let bridge = self.bridge
        return TextRun(securityScopes: securityScopes) {
            await bridge.query(request: request)
        }
    }

    /// Starts chat generation from ordered conversation messages.
    public func chat(
        messages: [ChatMessage],
        endpoint: EndpointRef? = nil,
        requestID: String? = nil,
        options: TextOptions = TextOptions(),
        local: LocalTextOptions = LocalTextOptions()
    ) -> TextRun {
        let request = FfiChatRequest(
            requestId: requestID,
            endpoint: endpoint?.id,
            messages: messages.map(\.bridgeValue),
            options: options.bridgeValue,
            local: local.bridgeValue,
            emitTokens: true
        )
        let bridge = self.bridge
        return TextRun(securityScopes: securityScopes) {
            await bridge.chat(request: request)
        }
    }

    /// Starts single-input embedding generation.
    public func embed(
        _ input: String,
        endpoint: EndpointRef? = nil,
        requestID: String? = nil,
        local: LocalEmbedOptions = LocalEmbedOptions()
    ) -> EmbeddingRun {
        let request = FfiEmbedRequest(
            requestId: requestID,
            endpoint: endpoint?.id,
            input: input,
            local: local.bridgeValue
        )
        let bridge = self.bridge
        return EmbeddingRun(securityScopes: securityScopes) {
            await bridge.embed(request: request)
        }
    }

    /// Starts speech recognition from encoded WAV, MP3, or FLAC audio.
    /// A nil `maxTokens` value uses the core transcription default.
    public func listen(
        _ audio: Data,
        endpoint: EndpointRef? = nil,
        requestID: String? = nil,
        language: String? = nil,
        maxTokens: UInt32? = nil
    ) -> TextRun {
        let request = FfiListenRequest(
            requestId: requestID,
            endpoint: endpoint?.id,
            audio: audio,
            language: language,
            maxTokens: maxTokens
        )
        let bridge = self.bridge
        return TextRun(securityScopes: securityScopes) {
            await bridge.listen(request: request)
        }
    }

    /// Starts speech synthesis and returns a mono PCM16 WAV response.
    /// A nil `maxDurationMs` uses the loaded model adapter's generation default.
    public func speak(
        _ text: String,
        endpoint: EndpointRef? = nil,
        requestID: String? = nil,
        language: String? = nil,
        speakerAudio: Data? = nil,
        maxDurationMs: UInt32? = nil
    ) -> AudioRun {
        let request = FfiSpeakRequest(
            requestId: requestID,
            endpoint: endpoint?.id,
            text: text,
            language: language,
            speakerAudio: speakerAudio,
            maxDurationMs: maxDurationMs
        )
        let bridge = self.bridge
        return AudioRun(securityScopes: securityScopes) {
            await bridge.speak(request: request)
        }
    }
}

/// Model registration operations scoped to one Sipp client.
public struct ModelStore: Sendable {
    private let bridge: FfiModelStore
    private let lifecycleOperations: LifecycleOperationQueue
    private let securityScopes: SecurityScopeManager
    private let localSourceRoot: URL?

    init(
        bridge: FfiModelStore,
        lifecycleOperations: LifecycleOperationQueue,
        securityScopes: SecurityScopeManager,
        localSourceRoot: URL?
    ) {
        self.bridge = bridge
        self.lifecycleOperations = lifecycleOperations
        self.securityScopes = securityScopes
        self.localSourceRoot = localSourceRoot
    }

    /// Registers local file URLs or HTTP(S) sources as one model.
    public func add(_ sources: [URL]) async throws -> ManagedModel {
        try await lifecycleOperations.perform {
            let acquisition = try AcquiredSecurityScopes.acquire(
                for: sources,
                localSourceRoot: localSourceRoot
            )
            let nativeSources = try sources.map(nativeModelSource)
            let registration = try await callBridge {
                try await self.bridge.add(sources: nativeSources)
            }

            do {
                try await self.securityScopes.register(
                    acquisition,
                    for: registration.model.id
                )
            } catch {
                let commitError = error
                if registration.created {
                    do {
                        try await callBridge {
                            try await self.bridge.remove(modelId: registration.model.id)
                        }
                    } catch {
                        throw SippError.bookmarkTransaction(
                            commit: commitError.localizedDescription,
                            rollback: error.localizedDescription
                        )
                    }
                }
                throw commitError
            }
            return ManagedModel(registration.model)
        }
    }

    /// Lists the models registered in this client.
    public func list() async throws -> [ManagedModel] {
        try await lifecycleOperations.perform {
            let models = try await callBridge {
                try await self.bridge.list()
            }
            return models.map(ManagedModel.init)
        }
    }

    /// Removes a model that no endpoint currently uses.
    public func remove(_ model: ManagedModel) async throws {
        try await lifecycleOperations.perform {
            try await callBridge {
                try await self.bridge.remove(modelId: model.id)
            }
            try await self.securityScopes.remove(modelID: model.id)
        }
    }
}
