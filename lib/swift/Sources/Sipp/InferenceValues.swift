import Foundation
import SippCoreBindings

/// A role/content pair in an ordered chat conversation.
public struct ChatMessage: Equatable, Sendable {
    public let role: ChatRole
    public let content: String

    public init(role: ChatRole, content: String) {
        self.role = role
        self.content = content
    }
}

/// The speaker associated with a chat message.
public enum ChatRole: Equatable, Sendable {
    case system
    case user
    case assistant
}

/// Provider-neutral text generation options.
public struct TextOptions: Equatable, Sendable {
    public var maxTokens: UInt32?
    public var temperature: Float?
    public var topP: Float?
    public var stop: [String]

    public init(
        maxTokens: UInt32? = nil,
        temperature: Float? = nil,
        topP: Float? = nil,
        stop: [String] = []
    ) {
        self.maxTokens = maxTokens
        self.temperature = temperature
        self.topP = topP
        self.stop = stop
    }
}

/// Options specific to local text inference.
public struct LocalTextOptions: Equatable, Sendable {
    public var contextKey: String?
    public var grammar: String?
    public var jsonSchema: String?
    public var media: [Data]

    public init(
        contextKey: String? = nil,
        grammar: String? = nil,
        jsonSchema: String? = nil,
        media: [Data] = []
    ) {
        self.contextKey = contextKey
        self.grammar = grammar
        self.jsonSchema = jsonSchema
        self.media = media
    }
}

/// Options specific to local embedding inference.
public struct LocalEmbedOptions: Equatable, Sendable {
    public var contextKey: String?
    public var normalize: Bool?

    public init(contextKey: String? = nil, normalize: Bool? = nil) {
        self.contextKey = contextKey
        self.normalize = normalize
    }
}

/// Why text generation stopped.
public enum FinishReason: Equatable, Sendable {
    case stop
    case length
    case cancelled
    case error
}

/// Token accounting reported by an inference endpoint.
public struct TokenUsage: Equatable, Sendable {
    public let inputTokens: UInt32?
    public let outputTokens: UInt32?
    public let totalTokens: UInt32?
}

/// The local KV-cache reuse mode used by a request.
public enum CacheMode: Equatable, Sendable {
    case disabled
    case liveSlotPrefix
    case stateSnapshot
    case liveSlotAndSnapshot
}

/// The source of the cached prefix used by a request.
public enum CacheSource: Equatable, Sendable {
    case none
    case live
    case snapshot
}

/// Local timing, throughput, and cache statistics for a completed request.
public struct RequestStats: Equatable, Sendable {
    public let inputTokens: Int32
    public let outputTokens: Int32
    public let cacheMode: CacheMode
    public let cacheSource: CacheSource
    public let cacheHits: Int32
    public let prefillTokens: Int32
    public let timeToFirstTokenMilliseconds: Double?
    public let interTokenMilliseconds: Double?
    public let endToEndMilliseconds: Double?
    public let endToEndTokensPerSecond: Double?
    public let decodeTokensPerSecond: Double?
    public let prefillTokensPerSecond: Double?
    public let prefillMilliseconds: Double
    public let decodeMilliseconds: Double
}

/// Request identifiers preserved across the inference boundary.
public struct ResponseMetadata: Equatable, Sendable {
    public let requestID: String?
    public let upstreamRequestID: String?
    public let upstreamResponseID: String?
}

/// The terminal value of a query or chat run.
public struct TextResponse: Equatable, Sendable {
    public let endpoint: String
    public let text: String
    public let finishReason: FinishReason
    public let usage: TokenUsage?
    public let localStats: RequestStats?
    public let metadata: ResponseMetadata
}

/// The pooling strategy reported by an embedding endpoint.
public enum PoolingType: Equatable, Sendable {
    case unspecified
    case none
    case mean
    case cls
    case last
    case rank
}

/// The terminal value of an embedding run.
public struct EmbeddingResponse: Equatable, Sendable {
    public let endpoint: String
    public let values: [Float]
    public let usage: TokenUsage?
    public let localStats: RequestStats?
    public let pooling: PoolingType?
    public let normalized: Bool?
    public let metadata: ResponseMetadata
}

/// Cumulative transport counters attached to a token batch.
public struct TokenEmissionStats: Equatable, Sendable {
    public let framesSent: UInt64
    public let bytesSent: UInt64
    public let batchesSent: UInt64
}

/// An ordered batch emitted by a text run.
public struct TokenBatch: Equatable, Sendable {
    public let requestID: String
    public let streamID: UInt32
    public let sequenceStart: UInt32
    public let text: String
    public let frameCount: UInt32
    public let byteCount: UInt32
    public let stats: TokenEmissionStats
}

extension ChatMessage {
    var bridgeValue: FfiChatMessage {
        FfiChatMessage(role: role.bridgeValue, content: content)
    }
}

extension ChatRole {
    var bridgeValue: FfiChatRole {
        switch self {
        case .system:
            .system
        case .user:
            .user
        case .assistant:
            .assistant
        }
    }
}

extension TextOptions {
    var bridgeValue: FfiTextOptions {
        FfiTextOptions(
            maxTokens: maxTokens,
            temperature: temperature,
            topP: topP,
            stop: stop
        )
    }
}

extension LocalTextOptions {
    var bridgeValue: FfiLocalTextOptions {
        FfiLocalTextOptions(
            contextKey: contextKey,
            grammar: grammar,
            jsonSchema: jsonSchema,
            media: media
        )
    }
}

extension LocalEmbedOptions {
    var bridgeValue: FfiLocalEmbedOptions {
        FfiLocalEmbedOptions(contextKey: contextKey, normalize: normalize)
    }
}

extension FinishReason {
    init(_ reason: FfiFinishReason) {
        switch reason {
        case .stop:
            self = .stop
        case .length:
            self = .length
        case .cancelled:
            self = .cancelled
        case .error:
            self = .error
        }
    }
}

extension TokenUsage {
    init(_ usage: FfiTokenUsage) {
        inputTokens = usage.inputTokens
        outputTokens = usage.outputTokens
        totalTokens = usage.totalTokens
    }
}

extension CacheMode {
    init(_ mode: FfiCacheMode) {
        switch mode {
        case .disabled:
            self = .disabled
        case .liveSlotPrefix:
            self = .liveSlotPrefix
        case .stateSnapshot:
            self = .stateSnapshot
        case .liveSlotAndSnapshot:
            self = .liveSlotAndSnapshot
        }
    }
}

extension CacheSource {
    init(_ source: FfiCacheSource) {
        switch source {
        case .none:
            self = .none
        case .live:
            self = .live
        case .snapshot:
            self = .snapshot
        }
    }
}

extension RequestStats {
    init(_ stats: FfiRequestStats) {
        inputTokens = stats.inputTokens
        outputTokens = stats.outputTokens
        cacheMode = CacheMode(stats.cacheMode)
        cacheSource = CacheSource(stats.cacheSource)
        cacheHits = stats.cacheHits
        prefillTokens = stats.prefillTokens
        timeToFirstTokenMilliseconds = stats.ttftMs
        interTokenMilliseconds = stats.interTokenMs
        endToEndMilliseconds = stats.e2eMs
        endToEndTokensPerSecond = stats.e2eTokensPerSecond
        decodeTokensPerSecond = stats.decodeTokensPerSecond
        prefillTokensPerSecond = stats.prefillTokensPerSecond
        prefillMilliseconds = stats.prefillMs
        decodeMilliseconds = stats.decodeMs
    }
}

extension ResponseMetadata {
    init(_ metadata: FfiResponseMetadata) {
        requestID = metadata.requestId
        upstreamRequestID = metadata.upstreamRequestId
        upstreamResponseID = metadata.upstreamResponseId
    }
}

extension TextResponse {
    init(_ response: FfiTextResponse) {
        endpoint = response.endpoint
        text = response.text
        finishReason = FinishReason(response.finishReason)
        usage = response.usage.map(TokenUsage.init)
        localStats = response.localStats.map(RequestStats.init)
        metadata = ResponseMetadata(response.metadata)
    }
}

extension PoolingType {
    init(_ pooling: FfiPoolingType) {
        switch pooling {
        case .unspecified:
            self = .unspecified
        case .none:
            self = .none
        case .mean:
            self = .mean
        case .cls:
            self = .cls
        case .last:
            self = .last
        case .rank:
            self = .rank
        }
    }
}

extension EmbeddingResponse {
    init(_ response: FfiEmbeddingResponse) {
        endpoint = response.endpoint
        values = response.values
        usage = response.usage.map(TokenUsage.init)
        localStats = response.localStats.map(RequestStats.init)
        pooling = response.pooling.map(PoolingType.init)
        normalized = response.normalized
        metadata = ResponseMetadata(response.metadata)
    }
}

extension TokenEmissionStats {
    init(_ stats: FfiTokenEmissionStats) {
        framesSent = stats.framesSent
        bytesSent = stats.bytesSent
        batchesSent = stats.batchesSent
    }
}

extension TokenBatch {
    init(_ batch: FfiTokenBatch) {
        requestID = batch.requestId
        streamID = batch.streamId
        sequenceStart = batch.sequenceStart
        text = batch.text
        frameCount = batch.frameCount
        byteCount = batch.byteCount
        stats = TokenEmissionStats(batch.stats)
    }
}
