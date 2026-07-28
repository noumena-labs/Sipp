import SippCoreBindings
import XCTest
@testable import Sipp

final class BridgeValueTests: XCTestCase {
    func testManagedModelUsesSwiftValues() {
        let model = ManagedModel(
            FfiManagedModel(
                id: "model-id",
                name: "Model",
                bytes: 42,
                modality: .vision,
                status: .needsProjector
            )
        )

        XCTAssertEqual(model.id, "model-id")
        XCTAssertEqual(model.modality, .vision)
        XCTAssertEqual(model.status, .needsProjector)
    }

    func testTextResponsePreservesEveryBridgeField() {
        let stats = FfiRequestStats(
            inputTokens: 4,
            outputTokens: 2,
            cacheMode: .liveSlotAndSnapshot,
            cacheSource: .snapshot,
            cacheHits: 3,
            prefillTokens: 1,
            ttftMs: 4.5,
            interTokenMs: 2.5,
            e2eMs: 9.0,
            e2eTokensPerSecond: 8.0,
            decodeTokensPerSecond: 7.0,
            prefillTokensPerSecond: 6.0,
            prefillMs: 3.0,
            decodeMs: 5.0
        )
        let response = TextResponse(
            FfiTextResponse(
                endpoint: "chat",
                text: "hello",
                finishReason: .length,
                usage: FfiTokenUsage(inputTokens: 4, outputTokens: 2, totalTokens: 6),
                localStats: stats,
                metadata: FfiResponseMetadata(
                    requestId: "request",
                    upstreamRequestId: "upstream-request",
                    upstreamResponseId: "upstream-response"
                )
            )
        )

        XCTAssertEqual(response.endpoint, "chat")
        XCTAssertEqual(response.text, "hello")
        XCTAssertEqual(response.finishReason, .length)
        XCTAssertEqual(response.usage?.totalTokens, 6)
        XCTAssertEqual(response.localStats?.cacheMode, .liveSlotAndSnapshot)
        XCTAssertEqual(response.localStats?.cacheSource, .snapshot)
        XCTAssertEqual(response.localStats?.timeToFirstTokenMilliseconds, 4.5)
        XCTAssertEqual(response.metadata.requestID, "request")
        XCTAssertEqual(response.metadata.upstreamRequestID, "upstream-request")
        XCTAssertEqual(response.metadata.upstreamResponseID, "upstream-response")
    }

    func testEmbeddingAndTokenBatchPreserveEveryBridgeCategory() {
        let embedding = EmbeddingResponse(
            FfiEmbeddingResponse(
                endpoint: "embed",
                values: [1, 2],
                usage: nil,
                localStats: nil,
                pooling: .cls,
                normalized: true,
                metadata: FfiResponseMetadata(
                    requestId: nil,
                    upstreamRequestId: nil,
                    upstreamResponseId: nil
                )
            )
        )
        let batch = TokenBatch(
            FfiTokenBatch(
                requestId: "request",
                streamId: 3,
                sequenceStart: 4,
                text: "hello",
                frameCount: 2,
                byteCount: 5,
                stats: FfiTokenEmissionStats(framesSent: 2, bytesSent: 5, batchesSent: 1)
            )
        )

        XCTAssertEqual(embedding.pooling, .cls)
        XCTAssertEqual(embedding.normalized, true)
        XCTAssertEqual(embedding.values, [1, 2])
        XCTAssertEqual(batch.requestID, "request")
        XCTAssertEqual(batch.streamID, 3)
        XCTAssertEqual(batch.sequenceStart, 4)
        XCTAssertEqual(batch.stats, TokenEmissionStats(framesSent: 2, bytesSent: 5, batchesSent: 1))
    }
}
