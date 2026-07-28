import Foundation
import XCTest
@testable import Sipp

final class PublicAPITests: XCTestCase {
    func testInferenceOptionsHaveNoImplicitPolicy() {
        XCTAssertEqual(TextOptions(), TextOptions(maxTokens: nil, temperature: nil, topP: nil))
        XCTAssertEqual(LocalTextOptions(), LocalTextOptions(media: []))
        XCTAssertEqual(LocalEmbedOptions(), LocalEmbedOptions(normalize: nil))
    }

    // This fixture compiles the intended consumer surface without starting a
    // native run. The macOS package gate executes the corresponding flows.
    private func compilePublicSurface(client: SippClient, modelURL: URL) async throws {
        let model = try await client.models.add([modelURL])
        try await client.add("chat", model: model)

        let run = client.chat(
            messages: [ChatMessage(role: .user, content: "Explain on-device inference")],
            endpoint: "chat"
        )
        for await batch in run.tokens {
            _ = batch.text
        }
        _ = try await run.response
        run.cancel()

        _ = client.query("Hello", endpoint: "chat")
        _ = client.embed("Hello", endpoint: "chat")
        try await client.remove("chat")
        try await client.models.remove(model)
    }
}
