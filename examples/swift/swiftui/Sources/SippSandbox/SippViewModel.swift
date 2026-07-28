import Combine
import Foundation
import Sipp

@MainActor
final class SippViewModel: ObservableObject {
    enum Operation: String, CaseIterable {
        case query
        case chat
        case embed
    }

    private static let endpointID = "local"

    @Published var isImporterPresented = false
    @Published var operation = Operation.chat
    @Published var input = "Explain on-device inference"
    @Published private(set) var modelName = "No model selected"
    @Published private(set) var output = ""
    @Published private(set) var status = "Choose a local GGUF model"
    @Published private(set) var errorMessage: String?
    @Published private(set) var isRunning = false
    @Published private(set) var canCancel = false
    @Published private(set) var hasModel = false

    private let clientState: Result<SippClient, Error>
    private var operationTask: Task<Void, Never>?
    private var activeTextRun: TextRun?
    private var activeEmbeddingRun: EmbeddingRun?

    init() {
        do {
            clientState = .success(try SippClient())
        } catch {
            let message = "SippClient initialization failed: \(error.localizedDescription)"
            clientState = .failure(error)
            status = message
            errorMessage = message
        }
    }

    var canRun: Bool {
        hasModel && !isRunning && !input.isEmpty
    }

    func selectModel(_ result: Result<URL, Error>) {
        switch result {
        case let .success(url):
            begin("Registering \(url.lastPathComponent)")
            operationTask = Task {
                do {
                    let client = try clientState.get()
                    let model = try await client.models.add([url])
                    try await client.add(Self.endpointID, model: model)
                    hasModel = true
                    modelName = model.name
                    finish("Model ready")
                } catch let error as SippError {
                    fail("Sipp failed: \(error.localizedDescription)")
                } catch {
                    fail("Model selection failed: \(error.localizedDescription)")
                }
            }
        case let .failure(error):
            fail("File importer failed: \(error.localizedDescription)")
        }
    }

    func run() {
        begin("Running \(operation.rawValue)")
        operationTask = Task {
            do {
                let client = try clientState.get()
                switch operation {
                case .query:
                    try await consume(
                        client.query(
                            input,
                            endpoint: Self.endpointID,
                            options: TextOptions(maxTokens: 256, temperature: 0.7)
                        )
                    )
                case .chat:
                    try await consume(
                        client.chat(
                            messages: [
                                ChatMessage(role: .system, content: "Answer concisely."),
                                ChatMessage(role: .user, content: input),
                            ],
                            endpoint: Self.endpointID,
                            options: TextOptions(maxTokens: 256, temperature: 0.7)
                        )
                    )
                case .embed:
                    try await runEmbedding(client)
                }
                finish("Completed \(operation.rawValue)")
            } catch let SippError.cancelled(reason) {
                finish("Cancelled: \(reason)")
            } catch let error as SippError {
                fail("Sipp failed: \(error.localizedDescription)")
            } catch is CancellationError {
                finish("Cancelled")
            } catch {
                fail("Unexpected failure: \(error.localizedDescription)")
            }
        }
    }

    func cancel() {
        activeTextRun?.cancel()
        activeEmbeddingRun?.cancel()
        operationTask?.cancel()
        status = "Cancellation requested"
    }

    private func consume(_ run: TextRun) async throws {
        activeTextRun = run
        canCancel = true
        output = ""
        for await batch in run.tokens {
            output += batch.text
        }
        let response = try await run.response
        output = response.text
        activeTextRun = nil
    }

    private func runEmbedding(_ client: SippClient) async throws {
        let run = client.embed(
            input,
            endpoint: Self.endpointID,
            local: LocalEmbedOptions(normalize: true)
        )
        activeEmbeddingRun = run
        canCancel = true
        let response = try await run.response
        let preview = response.values.prefix(8).map {
            String(format: "%.6f", $0)
        }
        output = "Dimensions: \(response.values.count)\n\n\(preview.joined(separator: ", "))"
        activeEmbeddingRun = nil
    }

    private func begin(_ message: String) {
        isRunning = true
        errorMessage = nil
        output = ""
        status = message
    }

    private func finish(_ message: String) {
        isRunning = false
        activeTextRun = nil
        activeEmbeddingRun = nil
        canCancel = false
        status = message
    }

    private func fail(_ message: String) {
        finish("Failed")
        errorMessage = message
    }
}
