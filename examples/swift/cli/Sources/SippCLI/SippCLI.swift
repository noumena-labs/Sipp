import Darwin
import Foundation
import Sipp

private let endpointID = "local"

private enum Operation: String {
    case query
    case chat
    case embed
    case cancel
}

private enum CLIError: Error, LocalizedError {
    case usage
    case cancellationWasNotObserved

    var errorDescription: String? {
        switch self {
        case .usage:
            "Usage: SippCLI <query|chat|embed|cancel> <model.gguf> <input>"
        case .cancellationWasNotObserved:
            "The cancellation example completed without a cancellation error"
        }
    }
}

@main
private struct SippCLI {
    static func main() async {
        do {
            try await run()
        } catch {
            let message = (error as NSError).localizedDescription
            FileHandle.standardError.write(Data("SippCLI: \(message)\n".utf8))
            exit(EXIT_FAILURE)
        }
    }

    private static func run() async throws {
        let arguments = Array(CommandLine.arguments.dropFirst())
        guard arguments.count >= 3,
              let operation = Operation(rawValue: arguments[0])
        else {
            throw CLIError.usage
        }

        let modelURL = URL(fileURLWithPath: arguments[1]).standardizedFileURL
        let input = arguments.dropFirst(2).joined(separator: " ")
        let client = try SippClient()
        let model = try await client.models.add([modelURL])
        try await client.add(endpointID, model: model)

        switch operation {
        case .query:
            let response = try await client.query(
                input,
                endpoint: endpointID,
                options: TextOptions(maxTokens: 256, temperature: 0.7)
            ).response
            print(response.text)
            printTextSummary(response)
        case .chat:
            let run = client.chat(
                messages: [
                    ChatMessage(role: .system, content: "Answer concisely."),
                    ChatMessage(role: .user, content: input),
                ],
                endpoint: endpointID,
                options: TextOptions(maxTokens: 256, temperature: 0.7)
            )
            for await batch in run.tokens {
                print(batch.text, terminator: "")
            }
            print()
            printTextSummary(try await run.response)
        case .embed:
            let response = try await client.embed(
                input,
                endpoint: endpointID,
                local: LocalEmbedOptions(normalize: true)
            ).response
            let preview = response.values.prefix(8).map {
                String(format: "%.6f", $0)
            }
            print("endpoint=\(response.endpoint)")
            print("dimensions=\(response.values.count)")
            print("preview=[\(preview.joined(separator: ", "))]")
        case .cancel:
            let run = client.chat(
                messages: [ChatMessage(role: .user, content: input)],
                endpoint: endpointID,
                options: TextOptions(maxTokens: 2048, temperature: 0.7)
            )
            run.cancel()
            do {
                _ = try await run.response
                throw CLIError.cancellationWasNotObserved
            } catch let SippError.cancelled(reason) {
                print("cancellation=\(reason)")
            }
        }

        try await client.remove(endpointID)
        try await client.models.remove(model)
    }

    private static func printTextSummary(_ response: TextResponse) {
        print("endpoint=\(response.endpoint)")
        print("finishReason=\(response.finishReason)")
        if let stats = response.localStats {
            print("outputTokens=\(stats.outputTokens)")
            if let rate = stats.decodeTokensPerSecond {
                print("decodeTokensPerSecond=\(rate)")
            }
        }
    }
}
