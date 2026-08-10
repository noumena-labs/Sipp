import Darwin
import Foundation
import Sipp

private let endpointID = "local"

private enum Operation: String {
    case query
    case chat
    case embed
    case listen
    case speak
    case cancel
}

private enum CLIError: Error, LocalizedError {
    case usage
    case cancellationWasNotObserved
    case invalidEnvironment(name: String, value: String)

    var errorDescription: String? {
        switch self {
        case .usage:
            "Usage: SippCLI <query|chat|embed|cancel> <model.gguf> <input>\n" +
                "       SippCLI listen <model.gguf> <projector.gguf> <audio>\n" +
                "       SippCLI speak <model.gguf> <projector.gguf> <output.wav> [text]"
        case .cancellationWasNotObserved:
            "The cancellation example completed without a cancellation error"
        case let .invalidEnvironment(name, value):
            "\(name) must be a positive 32-bit integer; received \(value)"
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
        let isSpeech = operation == .listen || operation == .speak
        guard !isSpeech || arguments.count >= 4 else {
            throw CLIError.usage
        }
        let modelSources = isSpeech
            ? [
                modelURL,
                URL(fileURLWithPath: arguments[2]).standardizedFileURL,
            ]
            : [modelURL]
        let input = arguments.dropFirst(isSpeech ? 3 : 2).joined(separator: " ")
        let client = try SippClient()
        let model = try await client.models.add(modelSources)
        let endpoint = try await client.add(endpointID, .local(model))

        switch operation {
        case .query:
            let response = try await client.query(
                input,
                endpoint: endpoint,
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
                endpoint: endpoint,
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
                endpoint: endpoint,
                local: LocalEmbedOptions(normalize: true)
            ).response
            let preview = response.values.prefix(8).map {
                String(format: "%.6f", $0)
            }
            print("endpoint=\(response.endpoint.id)")
            print("dimensions=\(response.values.count)")
            print("preview=[\(preview.joined(separator: ", "))]")
        case .listen:
            let audio = try Data(contentsOf: URL(fileURLWithPath: arguments[3]))
            let response = try await client.listen(
                audio,
                endpoint: endpoint,
                language: ProcessInfo.processInfo.environment["SIPP_LANGUAGE"]
            ).response
            print(response.text)
        case .speak:
            let text = arguments.dropFirst(4).joined(separator: " ")
            let speakerAudio = try ProcessInfo.processInfo.environment["SIPP_SPEAKER_AUDIO"]
                .map { try Data(contentsOf: URL(fileURLWithPath: $0)) }
            let maxDurationMs = try positiveUInt32Environment("SIPP_MAX_DURATION_MS")
            let response = try await client.speak(
                text.isEmpty ? "Hello from Sipp." : text,
                endpoint: endpoint,
                language: ProcessInfo.processInfo.environment["SIPP_LANGUAGE"],
                speakerAudio: speakerAudio,
                maxDurationMs: maxDurationMs
            ).response
            let outputURL = URL(fileURLWithPath: arguments[3])
            try response.audio.write(to: outputURL)
            print("wrote \(response.durationMs) ms at \(response.sampleRateHz) Hz to \(outputURL.path)")
        case .cancel:
            let run = client.chat(
                messages: [ChatMessage(role: .user, content: input)],
                endpoint: endpoint,
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
        print("endpoint=\(response.endpoint.id)")
        print("finishReason=\(response.finishReason)")
        if let stats = response.localStats {
            print("outputTokens=\(stats.outputTokens)")
            if let rate = stats.decodeTokensPerSecond {
                print("decodeTokensPerSecond=\(rate)")
            }
        }
    }

    private static func positiveUInt32Environment(_ name: String) throws -> UInt32? {
        guard let value = ProcessInfo.processInfo.environment[name] else {
            return nil
        }
        guard let parsed = UInt32(value), parsed > 0 else {
            throw CLIError.invalidEnvironment(name: name, value: value)
        }
        return parsed
    }
}
