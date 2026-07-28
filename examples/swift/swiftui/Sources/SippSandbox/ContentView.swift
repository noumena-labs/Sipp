import SwiftUI
import UniformTypeIdentifiers

struct ContentView: View {
    @ObservedObject var model: SippViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Button("Choose GGUF") {
                    model.isImporterPresented = true
                }
                .disabled(model.hasModel || model.isRunning)

                Text(model.modelName)
                    .lineLimit(1)
                    .truncationMode(.middle)
            }

            Picker("Operation", selection: $model.operation) {
                ForEach(SippViewModel.Operation.allCases, id: \.self) { operation in
                    Text(operation.rawValue.capitalized).tag(operation)
                }
            }
            .pickerStyle(.segmented)

            TextEditor(text: $model.input)
                .font(.system(.body, design: .monospaced))
                .frame(minHeight: 100)
                .border(Color.secondary)

            HStack {
                Button("Run") {
                    model.run()
                }
                .keyboardShortcut(.return, modifiers: [.command])
                .disabled(!model.canRun)

                Button("Cancel") {
                    model.cancel()
                }
                .disabled(!model.canCancel)

                ProgressView()
                    .controlSize(.small)
                    .opacity(model.isRunning ? 1 : 0)
            }

            Text(model.status)
                .font(.caption)
                .foregroundColor(.secondary)

            if let error = model.errorMessage {
                Text(error)
                    .foregroundColor(.red)
            }

            ScrollView {
                Text(model.output)
                    .frame(maxWidth: .infinity, alignment: .topLeading)
            }
            .frame(maxHeight: .infinity)
        }
        .padding(20)
        .fileImporter(
            isPresented: $model.isImporterPresented,
            allowedContentTypes: [.data]
        ) { result in
            model.selectModel(result)
        }
    }
}
