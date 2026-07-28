import SwiftUI

@main
@MainActor
struct SippSandboxApp: App {
    @StateObject private var model = SippViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView(model: model)
                .frame(minWidth: 720, minHeight: 560)
        }
    }
}
