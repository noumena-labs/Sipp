import SwiftUI

@main
@MainActor
struct SippIOSApp: App {
    @StateObject private var model = SippViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView(model: model)
        }
    }
}
