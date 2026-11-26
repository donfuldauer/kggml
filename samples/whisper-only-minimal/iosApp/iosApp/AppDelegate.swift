import UIKit
import WhisperOnlyMinimal

/// Host app of `:samples:whisper-only-minimal`: the whole UI is the Kotlin framework's view controller.
@main
final class AppDelegate: UIResponder, UIApplicationDelegate {
    var window: UIWindow?

    func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
    ) -> Bool {
        let window = UIWindow(frame: UIScreen.main.bounds)
        window.rootViewController = IosEntryPointKt.MainViewController()
        window.makeKeyAndVisible()
        self.window = window
        return true
    }
}
