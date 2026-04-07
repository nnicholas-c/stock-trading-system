// AXIOM iOS — NotificationService.swift
// Handles push notification registration (APNs) + local buy/sell alerts.

import UserNotifications
import UIKit

final class NotificationService: NSObject, UNUserNotificationCenterDelegate {
    static let shared = NotificationService()

    // ── Request permission on first launch ───────────────────────────────────
    func requestPermission() async -> Bool {
        let center = UNUserNotificationCenter.current()
        do {
            return try await center.requestAuthorization(options: [.alert, .sound, .badge])
        } catch { return false }
    }

    // ── Register for APNs (call from AppDelegate.didFinishLaunching) ─────────
    func registerForAPNs() {
        DispatchQueue.main.async {
            UIApplication.shared.registerForRemoteNotifications()
        }
    }

    // ── Send APNs token to your backend ─────────────────────────────────────
    func sendTokenToBackend(_ tokenData: Data) async {
        let token = tokenData.map { String(format: "%02.2hhx", $0) }.joined()
        guard let url = URL(string: "\(APIConfig.baseURL)/notifications/register") else { return }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try? JSONEncoder().encode(["apns_token": token, "platform": "ios"])
        _ = try? await URLSession.shared.data(for: req)
    }

    // ── Schedule a LOCAL alert (for OpenClaw-style background checking) ──────
    func scheduleSignalAlert(ticker: String, signal: String, confidence: Double, price: Double) {
        let content = UNMutableNotificationContent()
        content.title = "🚨 AXIOM Signal — \(ticker)"
        content.body  = "\(signal) @ $\(String(format:"%.2f",price)) · \(Int(confidence*100))% confidence"
        content.sound = .defaultCritical
        content.categoryIdentifier = "TRADING_SIGNAL"

        // Fire immediately
        let trigger = UNTimeIntervalNotificationTrigger(timeInterval: 1, repeats: false)
        let req = UNNotificationRequest(identifier: "signal_\(ticker)_\(Date().timeIntervalSince1970)",
                                        content: content, trigger: trigger)
        UNUserNotificationCenter.current().add(req)
    }

    // ── Schedule daily pre-market brief (6:30 AM local) ─────────────────────
    func schedulePremarketBrief() {
        var components = DateComponents()
        components.hour   = 6
        components.minute = 30

        let content        = UNMutableNotificationContent()
        content.title      = "🌅 AXIOM Pre-Market Brief"
        content.body       = "Open the app to see today's signals, intraday predictions, and market news."
        content.sound      = .default
        content.userInfo   = ["action": "open_premarket"]

        let trigger = UNCalendarNotificationTrigger(dateMatching: components, repeats: true)
        let req = UNNotificationRequest(identifier: "daily_premarket", content: content, trigger: trigger)
        UNUserNotificationCenter.current().add(req)
    }

    // ── Category registration (for notification actions) ─────────────────────
    func registerCategories() {
        let viewAction = UNNotificationAction(identifier: "VIEW_SIGNAL",
                                              title: "View Signal", options: .foreground)
        let dismissAction = UNNotificationAction(identifier: "DISMISS", title: "Dismiss", options: .destructive)
        let category = UNNotificationCategory(identifier: "TRADING_SIGNAL",
                                               actions: [viewAction, dismissAction],
                                               intentIdentifiers: [], options: [])
        UNUserNotificationCenter.current().setNotificationCategories([category])
    }
}
