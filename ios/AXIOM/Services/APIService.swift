// AXIOM iOS — APIService.swift
// Central networking layer. Async/await + Combine.
// Replace BASE_URL with your Railway/Render deployment URL.

import Foundation
import Combine

// ── Base URL ─────────────────────────────────────────────────────────────────
enum APIConfig {
    static let baseURL = "https://YOUR-RAILWAY-APP.railway.app"
    // For local dev: static let baseURL = "http://localhost:8000"
}

// ── Errors ────────────────────────────────────────────────────────────────────
enum APIError: LocalizedError {
    case invalidURL, decodingError(Error), networkError(Error), notFound
    var errorDescription: String? {
        switch self {
        case .invalidURL:          return "Invalid URL"
        case .decodingError(let e): return "Decode error: \(e.localizedDescription)"
        case .networkError(let e): return "Network error: \(e.localizedDescription)"
        case .notFound:            return "Resource not found"
        }
    }
}

// ── Service ───────────────────────────────────────────────────────────────────
final class APIService {
    static let shared = APIService()
    private let session = URLSession.shared
    private let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        d.dateDecodingStrategy = .iso8601
        return d
    }()

    // Generic fetch
    private func fetch<T: Decodable>(_ endpoint: String) async throws -> T {
        guard let url = URL(string: APIConfig.baseURL + endpoint) else {
            throw APIError.invalidURL
        }
        do {
            let (data, resp) = try await session.data(from: url)
            if (resp as? HTTPURLResponse)?.statusCode == 404 { throw APIError.notFound }
            return try decoder.decode(T.self, from: data)
        } catch let e as APIError { throw e }
        catch let e as DecodingError { throw APIError.decodingError(e) }
        catch { throw APIError.networkError(error) }
    }

    // ── Signals ───────────────────────────────────────────────────────────────
    func allSignals() async throws -> AllSignalsResponse {
        try await fetch("/signals/")
    }
    func signal(for ticker: String) async throws -> SignalData {
        try await fetch("/signals/\(ticker)")
    }
    func signalSummary(for ticker: String) async throws -> SignalSummary {
        try await fetch("/signals/\(ticker)/summary")
    }

    // ── Predictions ───────────────────────────────────────────────────────────
    func intradayPrediction(for ticker: String) async throws -> IntradayPrediction {
        try await fetch("/predict/\(ticker)/intraday")
    }
    func weeklyPrediction(for ticker: String) async throws -> WeeklyPrediction {
        try await fetch("/predict/\(ticker)/weekly")
    }
    func scenarios(for ticker: String) async throws -> ScenariosResponse {
        try await fetch("/predict/\(ticker)/scenarios")
    }

    // ── News ──────────────────────────────────────────────────────────────────
    func news(for ticker: String) async throws -> NewsResponse {
        try await fetch("/news/\(ticker)")
    }
    func premarketBrief(for ticker: String) async throws -> PremarketBrief {
        try await fetch("/news/\(ticker)/premarket")
    }

    // ── Backtest ──────────────────────────────────────────────────────────────
    func backtest(for ticker: String) async throws -> BacktestResult {
        try await fetch("/backtest/\(ticker)")
    }

    // ── Health ────────────────────────────────────────────────────────────────
    func health() async throws -> HealthResponse {
        try await fetch("/health")
    }
}
