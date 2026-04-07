// AXIOM iOS — DataModels.swift
// All Codable structs matching the FastAPI response schemas.

import Foundation
import SwiftUI

// ── Signal ────────────────────────────────────────────────────────────────────
struct AllSignalsResponse: Codable {
    let generatedAt: String
    let signals: [String: SignalData]
}

struct SignalData: Codable, Identifiable {
    var id: String { ticker }
    let ticker: String
    let price: Double
    let signal: String          // "STRONG BUY" / "BUY" / "HOLD" / "SELL"
    let signalInt: Int
    let confidence: Double
    let color: String
    let analystTarget: Double
    let analystUpside: Double
    let bullPct: Double
    let lgbFwdRet: Double
    let lstmForecast: [Double]
    let volRegime: String
    let currentVol: Double
    let riskScore: Double
    let rsi14: Double
    let macdHist: Double
    let trendScore: Double
    let topFeatures: [[AnyCodable]]
    let sector: String
    let marketCap: Double
    let pe: Double
    let revGrowth: Double
    let grossMargin: Double
    let netMargin: Double
    let fcf: Double
    let r40: Double

    // Computed
    var signalColor: Color {
        switch signal {
        case "STRONG BUY": return Color(hex: "AA44FF")
        case "BUY":        return Color(hex: "00E5A0")
        case "HOLD":       return Color(hex: "FFB300")
        case "SELL":       return Color(hex: "FF4560")
        default:           return .gray
        }
    }
    var signalIcon: String {
        switch signal {
        case "STRONG BUY": return "arrow.up.forward.circle.fill"
        case "BUY":        return "arrow.up.circle.fill"
        case "HOLD":       return "equal.circle.fill"
        case "SELL":       return "arrow.down.circle.fill"
        default:           return "questionmark.circle"
        }
    }
    var lstm4wPct: Double {
        guard let last = lstmForecast.last else { return 0 }
        return (last / price - 1) * 100
    }
    var lgbPct: Double { lgbFwdRet * 100 }
    var marketCapFormatted: String {
        marketCap >= 1e12 ? String(format: "$%.1fT", marketCap/1e12)
                          : String(format: "$%.0fB", marketCap/1e9)
    }
}

struct SignalSummary: Codable {
    let ticker: String
    let signal: String
    let confidence: String
    let price: String
    let target: String
    let upside: String
    let lstm4w: String
    let lgbEst: String
    let regime: String
    let risk: String
    let alertText: String
}

// ── Predictions ───────────────────────────────────────────────────────────────
struct IntradayPrediction: Codable {
    let ticker: String
    let generatedAt: String
    let direction: String   // UP / DOWN / FLAT
    let confidence: Double
    let expectedRangeLo: Double
    let expectedRangeHi: Double
    let catalyst: String
    let newsSentiment: String
    let mlSignal: String
    let macdDirection: String
    let rsi: Double
    let volRegime: String
    let score: Double

    var directionColor: Color {
        direction == "UP" ? Color(hex: "00E5A0") : direction == "DOWN" ? Color(hex: "FF4560") : Color(hex: "FFB300")
    }
    var directionIcon: String {
        direction == "UP" ? "arrow.up.right" : direction == "DOWN" ? "arrow.down.right" : "arrow.right"
    }
}

struct WeekTarget: Codable {
    let week: Int
    let price: Double
    let pctChg: Double
    let direction: String
}

struct WeeklyPrediction: Codable {
    let ticker: String
    let generatedAt: String
    let currentPrice: Double
    let weekTargets: [WeekTarget]
    let lgb4wEst: Double
    let modelSignal: String
    let conviction: String
    let analystTarget: Double
    let analystUpside: Double
}

struct Scenario: Codable {
    let price: Double
    let pct: Double
    let catalyst: String
    let probability: Double
}

struct ScenariosResponse: Codable {
    let ticker: String
    let currentPrice: Double
    let horizon: String
    let scenarios: [String: Scenario]  // "bull", "base", "bear"
}

// ── News ──────────────────────────────────────────────────────────────────────
struct NewsArticle: Codable, Identifiable {
    var id: String { headline }
    let ticker: String
    let headline: String
    let sentiment: String
    let impact: String
    let source: String
    let url: String
    let published: String
    let netScore: Int

    var sentimentColor: Color {
        sentiment == "BULLISH" ? Color(hex: "00E5A0") : sentiment == "BEARISH" ? Color(hex: "FF4560") : Color(hex: "8BA3C7")
    }
}

struct NewsResponse: Codable {
    let ticker: String
    let generatedAt: String
    let overallSentiment: String
    let articles: [NewsArticle]
    let materialEvents: Int
    let intradayImpact: String
    let cached: Bool
}

struct PremarketBrief: Codable {
    let ticker: String
    let sentiment: String
    let intradayEst: String
    let material: Int
    let topStories: [PremarketStory]
}
struct PremarketStory: Codable {
    let headline: String
    let sentiment: String
    let impact: String
}

// ── Backtest ──────────────────────────────────────────────────────────────────
struct BacktestResult: Codable, Identifiable {
    var id: String { ticker }
    let ticker: String
    let startDate: String
    let endDate: String
    let strategyReturn: Double
    let bahReturn: Double
    let alpha: Double
    let sharpe: Double
    let sortino: Double
    let maxDrawdown: Double
    let nTrades: Int
    let winRate: Double
    let rlReturn: Double
    let rlSharpe: Double
    let rlMaxDd: Double
}

// ── Health ────────────────────────────────────────────────────────────────────
struct HealthResponse: Codable {
    let status: String
    let modelsLoaded: Bool
    let lastUpdated: String
    let uptimeS: Double
    let tickers: [String]
}

// ── Helpers ───────────────────────────────────────────────────────────────────
struct AnyCodable: Codable {
    let value: Any
    init(_ value: Any) { self.value = value }
    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if let v = try? c.decode(Double.self) { value = v }
        else if let v = try? c.decode(String.self) { value = v }
        else if let v = try? c.decode(Int.self) { value = v }
        else { value = "" }
    }
    func encode(to encoder: Encoder) throws {
        var c = encoder.singleValueContainer()
        if let v = value as? Double { try c.encode(v) }
        else if let v = value as? String { try c.encode(v) }
        else if let v = value as? Int { try c.encode(v) }
    }
}

extension Color {
    init(hex: String) {
        let h = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var int: UInt64 = 0
        Scanner(string: h).scanHexInt64(&int)
        let r, g, b: UInt64
        switch h.count {
        case 6: (r,g,b) = (int>>16, int>>8&0xFF, int&0xFF)
        default:(r,g,b) = (0,0,0)
        }
        self.init(red: Double(r)/255, green: Double(g)/255, blue: Double(b)/255)
    }
}
