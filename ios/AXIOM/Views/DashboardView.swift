// AXIOM iOS — DashboardView.swift
// Main tab view with signal cards, predictions, and news.

import SwiftUI
import Charts  // Swift Charts (iOS 16+)

// ── App Entry ─────────────────────────────────────────────────────────────────
@main
struct AXIOMApp: App {
    @StateObject private var store = AppStore()
    var body: some Scene {
        WindowGroup {
            ContentView().environmentObject(store)
        }
    }
}

// ── App Store (ObservableObject) ──────────────────────────────────────────────
@MainActor
final class AppStore: ObservableObject {
    @Published var signals: [String: SignalData] = [:]
    @Published var intraday: [String: IntradayPrediction] = [:]
    @Published var weekly: [String: WeeklyPrediction] = [:]
    @Published var news: [String: NewsResponse] = [:]
    @Published var backtest: [String: BacktestResult] = [:]
    @Published var isLoading = false
    @Published var error: String? = nil

    let tickers = ["PLTR", "AAPL", "NVDA", "TSLA"]

    func loadAll() async {
        isLoading = true
        async let sig = APIService.shared.allSignals()
        do {
            let all = try await sig
            signals = all.signals
        } catch { self.error = error.localizedDescription }
        isLoading = false
    }

    func loadIntraday() async {
        for ticker in tickers {
            if let p = try? await APIService.shared.intradayPrediction(for: ticker) {
                intraday[ticker] = p
            }
        }
    }

    func loadWeekly() async {
        for ticker in tickers {
            if let w = try? await APIService.shared.weeklyPrediction(for: ticker) {
                weekly[ticker] = w
            }
        }
    }

    func loadNews() async {
        for ticker in tickers {
            if let n = try? await APIService.shared.news(for: ticker) {
                news[ticker] = n
            }
        }
    }
}

// ── Root Navigation ───────────────────────────────────────────────────────────
struct ContentView: View {
    @EnvironmentObject var store: AppStore
    var body: some View {
        TabView {
            DashboardView()
                .tabItem { Label("Dashboard", systemImage: "chart.xyaxis.line") }
            PredictionsView()
                .tabItem { Label("Predict", systemImage: "brain") }
            NewsView()
                .tabItem { Label("News", systemImage: "newspaper") }
            BacktestView()
                .tabItem { Label("Backtest", systemImage: "flask") }
        }
        .preferredColorScheme(.dark)
        .task { await store.loadAll(); await store.loadIntraday() }
    }
}

// ── Dashboard ─────────────────────────────────────────────────────────────────
struct DashboardView: View {
    @EnvironmentObject var store: AppStore
    @State private var selectedTicker = "PLTR"

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 16) {
                    // KPI strip
                    KPIStripView()
                        .padding(.horizontal)

                    // Ticker selector
                    TickerSelectorView(selected: $selectedTicker)
                        .padding(.horizontal)

                    // Signal card
                    if let sig = store.signals[selectedTicker] {
                        SignalCardView(signal: sig)
                            .padding(.horizontal)

                        // LSTM Chart
                        LSTMChartView(signal: sig)
                            .padding(.horizontal)
                    }

                    // All positions grid
                    AllPositionsGrid()
                        .padding(.horizontal)
                }
                .padding(.vertical)
            }
            .navigationTitle("AXIOM")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button(action: { Task { await store.loadAll() } }) {
                        Image(systemName: "arrow.clockwise")
                    }
                }
                ToolbarItem(placement: .topBarLeading) {
                    HStack(spacing: 4) {
                        Circle().fill(.green).frame(width:6,height:6)
                        Text("LIVE").font(.caption2).foregroundStyle(.green)
                    }
                }
            }
            .background(Color(hex: "060A10"))
        }
    }
}

// ── KPI Strip ─────────────────────────────────────────────────────────────────
struct KPIStripView: View {
    let kpis: [(String, String, Color)] = [
        ("Avg Alpha", "+73.9%", Color(hex: "00E5A0")),
        ("Best Sharpe", "2.17",   Color(hex: "2979FF")),
        ("Models", "5",           Color(hex: "AA44FF")),
        ("Features", "65",        Color(hex: "FFB300")),
    ]
    var body: some View {
        HStack(spacing: 10) {
            ForEach(kpis, id: \.0) { kpi in
                VStack(alignment: .leading, spacing: 2) {
                    Text(kpi.0).font(.caption2).foregroundStyle(.secondary)
                    Text(kpi.1).font(.headline).fontWeight(.bold).foregroundStyle(kpi.2)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(10)
                .background(Color(hex: "0F1623"))
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color(hex: "1E2D45"), lineWidth: 1))
            }
        }
    }
}

// ── Ticker Selector ───────────────────────────────────────────────────────────
struct TickerSelectorView: View {
    @Binding var selected: String
    let tickers = ["PLTR","AAPL","NVDA","TSLA"]
    var body: some View {
        HStack(spacing: 8) {
            ForEach(tickers, id: \.self) { t in
                Button(t) { selected = t }
                    .font(.caption).fontWeight(.semibold).fontDesign(.monospaced)
                    .padding(.horizontal, 12).padding(.vertical, 6)
                    .background(selected == t ? Color(hex: "1E3A5F") : Color(hex: "0F1623"))
                    .foregroundStyle(selected == t ? Color(hex: "2979FF") : .secondary)
                    .clipShape(RoundedRectangle(cornerRadius: 6))
                    .overlay(RoundedRectangle(cornerRadius: 6)
                        .stroke(selected == t ? Color(hex: "2979FF").opacity(0.5) : Color(hex: "1E2D45"), lineWidth: 1))
            }
        }
    }
}

// ── Signal Card ───────────────────────────────────────────────────────────────
struct SignalCardView: View {
    let signal: SignalData
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text(signal.ticker).font(.title2).fontWeight(.bold).fontDesign(.monospaced)
                    Text(signal.sector).font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                VStack(alignment: .trailing, spacing: 2) {
                    HStack(spacing: 4) {
                        Image(systemName: signal.signalIcon).foregroundStyle(signal.signalColor)
                        Text(signal.signal).font(.caption).fontWeight(.bold).foregroundStyle(signal.signalColor)
                    }
                    Text(String(format: "%.0f%% conf", signal.confidence * 100))
                        .font(.caption2).foregroundStyle(.secondary)
                }
            }

            HStack(alignment: .bottom, spacing: 4) {
                Text(String(format: "$%.2f", signal.price))
                    .font(.system(size: 32, weight: .bold, design: .monospaced))
                Spacer()
                VStack(alignment: .trailing, spacing: 2) {
                    Text("Target: \(String(format: "$%.2f", signal.analystTarget))")
                        .font(.caption).foregroundStyle(Color(hex: "00E5A0"))
                    Text(String(format: "+%.1f%% upside", signal.analystUpside))
                        .font(.caption2).foregroundStyle(.secondary)
                }
            }

            // Metrics grid
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible()), GridItem(.flexible())], spacing: 8) {
                MetricCell(label: "LSTM W+4", value: String(format: "$%.0f", signal.lstmForecast.last ?? 0),
                           sub: String(format: "%+.1f%%", signal.lstm4wPct),
                           color: signal.lstm4wPct >= 0 ? Color(hex:"00E5A0") : Color(hex:"FF4560"))
                MetricCell(label: "LGB 4w Est", value: String(format: "%+.1f%%", signal.lgbPct),
                           sub: "Return est.", color: signal.lgbPct >= 0 ? Color(hex:"00E5A0") : Color(hex:"FF4560"))
                MetricCell(label: "Vol Regime", value: signal.volRegime.replacingOccurrences(of:"_VOL",""),
                           sub: String(format: "%.0f%% ann", signal.currentVol*100),
                           color: signal.volRegime=="HIGH_VOL" ? Color(hex:"FF4560") : signal.volRegime=="LOW_VOL" ? Color(hex:"00E5A0") : Color(hex:"FFB300"))
                MetricCell(label: "RSI 14", value: String(format: "%.1f", signal.rsi14),
                           sub: signal.rsi14 > 70 ? "Overbought" : signal.rsi14 < 30 ? "Oversold" : "Neutral",
                           color: .secondary)
                MetricCell(label: "P/E", value: String(format: "%.0fx", signal.pe), sub: "TTM", color: .secondary)
                MetricCell(label: "Rev Growth", value: String(format: "%+.0f%%", signal.revGrowth),
                           sub: "YoY", color: signal.revGrowth > 0 ? Color(hex:"00E5A0") : Color(hex:"FF4560"))
            }

            // Confidence bar
            VStack(alignment: .leading, spacing: 4) {
                Text("Model Confidence").font(.caption2).foregroundStyle(.secondary)
                GeometryReader { g in
                    ZStack(alignment: .leading) {
                        RoundedRectangle(cornerRadius: 2).fill(Color(hex:"1E2D45")).frame(height:4)
                        RoundedRectangle(cornerRadius: 2).fill(signal.signalColor)
                            .frame(width: g.size.width * signal.confidence, height: 4)
                    }
                }.frame(height: 4)
            }
        }
        .padding(16)
        .background(Color(hex: "0F1623"))
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color(hex: "1E2D45"), lineWidth: 1))
        .overlay(alignment: .top) {
            RoundedRectangle(cornerRadius: 2).fill(signal.signalColor).frame(height: 2)
                .frame(maxWidth: .infinity).offset(y: 0).clipShape(RoundedRectangle(cornerRadius: 12))
        }
    }
}

struct MetricCell: View {
    let label: String; let value: String; let sub: String; let color: Color
    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(label).font(.caption2).foregroundStyle(.secondary)
            Text(value).font(.caption).fontWeight(.semibold).fontDesign(.monospaced).foregroundStyle(color)
            Text(sub).font(.system(size: 9)).foregroundStyle(.secondary)
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(hex: "060A10"))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }
}

// ── LSTM Chart (Swift Charts) ─────────────────────────────────────────────────
struct PricePoint: Identifiable {
    let id = UUID(); let week: Int; let price: Double; let isForecast: Bool
}

struct LSTMChartView: View {
    let signal: SignalData

    var historyPoints: [PricePoint] {
        // Simulated - in production pull from /signals history endpoint
        [-8,-7,-6,-5,-4,-3,-2,-1,0].enumerated().map { i, w in
            PricePoint(week: w, price: signal.price * (1 - Double(8-i)*0.015 + Double.random(in:-0.01...0.01)), isForecast: false)
        }
    }
    var forecastPoints: [PricePoint] {
        [(0, signal.price)] + signal.lstmForecast.enumerated().map { i, p in
            (i+1, p)
        }.map { PricePoint(week: $0.0, price: $0.1, isForecast: true) }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Price + LSTM Forecast").font(.caption).fontWeight(.semibold).foregroundStyle(.secondary)
                Spacer()
                HStack(spacing: 8) {
                    Legend(color: .white, label: "History")
                    Legend(color: Color(hex: "AA44FF"), label: "LSTM 4w")
                    Legend(color: Color(hex: "00E5A0"), label: "Analyst")
                }
            }

            Chart {
                ForEach(historyPoints) { p in
                    LineMark(x: .value("Week", p.week), y: .value("Price", p.price))
                        .foregroundStyle(.white).interpolationMethod(.catmullRom)
                }
                ForEach(forecastPoints) { p in
                    LineMark(x: .value("Week", p.week), y: .value("Price", p.price))
                        .foregroundStyle(Color(hex: "AA44FF")).lineStyle(.init(dash: [5,3]))
                        .interpolationMethod(.catmullRom)
                }
                RuleMark(y: .value("Target", signal.analystTarget))
                    .foregroundStyle(Color(hex: "00E5A0").opacity(0.6))
                    .lineStyle(.init(dash: [3,5]))
                    .annotation(position: .trailing) {
                        Text("$\(Int(signal.analystTarget))").font(.system(size: 8)).foregroundStyle(Color(hex: "00E5A0"))
                    }
            }
            .frame(height: 180)
            .chartXAxis {
                AxisMarks { v in
                    AxisValueLabel { Text("W\(v.as(Int.self) ?? 0)").font(.caption2).foregroundStyle(.secondary) }
                    AxisGridLine().foregroundStyle(Color(hex: "1E2D45"))
                }
            }
            .chartYAxis {
                AxisMarks { v in
                    AxisValueLabel { Text("$\(Int(v.as(Double.self) ?? 0))").font(.caption2).foregroundStyle(.secondary) }
                    AxisGridLine().foregroundStyle(Color(hex: "1E2D45"))
                }
            }
        }
        .padding(16)
        .background(Color(hex: "0F1623"))
        .clipShape(RoundedRectangle(cornerRadius: 12))
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color(hex: "1E2D45"), lineWidth: 1))
    }
}

struct Legend: View {
    let color: Color; let label: String
    var body: some View {
        HStack(spacing: 3) {
            RoundedRectangle(cornerRadius: 1).fill(color).frame(width:12, height:2)
            Text(label).font(.system(size:9)).foregroundStyle(.secondary)
        }
    }
}

// ── All Positions Grid ────────────────────────────────────────────────────────
struct AllPositionsGrid: View {
    @EnvironmentObject var store: AppStore
    let columns = [GridItem(.flexible()), GridItem(.flexible())]

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("ALL POSITIONS").font(.caption2).fontWeight(.semibold).foregroundStyle(.secondary)
            LazyVGrid(columns: columns, spacing: 10) {
                ForEach(Array(store.signals.values).sorted(by: {$0.ticker < $1.ticker})) { sig in
                    PositionMiniCard(signal: sig)
                }
            }
        }
    }
}

struct PositionMiniCard: View {
    let signal: SignalData
    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(signal.ticker).font(.caption).fontWeight(.bold).fontDesign(.monospaced)
                Spacer()
                Text(signal.signal).font(.system(size:8)).fontWeight(.bold).padding(3)
                    .background(signal.signalColor.opacity(0.15)).foregroundStyle(signal.signalColor)
                    .clipShape(RoundedRectangle(cornerRadius:3))
            }
            Text(String(format:"$%.2f", signal.price)).font(.title3).fontWeight(.bold).fontDesign(.monospaced)
            HStack {
                Text(String(format:"LSTM +%.0f%%", abs(signal.lstm4wPct))).font(.system(size:9))
                    .foregroundStyle(signal.lstm4wPct >= 0 ? Color(hex:"00E5A0") : Color(hex:"FF4560"))
                Spacer()
                Text(String(format:"%.0f%% conf", signal.confidence*100)).font(.system(size:9)).foregroundStyle(.secondary)
            }
        }
        .padding(10)
        .background(Color(hex: "0F1623"))
        .clipShape(RoundedRectangle(cornerRadius:10))
        .overlay(RoundedRectangle(cornerRadius:10).stroke(Color(hex:"1E2D45"), lineWidth:1))
        .overlay(alignment:.top) {
            RoundedRectangle(cornerRadius:2).fill(signal.signalColor).frame(height:2).frame(maxWidth:.infinity)
        }
    }
}

// ── Predictions View ──────────────────────────────────────────────────────────
struct PredictionsView: View {
    @EnvironmentObject var store: AppStore
    @State private var selected = "PLTR"

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing:16) {
                    TickerSelectorView(selected: $selected).padding(.horizontal)

                    if let intra = store.intraday[selected] {
                        IntradayCard(prediction: intra).padding(.horizontal)
                    }
                    if let weekly = store.weekly[selected] {
                        WeeklyCard(prediction: weekly).padding(.horizontal)
                    }
                }
                .padding(.vertical)
            }
            .navigationTitle("Predictions")
            .background(Color(hex:"060A10"))
            .task { await store.loadWeekly() }
        }
    }
}

struct IntradayCard: View {
    let prediction: IntradayPrediction
    var body: some View {
        VStack(alignment:.leading, spacing:12) {
            HStack {
                Label("TODAY'S DIRECTION", systemImage: "clock").font(.caption).fontWeight(.semibold).foregroundStyle(.secondary)
                Spacer()
                HStack(spacing:4) {
                    Image(systemName: prediction.directionIcon).foregroundStyle(prediction.directionColor)
                    Text(prediction.direction).font(.subheadline).fontWeight(.bold).foregroundStyle(prediction.directionColor)
                }
            }
            Text(String(format: "%.0f%% confidence", prediction.confidence*100)).font(.caption).foregroundStyle(.secondary)
            Divider().background(Color(hex:"1E2D45"))
            Text("Range: $\(String(format:"%.2f",prediction.expectedRangeLo)) – $\(String(format:"%.2f",prediction.expectedRangeHi))").font(.caption).fontDesign(.monospaced)
            Text("Catalyst: \(prediction.catalyst)").font(.caption).foregroundStyle(.secondary).lineLimit(2)
            HStack(spacing:12) {
                Label(prediction.newsSentiment, systemImage: "newspaper").font(.caption2)
                Label(prediction.macdDirection, systemImage: "waveform.path.ecg").font(.caption2)
                Label("RSI \(String(format:"%.0f",prediction.rsi))", systemImage: "chart.bar").font(.caption2)
            }.foregroundStyle(.secondary)
        }
        .padding(16).background(Color(hex:"0F1623"))
        .clipShape(RoundedRectangle(cornerRadius:12))
        .overlay(RoundedRectangle(cornerRadius:12).stroke(Color(hex:"1E2D45"), lineWidth:1))
    }
}

struct WeeklyCard: View {
    let prediction: WeeklyPrediction
    var body: some View {
        VStack(alignment:.leading, spacing:12) {
            Label("4-WEEK LSTM FORECAST", systemImage: "brain").font(.caption).fontWeight(.semibold).foregroundStyle(.secondary)
            ForEach(prediction.weekTargets, id:\.week) { w in
                HStack {
                    Text("Week +\(w.week)").font(.caption).foregroundStyle(.secondary).frame(width:60,alignment:.leading)
                    Text(String(format:"$%.2f",w.price)).font(.caption).fontDesign(.monospaced).frame(width:80,alignment:.leading)
                    Text(String(format:"%+.1f%%",w.pctChg)).font(.caption).fontDesign(.monospaced)
                        .foregroundStyle(w.pctChg>=0 ? Color(hex:"00E5A0") : Color(hex:"FF4560"))
                    Spacer()
                    Image(systemName: w.direction=="UP" ? "arrow.up.right" : w.direction=="DOWN" ? "arrow.down.right" : "arrow.right")
                        .font(.caption2)
                        .foregroundStyle(w.direction=="UP" ? Color(hex:"00E5A0") : w.direction=="DOWN" ? Color(hex:"FF4560") : .secondary)
                }
            }
            Divider().background(Color(hex:"1E2D45"))
            HStack {
                Text("LGB Est:").font(.caption).foregroundStyle(.secondary)
                Text(String(format:"%+.1f%%",prediction.lgb4wEst)).font(.caption).fontDesign(.monospaced)
                    .foregroundStyle(prediction.lgb4wEst>=0 ? Color(hex:"00E5A0") : Color(hex:"FF4560"))
                Spacer()
                Text("Conviction: \(prediction.conviction)").font(.caption2).foregroundStyle(.secondary)
            }
        }
        .padding(16).background(Color(hex:"0F1623"))
        .clipShape(RoundedRectangle(cornerRadius:12))
        .overlay(RoundedRectangle(cornerRadius:12).stroke(Color(hex:"1E2D45"), lineWidth:1))
    }
}

// ── News View ─────────────────────────────────────────────────────────────────
struct NewsView: View {
    @EnvironmentObject var store: AppStore
    var body: some View {
        NavigationStack {
            List {
                ForEach(["PLTR","AAPL","NVDA","TSLA"], id:\.self) { ticker in
                    if let n = store.news[ticker] {
                        Section(header: Text(ticker).font(.caption).fontWeight(.bold).fontDesign(.monospaced)) {
                            ForEach(n.articles.prefix(3)) { art in
                                NewsRow(article: art)
                            }
                        }
                    }
                }
            }
            .navigationTitle("News Feed")
            .listStyle(.insetGrouped)
            .scrollContentBackground(.hidden)
            .background(Color(hex:"060A10"))
            .task { await store.loadNews() }
        }
    }
}

struct NewsRow: View {
    let article: NewsArticle
    var body: some View {
        VStack(alignment:.leading, spacing:4) {
            HStack(spacing:6) {
                Text(article.sentiment).font(.system(size:8)).fontWeight(.bold)
                    .padding(.horizontal,5).padding(.vertical,2)
                    .background(article.sentimentColor.opacity(0.15))
                    .foregroundStyle(article.sentimentColor).clipShape(Capsule())
                Text(article.impact).font(.system(size:8)).foregroundStyle(.secondary)
            }
            Text(article.headline).font(.caption).lineLimit(2)
        }.listRowBackground(Color(hex:"0F1623"))
    }
}

// ── Backtest View ─────────────────────────────────────────────────────────────
struct BacktestView: View {
    @EnvironmentObject var store: AppStore
    @State private var results: [BacktestResult] = []

    var body: some View {
        NavigationStack {
            List(results) { bt in
                BacktestRow(result: bt)
            }
            .navigationTitle("Backtest")
            .listStyle(.insetGrouped)
            .scrollContentBackground(.hidden)
            .background(Color(hex:"060A10"))
            .task {
                for ticker in ["PLTR","AAPL","NVDA","TSLA"] {
                    if let r = try? await APIService.shared.backtest(for: ticker) {
                        results.append(r)
                    }
                }
            }
        }
    }
}

struct BacktestRow: View {
    let result: BacktestResult
    var body: some View {
        VStack(alignment:.leading, spacing:8) {
            Text(result.ticker).font(.subheadline).fontWeight(.bold).fontDesign(.monospaced)
            HStack(spacing:16) {
                StatPill(label:"Strategy",value:String(format:"+%.0f%%",result.strategyReturn*100),color:Color(hex:"00E5A0"))
                StatPill(label:"B&H",value:String(format:"+%.0f%%",result.bahReturn*100),color:.secondary)
                StatPill(label:"Alpha",value:String(format:"+%.0f%%",result.alpha*100),color:Color(hex:"AA44FF"))
            }
            HStack(spacing:16) {
                StatPill(label:"Sharpe",value:String(format:"%.2f",result.sharpe),color:Color(hex:"2979FF"))
                StatPill(label:"MaxDD",value:String(format:"%.0f%%",result.maxDrawdown*100),color:Color(hex:"FF4560"))
                StatPill(label:"Win%",value:String(format:"%.0f%%",result.winRate*100),color:Color(hex:"FFB300"))
            }
        }.listRowBackground(Color(hex:"0F1623"))
    }
}

struct StatPill: View {
    let label:String; let value:String; let color:Color
    var body: some View {
        VStack(spacing:1) {
            Text(value).font(.caption).fontWeight(.semibold).fontDesign(.monospaced).foregroundStyle(color)
            Text(label).font(.system(size:8)).foregroundStyle(.secondary)
        }
    }
}
