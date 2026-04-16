(function () {
  "use strict";

  const COLORS = { PLTR: "#b06cfc", AAPL: "#4a9eff", NVDA: "#00d47e", TSLA: "#f7b731" };
  const TICKERS = ["PLTR", "AAPL", "NVDA", "TSLA"];
  const TIME_LABELS = { "1w": "1W", "1m": "1M", "3m": "3M", "6m": "6M", "1y": "1Y", all: "All" };

  const state = {
    bundle: null,
    cur: "PLTR",
    curTime: "1m",
    curPage: "terminal",
    inds: { EMA: true, MA: false, BB: false, VOL: true, PRED: true },
    chartRO: null,
    npActiveFilter: "ALL",
    npActiveId: null,
    renderedStaticPages: new Set(),
  };

  function fmtPct(value, digits = 2) {
    const num = Number(value || 0);
    return `${num >= 0 ? "+" : ""}${num.toFixed(digits)}%`;
  }

  function fmtPrice(value) {
    const num = Number(value || 0);
    return `$${num.toFixed(2)}`;
  }

  function fmtCompact(value) {
    const num = Number(value || 0);
    if (Math.abs(num) >= 1e9) return `${(num / 1e9).toFixed(2)}B`;
    if (Math.abs(num) >= 1e6) return `${(num / 1e6).toFixed(2)}M`;
    if (Math.abs(num) >= 1e3) return `${(num / 1e3).toFixed(1)}K`;
    return `${Math.round(num)}`;
  }

  function num(value, fallback = 0) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  }

  function fmtAgeFromIso(value) {
    if (!value) return "now";
    const then = new Date(value);
    if (Number.isNaN(then.getTime())) return "now";
    const diffH = Math.max(0, (Date.now() - then.getTime()) / 36e5);
    if (diffH < 1) return `${Math.max(1, Math.round(diffH * 60))}m`;
    if (diffH < 24) return `${Math.round(diffH)}h`;
    return `${Math.round(diffH / 24)}d`;
  }

  function statusClass(label) {
    const value = String(label || "").toUpperCase();
    if (value.includes("EXPERIMENTAL") || value.includes("STALE")) return "warn";
    if (value.includes("FALLBACK") || value.includes("CONFLICT")) return "bad";
    if (value.includes("LIVE") || value.includes("CHAMPION")) return "good";
    return "info";
  }

  function signalClass(signal) {
    const value = String(signal || "HOLD").toUpperCase();
    if (value === "BUY") return "buy";
    if (value === "SELL") return "sell";
    return "hold";
  }

  function toneClass(tone) {
    const value = String(tone || "").toLowerCase();
    if (value === "bull" || value === "buy" || value === "good") return "up";
    if (value === "bear" || value === "sell" || value === "risk" || value === "bad") return "dn";
    return "neu";
  }

  function currentPayload() {
    return state.bundle?.tickers?.[state.cur] || null;
  }

  function oneDayEdge(payload) {
    return payload?.one_day_edge || payload?.horizons?.["1d"]?.edge_assessment || {
      status: "calibrated",
      label: "CALIBRATED 1D",
      tone: "good",
      summary: "The next-day setup is calibrated and can be treated like a normal tactical input.",
    };
  }

  function currentCandles() {
    const payload = currentPayload();
    return payload?.chart?.timeframes?.[state.curTime] || payload?.chart?.timeframes?.[payload?.chart?.default_timeframe || "1m"] || [];
  }

  function flattenNewsItems() {
    if (!state.bundle) return [];
    return TICKERS.flatMap((ticker) => {
      const payload = state.bundle.tickers[ticker];
      return (payload.news_monitor.items || []).map((item, idx) => ({
        id: `${ticker}-${idx}-${item.headline}`,
        ticker,
        dir: item.net_score >= 0 ? "bull" : "bear",
        impact: item.impact || "LOW",
        vader: Math.max(-1, Math.min(1, Number(item.net_score || 0) / 3)),
        age: fmtAgeFromIso(item.published_at_et),
        url: item.url || "",
        headline: item.headline,
        source: item.source || "Google News",
        news_cat: item.section || "Today",
        summary: item.reason || payload.news_monitor.summary || "",
        body: `${item.description || item.headline}\n\n${item.reason || payload.news_monitor.summary || ""}`.trim(),
        px_impact: fmtPct((Number(payload.signal.pred_return_pct || 0) + Number(item.net_score || 0) * 0.22), 1),
        published_at_et: item.published_at_et || "",
        horizons: [
          { l: "1D", v: fmtPct(payload.horizons["1d"].expected_return_pct, 1) },
          { l: "1W", v: fmtPct(payload.horizons["5d"].expected_return_pct, 1) },
          { l: "2W", v: fmtPct(payload.horizons["10d"].expected_return_pct, 1) },
        ],
      }));
    });
  }

  function updateClock() {
    const now = new Date(new Date().toLocaleString("en-US", { timeZone: "America/Los_Angeles" }));
    const el = document.getElementById("clock");
    if (!el) return;
    el.textContent = [now.getHours(), now.getMinutes(), now.getSeconds()].map((n) => String(n).padStart(2, "0")).join(":") + " PDT";
  }

  function renderNavStatus() {
    if (!state.bundle) return;
    const mode = state.bundle.pages_publish_mode || {};
    const active = currentPayload();
    const livePill = document.getElementById("nav-live-pill");
    const statusPill = document.getElementById("nav-status-pill");
    const regimePill = document.getElementById("nav-regime-pill");
    if (livePill) livePill.innerHTML = `<div class="dot-g"></div>${active?.data_freshness?.is_stale ? "STALE" : "LIVE"}`;
    if (statusPill) {
      statusPill.textContent = mode.label || "CHAMPION";
      statusPill.className = `pill ${mode.artifact_status === "experimental" ? "pill-r" : "pill-g"}`;
    }
    if (regimePill) {
      const regime = active?.market_context?.macro_regime || "NEUTRAL";
      regimePill.textContent = `${regime} REGIME`;
      regimePill.className = `pill ${regime === "BEAR" ? "pill-r" : regime === "BULL" ? "pill-g" : "pill-r"}`;
    }
  }

  function renderRibbon() {
    if (!state.bundle) return;
    const ribbon = document.getElementById("tickerRibbon");
    const macro = document.getElementById("macroStrip");
    if (ribbon) {
      ribbon.innerHTML = state.bundle.ticker_strip
        .map(
          (item) => `
          <div class="rib ${item.ticker === state.cur ? "on" : ""}" id="rib-${item.ticker}" onclick="selectTicker('${item.ticker}')">
            <span class="rsym" style="color:${COLORS[item.ticker]}">${item.ticker}</span>
            <span class="rpx">${fmtPrice(item.price)}</span>
            <span class="rchg ${item.change_pct >= 0 ? "up" : "dn"}">${fmtPct(item.change_pct)}</span>
          </div>
          <div class="rsep"></div>`
        )
        .join("");
    }
    if (macro) {
      macro.innerHTML = (state.bundle.macro_strip || [])
        .map(
          (item) => `<span>${item.display_symbol} <span class="v ${item.change_pct >= 0 ? "up" : "dn"}">${item.display_symbol === "VIX" ? item.price.toFixed(2) : fmtPrice(item.price)}</span> <span class="${item.change_pct >= 0 ? "up" : "dn"}">${fmtPct(item.change_pct, 1)}</span></span>`
        )
        .join("");
    }
  }

  function renderWatchMeta() {
    const payload = currentPayload();
    const el = document.getElementById("watchMeta");
    const badge = document.getElementById("watchlist-status");
    if (!payload || !el) return;
    const mode = state.bundle.pages_publish_mode || {};
    const edge = oneDayEdge(payload);
    if (badge) badge.textContent = `${mode.label || "Latest"} · ${payload.market_date}`;
    el.innerHTML = `
      <div style="display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap">
        <span class="status-chip ${statusClass(mode.label)}">${mode.label || "CHAMPION"}</span>
        <span class="status-chip ${payload.news_monitor.used_recent_fallback ? "bad" : "good"}">${payload.news_monitor.status_label}</span>
        <span class="status-chip ${edge.tone || statusClass(edge.label)}">${edge.label}</span>
      </div>
      <div class="watch-kpi">
        <div class="watch-kpi-card">
          <div class="watch-kpi-l">MARKET DATE</div>
          <div class="watch-kpi-v">${payload.market_date}</div>
        </div>
        <div class="watch-kpi-card">
          <div class="watch-kpi-l">FORECAST FOR</div>
          <div class="watch-kpi-v">${payload.forecast_for_date}</div>
        </div>
        <div class="watch-kpi-card">
          <div class="watch-kpi-l">TRUST</div>
          <div class="watch-kpi-v ${toneClass(payload.signal.signal)}">${Number(payload.signal.trust_score).toFixed(1)}%</div>
        </div>
        <div class="watch-kpi-card">
          <div class="watch-kpi-l">TREND</div>
          <div class="watch-kpi-v ${toneClass(payload.trend_snapshot.state === "BULLISH" ? "bull" : payload.trend_snapshot.state === "BEARISH" ? "bear" : "neutral")}">${payload.trend_snapshot.state}</div>
        </div>
      </div>
      <div style="margin-top:7px;font-size:8px;color:var(--t2);line-height:1.55">${edge.summary}</div>`;
  }

  function renderSigTable() {
    const el = document.getElementById("sigTable");
    if (!el || !state.bundle) return;
    el.innerHTML = TICKERS.map((ticker) => {
      const payload = state.bundle.tickers[ticker];
      const signal = payload.signal;
      const sc = signalClass(signal.signal);
      const edge = oneDayEdge(payload);
      return `<div class="srow ${sc}${ticker === state.cur ? " on" : ""}" onclick="selectTicker('${ticker}')">
        <div class="sr-row">
          <div><div class="ssym" style="color:${COLORS[ticker]}">${ticker}</div><div class="ssec">${payload.company_name} · ${edge.label}</div></div>
          <div style="text-align:right">
            <div class="spx">${fmtPrice(payload.quote_snapshot.close)}</div>
            <div class="schg ${payload.quote_snapshot.change_pct >= 0 ? "up" : "dn"}">${fmtPct(payload.quote_snapshot.change_pct)}</div>
            <div class="sbadge ${sc}">${signal.signal}</div>
          </div>
        </div>
        <div class="cstrip"><div class="cfill" style="width:${Math.round(Number(signal.trust_score))}%;background:${sc === "buy" ? "var(--G)" : sc === "sell" ? "var(--R)" : "var(--gold)"}"></div></div>
      </div>`;
    }).join("");
  }

  function updateChartHeader() {
    const payload = currentPayload();
    if (!payload) return;
    const quote = payload.quote_snapshot;
    const signal = payload.signal;
    const mode = state.bundle.pages_publish_mode || {};
    const sym = document.getElementById("ch-sym");
    const px = document.getElementById("ch-px");
    const chg = document.getElementById("ch-chg");
    const mainSig = document.getElementById("main-sig");
    const prob = document.getElementById("ch-conf");
    const trust = document.getElementById("ch-trust");
    const trend = document.getElementById("ch-trend");
    const fresh = document.getElementById("ch-fresh");
    const status = document.getElementById("ch-status");
    const edge = oneDayEdge(payload);
    if (sym) {
      sym.textContent = state.cur;
      sym.style.color = COLORS[state.cur];
    }
    if (px) px.textContent = fmtPrice(quote.close);
    if (chg) {
      chg.textContent = fmtPct(quote.change_pct);
      chg.className = `cchg ${quote.change_pct >= 0 ? "up" : "dn"}`;
    }
    if (mainSig) {
      const sc = signalClass(signal.signal);
      mainSig.textContent = signal.signal === "BUY" ? "▲ BUY" : signal.signal === "SELL" ? "▼ SELL" : "◆ HOLD";
      mainSig.className = `msig ${sc}`;
    }
    if (prob) prob.textContent = `${Number(signal.probability_up).toFixed(1)}%`;
    if (trust) {
      trust.textContent = `${Number(signal.trust_score).toFixed(1)}%`;
      trust.style.color = Number(signal.trust_score) >= 65 ? "var(--G)" : Number(signal.trust_score) >= 50 ? "var(--cyan)" : "var(--R)";
    }
    if (trend) {
      trend.textContent = `${payload.trend_snapshot.state} ${Number(payload.trend_snapshot.score).toFixed(2)}`;
      trend.style.color = payload.trend_snapshot.state === "BULLISH" ? "var(--G)" : payload.trend_snapshot.state === "BEARISH" ? "var(--R)" : "var(--cyan)";
    }
    if (status) {
      const label = payload.data_freshness.is_stale ? "STALE" : (mode.label || "CHAMPION");
      status.textContent = label;
      status.className = `status-chip ${statusClass(label)}`;
    }
    if (fresh) {
      const flags = [];
      flags.push(`mkt ${payload.market_date}`);
      flags.push(`fcst ${payload.forecast_for_date}`);
      flags.push(edge.label.toLowerCase());
      if (payload.news_monitor.used_recent_fallback) flags.push("fallback news");
      if (payload.data_freshness.is_stale) flags.push("stale quote");
      fresh.textContent = flags.join(" · ");
    }
  }

  function updatePredictionCard() {
    const payload = currentPayload();
    if (!payload) return;
    const overlay = payload.forecast_overlay;
    const edge = oneDayEdge(payload);
    const one = overlay.points.find((p) => p.key === "1d") || overlay.points[0];
    const five = overlay.points.find((p) => p.key === "5d") || overlay.points[1];
    const ten = overlay.points.find((p) => p.key === "10d") || overlay.points[2];
    const newsBias = Number(payload.news_monitor.net_score || 0) * 0.45;
    const nextOpen = Number(payload.card.l4h || one.expected_return_pct * 0.5);
    const month = Number(payload.card.l20d || ten.expected_return_pct * 1.8);
    const cells = {
      "1h": newsBias,
      "4h": nextOpen,
      "1d": Number(one.expected_return_pct || 0),
      "5d": Number(five.expected_return_pct || 0),
      "10d": Number(ten.expected_return_pct || 0),
      "20d": month,
    };
    Object.entries(cells).forEach(([key, value]) => {
      const val = document.getElementById(`pred-${key}`);
      const px = document.getElementById(`pred-${key}-px`);
      const priceBase = Number(payload.quote_snapshot.close) * (1 + Number(value) / 100);
      if (val) {
        val.textContent = fmtPct(value);
        val.style.color = Number(value) >= 0 ? "var(--G)" : "var(--R)";
      }
      if (px) px.textContent = fmtPrice(priceBase);
    });

    const ml = Number(payload.signal.probability_up);
    const newsScaled = Math.max(0, Math.min(100, 50 + Number(payload.news_monitor.net_score || 0) * 50));
    const trendScaled = Math.max(0, Math.min(100, 50 + Number(payload.trend_snapshot.score || 0) * 50));
    const composite = (ml * 0.5) + (Number(payload.signal.trust_score) * 0.3) + (trendScaled * 0.2);
    const zfSell = document.getElementById("zfSell");
    const zfBuy = document.getElementById("zfBuy");
    const zoneMarker = document.getElementById("zoneMarker");
    const zoneLabel = document.getElementById("zone-label");
    if (zfSell) zfSell.style.width = `${100 - composite}%`;
    if (zfBuy) zfBuy.style.width = `${composite}%`;
    if (zoneMarker) zoneMarker.style.left = `${composite}%`;
    if (zoneLabel) {
      zoneLabel.textContent = composite > 62 ? `BUY ZONE · ${payload.trend_snapshot.state}` : composite < 38 ? `SELL ZONE · ${payload.trend_snapshot.state}` : `NEUTRAL · ${payload.trend_snapshot.state}`;
      zoneLabel.style.color = composite > 62 ? "var(--G)" : composite < 38 ? "var(--R)" : "var(--t2)";
    }

    const sentMl = document.getElementById("sent-ml");
    const sentMlV = document.getElementById("sent-ml-v");
    const sentNews = document.getElementById("sent-news");
    const sentNewsV = document.getElementById("sent-news-v");
    const sentTrend = document.getElementById("sent-jpn");
    const sentTrendV = document.getElementById("sent-jpn-v");
    if (sentMl) sentMl.style.width = `${ml}%`;
    if (sentMlV) sentMlV.textContent = `${ml.toFixed(1)}%`;
    if (sentNews) {
      sentNews.style.width = `${newsScaled}%`;
      sentNews.style.background = Number(payload.news_monitor.net_score || 0) >= 0 ? "var(--G)" : "var(--R)";
    }
    if (sentNewsV) {
      sentNewsV.textContent = Number(payload.news_monitor.net_score || 0).toFixed(2);
      sentNewsV.style.color = Number(payload.news_monitor.net_score || 0) >= 0 ? "var(--G)" : "var(--R)";
    }
    if (sentTrend) {
      sentTrend.style.width = `${trendScaled}%`;
      sentTrend.style.background = payload.trend_snapshot.state === "BULLISH" ? "var(--G)" : payload.trend_snapshot.state === "BEARISH" ? "var(--R)" : "var(--cyan)";
    }
    if (sentTrendV) {
      sentTrendV.textContent = payload.trend_snapshot.state.slice(0, 4);
      sentTrendV.style.color = payload.trend_snapshot.state === "BULLISH" ? "var(--G)" : payload.trend_snapshot.state === "BEARISH" ? "var(--R)" : "var(--cyan)";
    }

    const title = document.getElementById("intel-title");
    const badge = document.getElementById("intel-badge");
    if (title) title.textContent = `Focus Intel · ${state.cur}`;
    if (badge) badge.textContent = `${state.bundle.artifact_version.toUpperCase()} · ${edge.label}`;
  }

  function renderTradeVisualizer() {
    const payload = currentPayload();
    const el = document.getElementById("tradeVisualizer");
    if (!payload || !el) return;

    const overlay = payload.forecast_overlay || {};
    const points = overlay.points || [];
    const one = points.find((point) => point.key === "1d") || points[0] || {};
    const ten = points.find((point) => point.key === "10d") || points[points.length - 1] || one;
    const quote = payload.quote_snapshot || {};
    const levels = payload.levels || {};
    const tech = payload.technical_snapshot || {};
    const market = payload.market_context || {};
    const perf = payload.recent_performance || {};
    const edge = oneDayEdge(payload);

    const current = num(quote.close);
    const support = num(levels.support_20d, current);
    const resistance = num(levels.resistance_20d, current);
    const ma20 = num(levels.ma20, current);
    const ma50 = num(levels.ma50, current);
    const oneTarget = num(one.target_price, num(payload.signal.target_price, current));
    const tenTarget = num(ten.target_price, current);
    const bandLow = num(one.lower_price, current);
    const bandHigh = num(one.upper_price, current);
    const downsidePct = current ? ((support - current) / current) * 100 : 0;
    const upsidePct = current ? ((resistance - current) / current) * 100 : 0;
    const rr = Math.abs(downsidePct) > 0.05 ? upsidePct / Math.abs(downsidePct) : null;
    const invalidation = Math.min(support || current, ma50 || current);
    const reclaim = Math.max(ma20 || current, current);
    const trendChip = payload.trend_snapshot.state === "BULLISH" ? "good" : payload.trend_snapshot.state === "BEARISH" ? "bad" : "info";
    const supportChip = payload.signal.trend_supported ? "good" : "bad";
    const edgeChip = edge.tone === "good" ? "good" : edge.tone === "bad" ? "bad" : "warn";
    const newsChip = payload.news_monitor.used_recent_fallback ? "warn" : "good";
    const signalChip = payload.signal.signal === "BUY" ? "good" : payload.signal.signal === "SELL" ? "bad" : "warn";
    const rangeMin = Math.min(support || current, bandLow, current);
    const rangeMax = Math.max(resistance || current, tenTarget, bandHigh, current);
    const pos = (value) => {
      if (!(rangeMax > rangeMin) || !Number.isFinite(value)) return 50;
      return Math.max(0, Math.min(100, ((value - rangeMin) / (rangeMax - rangeMin)) * 100));
    };
    const bandLeft = pos(Math.min(bandLow, bandHigh));
    const bandRight = pos(Math.max(bandLow, bandHigh));
    const bandWidth = Math.max(3, bandRight - bandLeft);
    const rewardText = rr === null ? "—" : `${rr.toFixed(2)}R`;
    const setupLabel = payload.signal.trend_supported ? "Trend aligned" : "Trend conflict";
    const summary = payload.signal.trend_supported
      ? `${payload.signal.summary} The daily trend is supporting the active direction, so the setup can be read more cleanly than the raw 1d edge alone.`
      : `${payload.signal.summary} The daily trend is pushing back against the next-day setup, so support, reclaim levels, and news quality matter more than the headline probability.`;

    el.innerHTML = `
      <div class="trade-viz-head">
        <span class="trade-viz-title">Trade Cockpit</span>
        <span class="status-chip ${supportChip}">${setupLabel}</span>
      </div>
      <div class="trade-chip-row">
        <span class="status-chip ${signalChip}">${payload.signal.signal}</span>
        <span class="status-chip ${edgeChip}">${edge.label}</span>
        <span class="status-chip ${trendChip}">${payload.trend_snapshot.state} ${num(payload.trend_snapshot.score).toFixed(2)}</span>
        <span class="status-chip ${newsChip}">${payload.news_monitor.status_label}</span>
        <span class="status-chip info">${market.macro_regime || "NEUTRAL"} REGIME</span>
      </div>
      <div class="trade-grid">
        <div class="trade-stat">
          <div class="trade-stat-l">LAST / CHANGE</div>
          <div class="trade-stat-v">${fmtPrice(current)}</div>
          <div class="trade-stat-s ${quote.change_pct >= 0 ? "up" : "dn"}">${fmtPct(quote.change_pct)} · ${quote.volume_label || fmtCompact(quote.volume)}</div>
        </div>
        <div class="trade-stat">
          <div class="trade-stat-l">1D TARGET</div>
          <div class="trade-stat-v ${num(one.expected_return_pct) >= 0 ? "up" : "dn"}">${fmtPrice(oneTarget)}</div>
          <div class="trade-stat-s">${fmtPct(num(one.expected_return_pct))} · ${num(one.probability_up).toFixed(1)}% up</div>
        </div>
        <div class="trade-stat">
          <div class="trade-stat-l">10D TARGET</div>
          <div class="trade-stat-v ${num(ten.expected_return_pct) >= 0 ? "up" : "dn"}">${fmtPrice(tenTarget)}</div>
          <div class="trade-stat-s">${fmtPct(num(ten.expected_return_pct))} · ${num(ten.trust_score).toFixed(1)} trust</div>
        </div>
        <div class="trade-stat">
          <div class="trade-stat-l">R:R TO RANGE</div>
          <div class="trade-stat-v ${rr !== null && rr >= 1 ? "up" : "dn"}">${rewardText}</div>
          <div class="trade-stat-s">risk ${fmtPct(downsidePct, 1)} · reward ${fmtPct(upsidePct, 1)}</div>
        </div>
        <div class="trade-stat">
          <div class="trade-stat-l">SUPPORT / RESIST</div>
          <div class="trade-stat-v">${fmtPrice(support)} / ${fmtPrice(resistance)}</div>
          <div class="trade-stat-s">MA20 ${fmtPrice(ma20)} · MA50 ${fmtPrice(ma50)}</div>
        </div>
        <div class="trade-stat">
          <div class="trade-stat-l">TACTICAL HEALTH</div>
          <div class="trade-stat-v ${num(perf.recent20_accuracy) >= 0.5 ? "up" : "dn"}">${(num(perf.recent20_accuracy) * 100).toFixed(1)}%</div>
          <div class="trade-stat-s">recent20 acc · Brier ${num(perf.recent20_brier).toFixed(3)}</div>
        </div>
      </div>
      <div class="trade-map">
        <div class="trade-map-head">
          <span>TRADE MAP · support to resistance</span>
          <span>${fmtPrice(rangeMin)} → ${fmtPrice(rangeMax)}</span>
        </div>
        <div class="trade-track">
          <div class="trade-band" style="left:${bandLeft}%;width:${bandWidth}%"></div>
          <div class="trade-marker ma20" style="left:${pos(ma20)}%"></div>
          <div class="trade-marker target10" style="left:${pos(tenTarget)}%"></div>
          <div class="trade-marker target1" style="left:${pos(oneTarget)}%"></div>
          <div class="trade-marker current" style="left:${pos(current)}%"></div>
        </div>
        <div class="trade-scale">
          <span>S20 ${fmtPrice(support)}</span>
          <span>LAST ${fmtPrice(current)}</span>
          <span>R20 ${fmtPrice(resistance)}</span>
        </div>
        <div class="trade-legend">
          <div class="trade-legend-item"><span class="trade-dot" style="background:#f4f4f4"></span>Last</div>
          <div class="trade-legend-item"><span class="trade-dot" style="background:var(--cyan)"></span>1D target</div>
          <div class="trade-legend-item"><span class="trade-dot" style="background:var(--pur)"></span>10D target</div>
          <div class="trade-legend-item"><span class="trade-dot" style="background:var(--gold)"></span>MA20</div>
          <div class="trade-legend-item"><span class="trade-dot" style="background:rgba(74,158,255,.7)"></span>1D band</div>
        </div>
      </div>
      <div class="trade-scan">
        <div class="trade-row">
          <span class="trade-row-l">1D ODDS / TRUST</span>
          <span class="trade-row-v">${num(payload.signal.probability_up).toFixed(1)}% up · ${num(payload.signal.trust_score).toFixed(1)} trust</span>
        </div>
        <div class="trade-row">
          <span class="trade-row-l">NEWS / CATALYST LOAD</span>
          <span class="trade-row-v">${payload.news_monitor.article_count || 0} items · score ${num(payload.news_monitor.net_score).toFixed(2)} · material ${payload.news_monitor.material_count || 0}</span>
        </div>
        <div class="trade-row">
          <span class="trade-row-l">MOMENTUM / TAPE</span>
          <span class="trade-row-v">RSI ${num(tech.rsi14).toFixed(1)} · vol ${num(tech.volume_ratio, 1).toFixed(2)}x · 5d ${fmtPct(num(tech.ret_5d_pct), 1)}</span>
        </div>
        <div class="trade-row">
          <span class="trade-row-l">MACRO CONTEXT</span>
          <span class="trade-row-v">SPY ${fmtPct(num(market.spy_ret_1_pct), 1)} · QQQ ${fmtPct(num(market.qqq_ret_1_pct), 1)} · ${market.risk_on ? "risk-on" : "risk-off"}</span>
        </div>
        <div class="trade-row">
          <span class="trade-row-l">INVALIDATION / RECLAIM</span>
          <span class="trade-row-v">below ${fmtPrice(invalidation)} · reclaim ${fmtPrice(reclaim)}</span>
        </div>
      </div>
      <div class="trade-summary">${summary}</div>`;
  }

  function renderIntel() {
    const payload = currentPayload();
    const el = document.getElementById("intelPanel");
    if (!payload || !el) return;
    const blocks = [];
    blocks.push(`<div class="iblock">
      <div class="itag nws">WHY IT'S MOVING</div>
      <div class="iline"><div class="ibul" style="background:${payload.news_monitor.net_score >= 0 ? "var(--G)" : "var(--R)"}"></div><span>${payload.news_monitor.summary || payload.summary}</span></div>
    </div>`);
    (payload.reasoning.drivers || []).slice(0, 4).forEach((driver) => {
      const cls = driver.direction === "bull" ? "var(--G)" : driver.direction === "bear" ? "var(--R)" : "var(--gold)";
      blocks.push(`<div class="iblock">
        <div class="itag ${driver.title.toLowerCase().includes("trend") ? "tec" : driver.title.toLowerCase().includes("model") ? "mdl" : "nws"}">${driver.title.toUpperCase()}</div>
        <div class="iline"><div class="ibul" style="background:${cls}"></div><span>${driver.detail}</span></div>
      </div>`);
    });
    blocks.push(`<div class="news-group">
      <div class="news-group-head"><span class="news-group-title">CATALYST STACK</span><span class="status-chip ${payload.news_monitor.used_recent_fallback ? "bad" : "good"}">${payload.news_monitor.status_label}</span></div>
      ${(payload.catalysts || []).map((catalyst) => `<div class="level-row">
        <span class="level-label">${catalyst.label}</span>
        <span class="level-val ${toneClass(catalyst.tone)}">${catalyst.value}</span>
      </div>`).join("")}
    </div>`);
    el.innerHTML = blocks.join("");
  }

  function renderBottomNews() {
    const payload = currentPayload();
    const panel = document.getElementById("newsPanel");
    const tape = document.getElementById("tape");
    const badge = document.getElementById("news-status-badge");
    if (!payload || !panel) return;
    if (badge) badge.textContent = `${payload.news_monitor.status_label} · ${payload.news_monitor.article_count} item(s)`;
    const groups = ["Premarket", "Today", "After Close", "Macro", "Analyst", "Company", "Fallback"];
    panel.innerHTML = groups
      .map((group) => {
        const items = (payload.news_monitor.items || []).filter((item) => item.section === group);
        if (!items.length) return "";
        return `<div class="news-group">
          <div class="news-group-head"><span class="news-group-title">${group}</span><span class="news-mini-meta">${items.length} item(s)</span></div>
          ${items.slice(0, 3).map((item) => `<div class="news-mini">
            <div class="news-mini-head">
              <span class="status-chip ${item.net_score >= 0 ? "good" : "bad"}">${item.impact}</span>
              <span class="news-mini-meta">${item.source} · ${fmtAgeFromIso(item.published_at_et)}</span>
            </div>
            <div class="news-mini-title">${item.headline}</div>
            <div class="news-mini-meta">${item.reason || payload.news_monitor.summary}</div>
          </div>`).join("")}
        </div>`;
      })
      .join("") || `<div class="news-group"><div class="news-mini-title">No current items in the selected focus feed.</div></div>`;
    if (tape) {
      const items = flattenNewsItems().slice(0, 16).map((item) => `<div class="ti">
        <span class="ti-s" style="color:${COLORS[item.ticker]}">${item.ticker}</span>
        <span style="color:#2a2a2a">${item.headline.slice(0, 52)}...</span>
        <span class="ti-v ${item.vader >= 0 ? "up" : "dn"}">${item.vader >= 0 ? "+" : ""}${item.vader.toFixed(2)}</span>
        <span style="color:#1c1c1c">·</span>
      </div>`).join("");
      tape.innerHTML = items + items;
    }
  }

  function renderResearchOps() {
    const payload = currentPayload();
    const stats = document.getElementById("opsStats");
    const rows = document.getElementById("modelRows");
    const badge = document.getElementById("ops-badge");
    if (!payload || !stats || !rows) return;
    if (badge) badge.textContent = `${payload.champion_model.family} · ${payload.champion_model.model_version.split("-").slice(-2).join(" ")}`;
    const perf = payload.recent_performance || {};
    const cal = payload.horizons["1d"].calibration || {};
    stats.innerHTML = `
      <div class="stc"><div class="scl">Recent20 Acc</div><div class="scv ${perf.recent20_accuracy >= 0.5 ? "up" : "dn"}">${(Number(perf.recent20_accuracy || 0) * 100).toFixed(1)}%</div></div>
      <div class="stc"><div class="scl">Brier</div><div class="scv" style="color:var(--blue)">${Number(cal.brier || 0).toFixed(3)}</div></div>
      <div class="stc"><div class="scl">ECE</div><div class="scv" style="color:var(--cyan)">${Number(cal.ece || 0).toFixed(3)}</div></div>`;
    const horizons = payload.forecast_overlay.scenario_ladder || [];
    rows.innerHTML = horizons.map((point) => {
      const fill = Math.max(0, Math.min(100, Number(point.probability_up || 0)));
      return `<div class="ladder-row">
        <span class="ladder-label">${point.label}</span>
        <div class="ladder-bar"><div class="ladder-fill" style="width:${fill}%;background:${point.expected_return_pct >= 0 ? "var(--G)" : "var(--R)"}"></div></div>
        <span class="ladder-val ${point.expected_return_pct >= 0 ? "up" : "dn"}">${fmtPct(point.expected_return_pct, 1)}</span>
      </div>`;
    }).join("") + `
      <div class="news-group">
        <div class="news-group-head"><span class="news-group-title">SUPPORT / RESISTANCE</span><span class="news-mini-meta">${payload.market_context.macro_regime}</span></div>
        <div class="level-row"><span class="level-label">Support 20d</span><span class="level-val">${fmtPrice(payload.levels.support_20d)}</span></div>
        <div class="level-row"><span class="level-label">Resistance 20d</span><span class="level-val">${fmtPrice(payload.levels.resistance_20d)}</span></div>
        <div class="level-row"><span class="level-label">MA20 / MA50</span><span class="level-val">${fmtPrice(payload.levels.ma20)} / ${fmtPrice(payload.levels.ma50)}</span></div>
      </div>`;
  }

  function renderCatalystLane() {
    const panel = document.getElementById("earnPanel");
    const badge = document.getElementById("catalyst-badge");
    if (!panel || !state.bundle) return;
    const rows = TICKERS.map((ticker) => {
      const payload = state.bundle.tickers[ticker];
      const catalyst = (payload.catalysts || [])[0] || {};
      const days = catalyst.detail || "";
      const value = catalyst.value || "—";
      const tag = catalyst.tone === "risk" ? "H" : catalyst.tone === "bull" ? "L" : "M";
      const dc = String(days).includes("999") ? "far" : /\d+/.test(days) && Number(days.match(/\d+/)[0]) <= 20 ? "urg" : /\d+/.test(days) && Number(days.match(/\d+/)[0]) <= 35 ? "mid" : "far";
      return `<div class="erow${ticker === state.cur ? " urg" : ""}" onclick="selectTicker('${ticker}')">
        <span class="esym" style="color:${COLORS[ticker]}">${ticker}</span>
        <span class="edate">${catalyst.label || "Catalyst"}</span>
        <span class="edays ${dc}">${days || "n/a"}</span>
        <span class="emv ${toneClass(catalyst.tone)}">${value}</span>
        <span class="erisk ${tag}">${tag}</span>
      </div>`;
    }).join("");
    panel.innerHTML = rows;
    if (badge) badge.textContent = `${state.cur} · ${(currentPayload()?.catalysts || [])[0]?.label || "Catalyst"}`;
  }

  function drawLine(ctx, values, toX, toY, color, width, dash = []) {
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.setLineDash(dash);
    let started = false;
    values.forEach((value, index) => {
      if (value == null || Number.isNaN(value)) return;
      const x = toX(index);
      const y = toY(value);
      if (!started) {
        ctx.moveTo(x, y);
        started = true;
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
    ctx.setLineDash([]);
  }

  function calcEMA(values, period) {
    const output = [];
    const k = 2 / (period + 1);
    let ema = null;
    values.forEach((value, index) => {
      if (index < period - 1) {
        output.push(null);
        return;
      }
      if (ema == null) {
        ema = values.slice(0, period).reduce((sum, item) => sum + item, 0) / period;
      } else {
        ema = value * k + ema * (1 - k);
      }
      output.push(ema);
    });
    return output;
  }

  function calcSMA(values, period) {
    return values.map((_, index) => {
      if (index < period - 1) return null;
      const window = values.slice(index - period + 1, index + 1);
      return window.reduce((sum, item) => sum + item, 0) / period;
    });
  }

  function calcBB(values, period, mult) {
    const sma = calcSMA(values, period);
    return {
      upper: values.map((_, index) => {
        if (sma[index] == null) return null;
        const slice = values.slice(index - period + 1, index + 1);
        const std = Math.sqrt(slice.reduce((sum, item) => sum + ((item - sma[index]) ** 2), 0) / period);
        return sma[index] + mult * std;
      }),
      lower: values.map((_, index) => {
        if (sma[index] == null) return null;
        const slice = values.slice(index - period + 1, index + 1);
        const std = Math.sqrt(slice.reduce((sum, item) => sum + ((item - sma[index]) ** 2), 0) / period);
        return sma[index] - mult * std;
      }),
    };
  }

  function renderChart() {
    const payload = currentPayload();
    const candles = currentCandles();
    const area = document.getElementById("chartArea");
    const main = document.getElementById("chartCanvas");
    const volCanvas = document.getElementById("volCanvas");
    if (!payload || !area || !main || !volCanvas || !candles.length) return;

    const rect = area.getBoundingClientRect();
    const width = Math.floor(rect.width);
    const height = Math.floor(rect.height);
    if (width < 40 || height < 40) return;

    const dpr = Math.min(window.devicePixelRatio || 1, 3);
    const axisWidth = 72;
    const volumeHeight = state.inds.VOL ? 58 : 0;
    const plotWidth = width - axisWidth;
    const plotHeight = height - volumeHeight;

    main.width = Math.floor(width * dpr);
    main.height = Math.floor(plotHeight * dpr);
    main.style.cssText = `position:absolute;top:0;left:0;width:${width}px;height:${plotHeight}px`;
    volCanvas.width = Math.floor(plotWidth * dpr);
    volCanvas.height = Math.floor(volumeHeight * dpr);
    volCanvas.style.cssText = state.inds.VOL ? `position:absolute;bottom:0;left:0;width:${plotWidth}px;height:${volumeHeight}px;border-top:1px solid #1e1e1e` : "display:none";

    const ctx = main.getContext("2d");
    const vctx = state.inds.VOL ? volCanvas.getContext("2d") : null;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (vctx) vctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, plotHeight);
    if (vctx) vctx.clearRect(0, 0, plotWidth, volumeHeight);

    const closes = candles.map((item) => Number(item.c));
    const overlay = payload.forecast_overlay.points || [];
    const overlayPrices = overlay.flatMap((point) => [point.target_price, point.upper_price, point.lower_price]);
    const highs = candles.map((item) => Number(item.h)).concat(overlayPrices);
    const lows = candles.map((item) => Number(item.l)).concat(overlayPrices);
    let maxPrice = Math.max(...highs);
    let minPrice = Math.min(...lows);
    const pad = (maxPrice - minPrice || 1) * 0.12;
    maxPrice += pad;
    minPrice -= pad;
    const range = maxPrice - minPrice || 1;
    const historyWidth = state.inds.PRED ? Math.floor(plotWidth * 0.78) : plotWidth - 8;
    const gap = Math.max(3, Math.floor((historyWidth - 8) / candles.length));
    const candleWidth = Math.max(2, Math.floor(gap * 0.72));
    const toX = (index) => 4 + index * gap + gap / 2;
    const toY = (price) => plotHeight * 0.04 + ((maxPrice - price) / range) * (plotHeight * 0.88);
    const predStartX = 4 + candles.length * gap;
    const predWidth = plotWidth - predStartX - 6;
    const predGap = overlay.length ? predWidth / overlay.length : 0;

    ctx.fillStyle = "#060606";
    ctx.fillRect(0, 0, plotWidth, plotHeight);
    if (state.inds.PRED && predWidth > 10) {
      ctx.fillStyle = "rgba(176,108,252,0.05)";
      ctx.fillRect(predStartX, 0, predWidth, plotHeight);
      ctx.strokeStyle = "rgba(176,108,252,0.24)";
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(predStartX, 0);
      ctx.lineTo(predStartX, plotHeight);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(255,255,255,0.35)";
      ctx.font = "bold 8px JetBrains Mono";
      ctx.fillText("NOW", predStartX - 28, 14);
      ctx.fillText("FORECAST", predStartX + 10, 14);
    }

    ctx.strokeStyle = "#0f0f0f";
    ctx.lineWidth = 1;
    for (let step = 0; step <= 5; step += 1) {
      const y = plotHeight * 0.04 + step * ((plotHeight * 0.88) / 5);
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(plotWidth, y);
      ctx.stroke();
      const price = maxPrice - (range * step) / 5;
      ctx.fillStyle = "#3a3a3a";
      ctx.font = "9px JetBrains Mono";
      ctx.fillText(fmtPrice(price), plotWidth + 4, y + 3);
    }

    const ema20 = calcEMA(closes, Math.min(20, closes.length));
    const ema50 = calcEMA(closes, Math.min(50, closes.length));
    const sma20 = calcSMA(closes, Math.min(20, closes.length));
    const bb = calcBB(closes, Math.min(20, closes.length), 2);
    if (state.inds.BB) {
      drawLine(ctx, bb.upper, toX, toY, "rgba(74,158,255,0.45)", 1, [3, 3]);
      drawLine(ctx, bb.lower, toX, toY, "rgba(74,158,255,0.45)", 1, [3, 3]);
    }
    if (state.inds.MA) drawLine(ctx, sma20, toX, toY, "rgba(247,183,49,0.55)", 1.1, []);
    if (state.inds.EMA) {
      drawLine(ctx, ema20, toX, toY, "rgba(247,183,49,0.82)", 1.1, []);
      drawLine(ctx, ema50, toX, toY, "rgba(74,158,255,0.82)", 1.1, []);
    }

    candles.forEach((candle, index) => {
      const x = toX(index);
      const up = Number(candle.c) >= Number(candle.o);
      const wick = up ? "#00a362" : "#cc1f3a";
      const body = up ? "#00d47e" : "#ff3d5a";
      ctx.strokeStyle = wick;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, toY(Number(candle.h)));
      ctx.lineTo(x, toY(Math.max(Number(candle.o), Number(candle.c))));
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(x, toY(Math.min(Number(candle.o), Number(candle.c))));
      ctx.lineTo(x, toY(Number(candle.l)));
      ctx.stroke();
      const top = toY(Math.max(Number(candle.o), Number(candle.c)));
      const bottom = toY(Math.min(Number(candle.o), Number(candle.c)));
      ctx.fillStyle = body;
      ctx.fillRect(x - candleWidth / 2, top, candleWidth, Math.max(1.5, bottom - top));
    });

    const latest = candles[candles.length - 1];
    const latestY = toY(Number(latest.c));
    ctx.strokeStyle = "rgba(255,255,255,0.18)";
    ctx.setLineDash([2, 5]);
    ctx.beginPath();
    ctx.moveTo(0, latestY);
    ctx.lineTo(plotWidth, latestY);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = Number(latest.c) >= Number(latest.o) ? "#00d47e" : "#ff3d5a";
    ctx.fillRect(plotWidth + 2, latestY - 9, axisWidth - 4, 18);
    ctx.fillStyle = "#000";
    ctx.font = "bold 9px JetBrains Mono";
    ctx.textAlign = "center";
    ctx.fillText(fmtPrice(Number(latest.c)), plotWidth + axisWidth / 2, latestY + 3);
    ctx.textAlign = "left";

    if (state.inds.PRED && overlay.length && predWidth > 10) {
      const baseX = predStartX + predGap / 2;
      const forecastXs = overlay.map((_, index) => baseX + index * predGap);
      const forecastYs = overlay.map((point) => toY(Number(point.target_price)));
      ctx.beginPath();
      ctx.moveTo(toX(candles.length - 1), toY(Number(latest.c)));
      overlay.forEach((point, index) => ctx.lineTo(forecastXs[index], toY(Number(point.upper_price))));
      for (let index = overlay.length - 1; index >= 0; index -= 1) ctx.lineTo(forecastXs[index], toY(Number(overlay[index].lower_price)));
      ctx.closePath();
      ctx.fillStyle = "rgba(176,108,252,0.08)";
      ctx.fill();

      ctx.beginPath();
      ctx.moveTo(toX(candles.length - 1), toY(Number(latest.c)));
      forecastYs.forEach((y, index) => ctx.lineTo(forecastXs[index], y));
      ctx.strokeStyle = "rgba(176,108,252,0.92)";
      ctx.lineWidth = 1.8;
      ctx.stroke();

      overlay.forEach((point, index) => {
        const x = forecastXs[index];
        const y = forecastYs[index];
        const up = Number(point.expected_return_pct) >= 0;
        const color = up ? "#00d47e" : "#ff3d5a";
        ctx.strokeStyle = up ? "rgba(0,212,126,0.22)" : "rgba(255,61,90,0.22)";
        ctx.setLineDash([2, 3]);
        ctx.beginPath();
        ctx.moveTo(x, plotHeight * 0.04);
        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.arc(x, y, 4, 0, Math.PI * 2);
        ctx.fillStyle = color;
        ctx.fill();
        ctx.strokeStyle = "#000";
        ctx.lineWidth = 1.5;
        ctx.stroke();
        ctx.fillStyle = "rgba(0,0,0,0.82)";
        ctx.fillRect(x - 24, y + (up ? -24 : 10), 58, 13);
        ctx.fillStyle = color;
        ctx.font = "bold 8px JetBrains Mono";
        ctx.textAlign = "center";
        ctx.fillText(`${point.label} ${fmtPct(point.expected_return_pct, 1)}`, x + 5, y + (up ? -15 : 19));
        ctx.textAlign = "left";
      });
    }

    ctx.fillStyle = "#373737";
    ctx.font = "8px JetBrains Mono";
    const step = Math.max(1, Math.floor(candles.length / 6));
    for (let index = 0; index < candles.length; index += step) {
      const x = toX(index);
      if (x > historyWidth) break;
      ctx.fillText(String(candles[index].d).slice(5), x - 12, plotHeight - 4);
    }

    if (vctx && state.inds.VOL) {
      const vols = candles.map((item) => Number(item.v));
      const maxVol = Math.max(...vols, 1);
      candles.forEach((candle, index) => {
        const up = Number(candle.c) >= Number(candle.o);
        const barHeight = Math.max(2, (Number(candle.v) / maxVol) * (volumeHeight - 8));
        vctx.fillStyle = up ? "rgba(0,212,126,0.45)" : "rgba(255,61,90,0.45)";
        vctx.fillRect(toX(index) - candleWidth / 2, volumeHeight - barHeight, candleWidth, barHeight);
      });
      vctx.fillStyle = "#2a2a2a";
      vctx.font = "7px JetBrains Mono";
      vctx.fillText(`VOL ${fmtCompact(latest.v)}`, 3, 9);
    }

    [["co", latest.o], ["ch2", latest.h], ["cl", latest.l], ["cc", latest.c], ["cv", fmtCompact(latest.v)]].forEach(([id, value]) => {
      const el = document.getElementById(id);
      if (el) el.textContent = typeof value === "string" ? value : fmtPrice(value);
    });
    const strip = document.getElementById("patStrip");
    if (strip) strip.innerHTML = "";
  }

  function setupCrosshair() {
    const area = document.getElementById("chartArea");
    if (!area || area.dataset.bound) return;
    area.dataset.bound = "1";
    area.addEventListener("mousemove", (event) => {
      const candles = currentCandles();
      const xhair = document.getElementById("xhair");
      if (!candles.length || !xhair) return;
      const rect = area.getBoundingClientRect();
      const width = Math.floor(rect.width);
      const axisWidth = 72;
      const plotWidth = width - axisWidth;
      const historyWidth = state.inds.PRED ? Math.floor(plotWidth * 0.78) : plotWidth - 8;
      const gap = Math.max(3, Math.floor((historyWidth - 8) / candles.length));
      const idx = Math.floor((event.clientX - rect.left - 4) / gap);
      if (idx < 0 || idx >= candles.length) return;
      const candle = candles[idx];
      xhair.style.display = "flex";
      xhair.innerHTML = `<span style="color:#333">${candle.d}</span>
        <span>O:<span class="ci-w">${fmtPrice(candle.o)}</span></span>
        <span>H:<span class="ci-${Number(candle.c) >= Number(candle.o) ? "up" : "dn"}">${fmtPrice(candle.h)}</span></span>
        <span>L:<span class="ci-${Number(candle.c) >= Number(candle.o) ? "up" : "dn"}">${fmtPrice(candle.l)}</span></span>
        <span>C:<span class="ci-w">${fmtPrice(candle.c)}</span></span>
        <span>Vol:<span class="ci-w">${fmtCompact(candle.v)}</span></span>`;
    });
    area.addEventListener("mouseleave", () => {
      const xhair = document.getElementById("xhair");
      if (xhair) xhair.style.display = "none";
    });
  }

  function initChart() {
    const area = document.getElementById("chartArea");
    if (!area) return;
    if (state.chartRO) state.chartRO.disconnect();
    state.chartRO = new ResizeObserver(() => renderChart());
    state.chartRO.observe(area);
    setupCrosshair();
  }

  function renderSignalsPage() {
    const grid = document.getElementById("signalsGrid");
    if (!grid || !state.bundle) return;
    grid.innerHTML = TICKERS.map((ticker) => {
      const payload = state.bundle.tickers[ticker];
      const sc = signalClass(payload.signal.signal);
      const horizons = payload.forecast_overlay.scenario_ladder || [];
      const edge = oneDayEdge(payload);
      return `<div class="sc">
        <div class="sc-head">
          <span class="sc-sym" style="color:${COLORS[ticker]}">${ticker}</span>
          <span class="sc-price">${fmtPrice(payload.quote_snapshot.close)}</span>
          <span class="sc-chg ${payload.quote_snapshot.change_pct >= 0 ? "up" : "dn"}">${fmtPct(payload.quote_snapshot.change_pct)}</span>
          <span class="sc-badge ${sc}">${payload.signal.signal}</span>
        </div>
        <div class="sc-body">
          <div class="news-group" style="padding:6px 0 8px;border-bottom:1px solid var(--ln)">
            <div class="news-group-head"><span class="news-group-title">1D EDGE</span><span class="status-chip ${edge.tone || statusClass(edge.label)}">${edge.label}</span></div>
            <div class="news-mini-meta">${edge.summary}</div>
          </div>
          <div class="hz-grid">
            ${horizons.map((point) => `<div class="hz-cell">
              <div class="hz-l">${point.label}</div>
              <div class="hz-v ${point.expected_return_pct >= 0 ? "up" : "dn"}">${fmtPct(point.expected_return_pct, 1)}</div>
              <div class="hz-p">${fmtPrice(point.target_price)}</div>
            </div>`).join("")}
          </div>
          <div class="cnf-row"><span class="cnf-l">Probability Up</span><div class="cnf-bar"><div class="cnf-fill" style="width:${payload.signal.probability_up}%;background:var(--pur)"></div></div><span class="cnf-v">${payload.signal.probability_up.toFixed(1)}%</span></div>
          <div class="cnf-row"><span class="cnf-l">Trust Score</span><div class="cnf-bar"><div class="cnf-fill" style="width:${payload.signal.trust_score}%;background:var(--cyan)"></div></div><span class="cnf-v">${payload.signal.trust_score.toFixed(1)}%</span></div>
          <div class="cnf-row"><span class="cnf-l">Trend Score</span><div class="cnf-bar"><div class="cnf-fill" style="width:${Math.max(0, Math.min(100, 50 + payload.trend_snapshot.score * 50))}%;background:${payload.trend_snapshot.score >= 0 ? "var(--G)" : "var(--R)"}"></div></div><span class="cnf-v">${payload.trend_snapshot.score.toFixed(2)}</span></div>
          <div class="news-group">
            <div class="news-group-head"><span class="news-group-title">Why it is moving</span><span class="news-mini-meta">${payload.trend_snapshot.state}</span></div>
            ${(payload.reasoning.drivers || []).slice(0, 3).map((driver) => `<div class="news-mini">
              <div class="news-mini-title">${driver.title}</div>
              <div class="news-mini-meta">${driver.detail}</div>
            </div>`).join("")}
          </div>
        </div>
      </div>`;
    }).join("");
  }

  function renderResearchPage() {
    const rc = document.getElementById("researchContent");
    if (!rc || !state.bundle) return;
    rc.innerHTML = `
      <div class="sec-head"><span class="sec-title">Terminal Research Snapshot · ${state.bundle.market_date}</span></div>
      <div class="kpi-grid">
        ${TICKERS.map((ticker) => {
          const payload = state.bundle.tickers[ticker];
          const edge = oneDayEdge(payload);
          return `<div class="kpi">
            <div class="kpi-l" style="color:${COLORS[ticker]}">${ticker} · ${payload.company_name}</div>
            <div class="kpi-v">${fmtPrice(payload.quote_snapshot.close)}</div>
            <div class="kpi-s">${payload.trend_snapshot.state} · Trust ${payload.signal.trust_score.toFixed(1)}% · ${edge.label}</div>
          </div>`;
        }).join("")}
      </div>
      <table class="rtable">
        <thead><tr><th>Ticker</th><th>Signal</th><th>Prob Up</th><th>1D</th><th>5D</th><th>10D</th><th>1D Edge</th><th>Trend</th><th>Freshness</th></tr></thead>
        <tbody>
          ${TICKERS.map((ticker) => {
            const payload = state.bundle.tickers[ticker];
            const edge = oneDayEdge(payload);
            return `<tr>
              <td class="m" style="color:${COLORS[ticker]}">${ticker}</td>
              <td class="${toneClass(payload.signal.signal)}">${payload.signal.signal}</td>
              <td class="m">${payload.signal.probability_up.toFixed(1)}%</td>
              <td class="${payload.horizons["1d"].expected_return_pct >= 0 ? "up" : "dn"}">${fmtPct(payload.horizons["1d"].expected_return_pct, 2)}</td>
              <td class="${payload.horizons["5d"].expected_return_pct >= 0 ? "up" : "dn"}">${fmtPct(payload.horizons["5d"].expected_return_pct, 2)}</td>
              <td class="${payload.horizons["10d"].expected_return_pct >= 0 ? "up" : "dn"}">${fmtPct(payload.horizons["10d"].expected_return_pct, 2)}</td>
              <td class="m">${edge.label}</td>
              <td class="m">${payload.trend_snapshot.state} ${payload.trend_snapshot.score.toFixed(2)}</td>
              <td class="m">${payload.data_freshness.is_stale ? "STALE" : "LIVE"}</td>
            </tr>`;
          }).join("")}
        </tbody>
      </table>`;
  }

  function renderModelsPage() {
    const mc = document.getElementById("modelsContent");
    if (!mc || !state.bundle) return;
    mc.innerHTML = `
      <div style="display:flex;align-items:center;gap:10px;padding:7px 14px;background:rgba(0,212,126,.03);border-bottom:1px solid var(--ln)">
        <div class="gdot"></div>
        <span style="font-size:9px;color:var(--t1)">Pages publish mode: ${state.bundle.pages_publish_mode.label} · backend remains champion-first</span>
      </div>
      ${TICKERS.map((ticker) => {
        const payload = state.bundle.tickers[ticker];
        return `<div class="mc">
          <div class="mc-head"><div><div class="mc-title">${ticker} · ${payload.champion_model.family}</div><div class="mc-sub">${payload.champion_model.model_version}</div></div></div>
          <div class="mm-grid">
            <div class="mm"><div class="mm-l">1D Prob</div><div class="mm-v">${payload.horizons["1d"].probability_up.toFixed(1)}%</div></div>
            <div class="mm"><div class="mm-l">5D Prob</div><div class="mm-v">${payload.horizons["5d"].probability_up.toFixed(1)}%</div></div>
            <div class="mm"><div class="mm-l">10D Prob</div><div class="mm-v">${payload.horizons["10d"].probability_up.toFixed(1)}%</div></div>
            <div class="mm"><div class="mm-l">Brier</div><div class="mm-v">${payload.horizons["1d"].calibration.brier.toFixed(3)}</div></div>
            <div class="mm"><div class="mm-l">Trend Fit</div><div class="mm-v">${payload.recent_performance.trend_stability.toFixed(2)}</div></div>
          </div>
          <div class="arch"><div class="arch-l">Selection Policy</div><div class="arch-t">${payload.champion_model.selection}</div></div>
        </div>`;
      }).join("")}`;
  }

  function renderRiskPage() {
    const rc = document.getElementById("riskContent");
    if (!rc || !state.bundle) return;
    rc.innerHTML = `
      <div class="risk-hd">
        <span class="regime bear">${currentPayload()?.market_context?.macro_regime || "NEUTRAL"} REGIME</span>
        <span style="font-size:9px;color:var(--t1)">Support, resistance, and trend conflict map</span>
      </div>
      <div class="risk-grid4">
        ${TICKERS.map((ticker) => {
          const payload = state.bundle.tickers[ticker];
          return `<div class="risk-card">
            <div class="risk-title">${ticker} · ${payload.signal.signal}</div>
            <div class="risk-item"><span class="ri-label">Support 20d</span><div class="ri-wrap"><div class="ri-bar low" style="width:${Math.max(10, 100 - Math.abs(payload.levels.distance_to_20d_low_pct) * 4)}%"></div></div><span class="ri-val">${fmtPrice(payload.levels.support_20d)}</span></div>
            <div class="risk-item"><span class="ri-label">Resistance</span><div class="ri-wrap"><div class="ri-bar high" style="width:${Math.max(10, 100 - Math.abs(payload.levels.distance_to_20d_high_pct) * 4)}%"></div></div><span class="ri-val">${fmtPrice(payload.levels.resistance_20d)}</span></div>
            <div class="risk-item"><span class="ri-label">Trust</span><div class="ri-wrap"><div class="ri-bar ${payload.signal.trust_score >= 60 ? "low" : payload.signal.trust_score >= 50 ? "mid" : "high"}" style="width:${payload.signal.trust_score}%"></div></div><span class="ri-val">${payload.signal.trust_score.toFixed(1)}%</span></div>
            <div class="risk-item"><span class="ri-label">Trend</span><div class="ri-wrap"><div class="ri-bar ${payload.trend_snapshot.score >= 0.25 ? "low" : payload.trend_snapshot.score <= -0.25 ? "high" : "mid"}" style="width:${Math.max(10, Math.min(100, 50 + payload.trend_snapshot.score * 50))}%"></div></div><span class="ri-val">${payload.trend_snapshot.score.toFixed(2)}</span></div>
          </div>`;
        }).join("")}
      </div>`;
  }

  function renderAnalysisPage() {
    const ac = document.getElementById("analysisContent");
    const payload = currentPayload();
    if (!ac || !payload) return;
    const edge = oneDayEdge(payload);
    ac.innerHTML = `
      <div class="anlys-hero">
        <div class="anlys-hero-title">${state.cur} Desk Analysis</div>
        <div class="anlys-hero-sub">Market date ${payload.market_date} · Forecast for ${payload.forecast_for_date} · ${state.bundle.pages_publish_mode.label} · ${edge.label}</div>
      </div>
      <div class="anlys-nlp">
        <div class="anlys-nlp-title">Desk Read</div>
        <div class="anlys-nlp-body">
          <strong>${payload.summary}</strong><br><br>
          ${payload.reasoning.summary}<br><br>
          <strong>1D edge:</strong> ${edge.summary}<br><br>
          <strong>News monitor:</strong> ${payload.news_monitor.summary || "No fresh catalyst in the current feed."}<br><br>
          <strong>Trend structure:</strong> ${payload.trend_snapshot.state} (${payload.trend_snapshot.score.toFixed(2)}) with 5d slope ${fmtPct(payload.trend_snapshot.close_slope_5d, 3)}/day and 20d slope ${fmtPct(payload.trend_snapshot.close_slope_20d, 3)}/day.
        </div>
      </div>
      <div class="anlys-grid">
        <div class="anlys-card"><div class="anlys-card-title">1D · ${edge.label}</div><div style="font-family:var(--mono);font-size:20px;font-weight:700;color:${payload.horizons["1d"].expected_return_pct >= 0 ? "var(--G)" : "var(--R)"}">${fmtPct(payload.horizons["1d"].expected_return_pct, 2)}</div><div style="font-size:9px;color:var(--t2);margin-top:6px">${fmtPrice(payload.horizons["1d"].target_price)}</div></div>
        <div class="anlys-card"><div class="anlys-card-title">5D</div><div style="font-family:var(--mono);font-size:20px;font-weight:700;color:${payload.horizons["5d"].expected_return_pct >= 0 ? "var(--G)" : "var(--R)"}">${fmtPct(payload.horizons["5d"].expected_return_pct, 2)}</div><div style="font-size:9px;color:var(--t2);margin-top:6px">${fmtPrice(payload.horizons["5d"].target_price)}</div></div>
        <div class="anlys-card"><div class="anlys-card-title">10D</div><div style="font-family:var(--mono);font-size:20px;font-weight:700;color:${payload.horizons["10d"].expected_return_pct >= 0 ? "var(--G)" : "var(--R)"}">${fmtPct(payload.horizons["10d"].expected_return_pct, 2)}</div><div style="font-size:9px;color:var(--t2);margin-top:6px">${fmtPrice(payload.horizons["10d"].target_price)}</div></div>
      </div>
      <div class="anlys-section">
        <div class="anlys-section-title">Current Drivers</div>
        ${(payload.reasoning.drivers || []).map((driver) => `<div class="anlys-factor">
          <div class="anlys-factor-row"><span class="anlys-factor-name">${driver.title}</span><span class="anlys-factor-val ${toneClass(driver.direction)}">${driver.direction.toUpperCase()}</span></div>
          <div class="anlys-bar-wrap"><div class="anlys-bar-fill ${driver.direction === "bull" ? "bull" : driver.direction === "bear" ? "bear" : "neutral"}" style="width:${driver.direction === "neutral" ? 55 : 82}%"></div></div>
          <div style="font-size:8px;color:var(--t2);line-height:1.55;margin-top:4px">${driver.detail}</div>
        </div>`).join("")}
      </div>`;
  }

  function renderNewsPage() {
    const feed = document.getElementById("npFeed");
    if (!feed || !state.bundle) return;
    const sourceStamp = document.getElementById("npSourceStamp");
    const liveSummary = document.getElementById("npLiveSummary");
    if (sourceStamp) sourceStamp.textContent = `${state.bundle.pages_publish_mode.label} · ${state.bundle.market_date} → ${state.bundle.forecast_for_date}`;
    if (liveSummary) liveSummary.textContent = `${flattenNewsItems().length} articles · grouped by terminal bundle sections`;
    let items = flattenNewsItems();
    if (state.npActiveFilter === "bull") items = items.filter((item) => item.dir === "bull");
    else if (state.npActiveFilter === "bear") items = items.filter((item) => item.dir === "bear");
    else if (state.npActiveFilter === "HIGH") items = items.filter((item) => item.impact === "HIGH");
    else if (TICKERS.includes(state.npActiveFilter)) items = items.filter((item) => item.ticker === state.npActiveFilter);
    items.sort((a, b) => Math.abs(b.vader) - Math.abs(a.vader));
    feed.innerHTML = items.map((item) => `<div class="nc${state.npActiveId === item.id ? " on" : ""}" onclick="npOpen('${item.id}')">
      <div class="nc-bar" style="background:${item.dir === "bull" ? "var(--G)" : "var(--R)"}"></div>
      <div class="nc-meta">
        <span class="nc-tick ${item.ticker}">${item.ticker}</span>
        <span class="nc-impact ${item.impact} ${item.dir}">${item.impact}</span>
        <span class="nc-sent ${item.dir === "bull" ? "up" : "dn"}">${item.vader >= 0 ? "+" : ""}${item.vader.toFixed(2)}</span>
        <span style="font-family:var(--mono);font-size:7px;color:var(--cyan);padding:1px 3px;background:rgba(0,212,212,.08);border-radius:2px">${item.news_cat}</span>
        <span class="nc-age">${item.age} ago</span>
      </div>
      <div class="nc-headline">${item.headline}</div>
      <div class="nc-source">${item.source}</div>
      <a class="nc-link" href="${item.url}" target="_blank" rel="noopener" onclick="event.stopPropagation()">Read source ↗</a>
    </div>`).join("");
    if (!state.npActiveId && items[0]) npOpen(items[0].id);
  }

  function npOpen(id) {
    state.npActiveId = id;
    const item = flattenNewsItems().find((entry) => entry.id === id);
    const dt = document.getElementById("npDetailTick");
    const title = document.getElementById("npDetailTitle");
    const body = document.getElementById("npDetailBody");
    if (!item || !dt || !title || !body) return;
    dt.textContent = item.ticker;
    dt.style.color = COLORS[item.ticker] || "var(--G)";
    title.textContent = item.headline;
    const color = item.dir === "bull" ? "var(--G)" : "var(--R)";
    body.innerHTML = `
      <div class="np-d-headline">${item.headline}</div>
      <div class="np-d-meta">
        <span style="color:${COLORS[item.ticker]}">${item.ticker}</span>
        <span>${item.source}</span>
        <span class="nc-impact ${item.impact} ${item.dir}" style="font-size:9px;padding:1px 5px">${item.impact}</span>
        <span style="color:${color}">${item.news_cat}</span>
        <span>${item.age} ago</span>
        <a href="${item.url}" target="_blank" rel="noopener" style="margin-left:auto;color:var(--blue);font-size:9px;font-family:var(--mono);text-decoration:none">Open source ↗</a>
      </div>
      ${item.body.split("\n\n").filter(Boolean).map((paragraph) => `<div class="np-d-text">${paragraph}</div>`).join("")}
      <div class="np-analysis">
        <div class="np-analysis-title">AXIOM Desk Read</div>
        <div class="np-analysis-row"><span class="np-analysis-label">Bundle Section</span><span class="np-analysis-val" style="color:var(--cyan)">${item.news_cat}</span></div>
        <div class="np-analysis-row"><span class="np-analysis-label">Calibrated Bias</span><span class="np-analysis-val ${item.dir === "bull" ? "up" : "dn"}">${item.px_impact}</span></div>
        <div class="np-analysis-row"><span class="np-analysis-label">Sentiment</span><span class="np-analysis-val ${item.dir === "bull" ? "up" : "dn"}">${item.vader >= 0 ? "+" : ""}${item.vader.toFixed(3)}</span></div>
        <div style="margin-top:10px;font-size:8px;color:var(--t2);font-family:var(--mono);margin-bottom:5px">AXIOM IMPACT BY HORIZON</div>
        <div class="np-horizon">${item.horizons.map((h) => `<div class="nh-cell"><div class="nh-l">${h.l}</div><div class="nh-v ${parseFloat(h.v) >= 0 ? "up" : "dn"}">${h.v}</div></div>`).join("")}</div>
      </div>`;
    document.querySelectorAll(".nc").forEach((node) => node.classList.toggle("on", node.getAttribute("onclick") === `npOpen('${id}')`));
  }

  function npFilter(filter) {
    state.npActiveFilter = filter;
    document.querySelectorAll(".np-filter").forEach((el) => {
      const txt = el.textContent.trim();
      const on = txt === filter || (filter === "bull" && txt === "BULL") || (filter === "bear" && txt === "BEAR") || (filter === "HIGH" && txt === "HIGH IMPACT");
      el.classList.toggle("on", on);
    });
    renderNewsPage();
  }

  function showPage(name) {
    state.curPage = name;
    document.querySelectorAll(".nl").forEach((el, index) => {
      el.classList.toggle("on", ["terminal", "news", "signals", "research", "models", "risk", "analysis"][index] === name);
    });
    document.querySelectorAll(".page").forEach((el) => el.classList.toggle("active", el.id === `pg-${name}`));
    if (name === "terminal") requestAnimationFrame(() => renderChart());
    if (name === "news") renderNewsPage();
    if (name === "signals") renderSignalsPage();
    if (name === "research") renderResearchPage();
    if (name === "models") renderModelsPage();
    if (name === "risk") renderRiskPage();
    if (name === "analysis") renderAnalysisPage();
  }

  function selectTicker(ticker) {
    state.cur = ticker;
    renderAll();
  }

  function setTime(timeframe) {
    state.curTime = timeframe;
    document.querySelectorAll(".ttab").forEach((el) => el.classList.toggle("on", el.textContent === TIME_LABELS[timeframe]));
    renderChart();
  }

  function toggleInd(key) {
    state.inds[key] = !state.inds[key];
    const button = document.getElementById(`tog-${key}`);
    if (button) button.classList.toggle("on", state.inds[key]);
    renderChart();
  }

  function renderAll() {
    renderNavStatus();
    renderRibbon();
    renderSigTable();
    renderWatchMeta();
    updateChartHeader();
    updatePredictionCard();
    renderTradeVisualizer();
    renderIntel();
    renderBottomNews();
    renderResearchOps();
    renderCatalystLane();
    if (state.curPage === "signals") renderSignalsPage();
    if (state.curPage === "research") renderResearchPage();
    if (state.curPage === "models") renderModelsPage();
    if (state.curPage === "risk") renderRiskPage();
    if (state.curPage === "analysis") renderAnalysisPage();
    if (state.curPage === "news") renderNewsPage();
    requestAnimationFrame(() => renderChart());
  }

  async function loadBundle() {
    const response = await fetch("terminal_live_bundle.json", { cache: "no-store" });
    if (!response.ok) throw new Error(`terminal_live_bundle.json ${response.status}`);
    const bundle = await response.json();
    state.bundle = bundle;
    state.cur = bundle.default_ticker || "PLTR";
    state.curTime = (bundle.timeframes || []).includes("1m") ? "1m" : (bundle.timeframes || [])[0] || "all";
  }

  function init() {
    window.__AXIOM_TERMINAL_V2__ = true;
    window.showPage = showPage;
    window.selectTicker = selectTicker;
    window.setTime = setTime;
    window.toggleInd = toggleInd;
    window.npFilter = npFilter;
    window.npOpen = npOpen;
    window.initChart = initChart;
    window.renderChart = renderChart;
    window.renderNewsPage = renderNewsPage;
    window.renderSignalsPage = renderSignalsPage;
    window.renderResearchPage = renderResearchPage;
    window.renderModelsPage = renderModelsPage;
    window.renderRiskPage = renderRiskPage;
    window.renderAnalysisPage = renderAnalysisPage;

    setInterval(updateClock, 1000);
    updateClock();
    initChart();

    loadBundle()
      .then(() => {
        renderAll();
      })
      .catch((error) => {
        console.error("AXIOM terminal bundle load failed:", error);
        const status = document.getElementById("nav-status-pill");
        if (status) status.textContent = "DATA ERROR";
        const live = document.getElementById("nav-live-pill");
        if (live) live.innerHTML = `<div class="dot-g"></div>OFFLINE`;
      });
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
}());
