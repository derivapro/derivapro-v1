const $ = (id) => document.getElementById(id);

const embeddedMarketData = {
  as_of: "2026-06-11",
  underlier: "DEMO",
  description: "Static sample market data for the DerivaPro Lite browser demo. Values are illustrative and not live market data.",
  spot: 100.0,
  risk_free_rate: 0.045,
  dividend_yield: 0.018,
  vol_surface: {
    maturities_years: [0.25, 0.5, 1.0, 2.0],
    moneyness: [0.8, 0.9, 1.0, 1.1, 1.2],
    volatility: [
      [0.285, 0.255, 0.225, 0.215, 0.225],
      [0.275, 0.245, 0.215, 0.205, 0.215],
      [0.265, 0.235, 0.205, 0.198, 0.208],
      [0.255, 0.228, 0.200, 0.195, 0.205],
    ],
  },
  yield_curve: [
    { tenor: "3M", years: 0.25, rate: 0.0430 },
    { tenor: "6M", years: 0.50, rate: 0.0440 },
    { tenor: "1Y", years: 1.00, rate: 0.0450 },
    { tenor: "2Y", years: 2.00, rate: 0.0460 },
    { tenor: "5Y", years: 5.00, rate: 0.0475 },
    { tenor: "10Y", years: 10.00, rate: 0.0490 },
    { tenor: "30Y", years: 30.00, rate: 0.0500 },
  ],
  portfolio: [
    { book: "Equity Options", driver: "Delta / Vega", delta_notional: 185000, vega_notional: 1200 },
    { book: "Rates", driver: "DV01", dv01: -134 },
    { book: "Credit", driver: "CS01", cs01: -42.9 },
  ],
};

let marketData = embeddedMarketData;

function number(id) {
  return Number($(id).value || 0);
}

function currency(value, digits = 0) {
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: digits,
    minimumFractionDigits: digits,
  }).format(value);
}

function pct(value, digits = 2) {
  return `${value.toFixed(digits)}%`;
}

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function lerp(x, x0, x1, y0, y1) {
  if (x1 === x0) return y0;
  const w = (x - x0) / (x1 - x0);
  return y0 + w * (y1 - y0);
}

function bracket(grid, value) {
  if (value <= grid[0]) return [0, 0];
  if (value >= grid[grid.length - 1]) return [grid.length - 1, grid.length - 1];
  for (let i = 0; i < grid.length - 1; i += 1) {
    if (value >= grid[i] && value <= grid[i + 1]) return [i, i + 1];
  }
  return [0, 0];
}

function interpolateYield(years) {
  const curve = marketData.yield_curve;
  if (years <= curve[0].years) return curve[0].rate;
  if (years >= curve[curve.length - 1].years) return curve[curve.length - 1].rate;
  for (let i = 0; i < curve.length - 1; i += 1) {
    const a = curve[i];
    const b = curve[i + 1];
    if (years >= a.years && years <= b.years) {
      return lerp(years, a.years, b.years, a.rate, b.rate);
    }
  }
  return curve[0].rate;
}

function interpolateVol(maturity, moneyness) {
  const surface = marketData.vol_surface;
  const tGrid = surface.maturities_years;
  const mGrid = surface.moneyness;
  const m = clamp(moneyness, mGrid[0], mGrid[mGrid.length - 1]);
  const t = clamp(maturity, tGrid[0], tGrid[tGrid.length - 1]);
  const [t0, t1] = bracket(tGrid, t);
  const [m0, m1] = bracket(mGrid, m);
  const v00 = surface.volatility[t0][m0];
  const v01 = surface.volatility[t0][m1];
  const v10 = surface.volatility[t1][m0];
  const v11 = surface.volatility[t1][m1];
  const vt0 = lerp(m, mGrid[m0], mGrid[m1], v00, v01);
  const vt1 = lerp(m, mGrid[m0], mGrid[m1], v10, v11);
  return lerp(t, tGrid[t0], tGrid[t1], vt0, vt1);
}

function parseNumberList(value, fallback) {
  const parsed = String(value || "")
    .split(",")
    .map((item) => Number(item.trim()))
    .filter((item) => Number.isFinite(item));
  return parsed.length ? parsed : fallback;
}

function normalPdf(x) {
  return Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
}

function erf(x) {
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const a1 = 0.254829592;
  const a2 = -0.284496736;
  const a3 = 1.421413741;
  const a4 = -1.453152027;
  const a5 = 1.061405429;
  const p = 0.3275911;
  const t = 1 / (1 + p * ax);
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-ax * ax);
  return sign * y;
}

function normalCdf(x) {
  return 0.5 * (1 + erf(x / Math.sqrt(2)));
}

function blackScholes(type, spot, strike, rate, vol, maturity) {
  const r = rate / 100;
  const sigma = Math.max(vol / 100, 0.0001);
  const t = Math.max(maturity, 0.0001);
  const d1 = (Math.log(spot / strike) + (r + 0.5 * sigma * sigma) * t) / (sigma * Math.sqrt(t));
  const d2 = d1 - sigma * Math.sqrt(t);
  const discount = Math.exp(-r * t);
  const call = spot * normalCdf(d1) - strike * discount * normalCdf(d2);
  const put = strike * discount * normalCdf(-d2) - spot * normalCdf(-d1);
  const price = type === "call" ? call : put;
  const delta = type === "call" ? normalCdf(d1) : normalCdf(d1) - 1;
  const gamma = normalPdf(d1) / (spot * sigma * Math.sqrt(t));
  const vega = spot * normalPdf(d1) * Math.sqrt(t) / 100;
  const thetaCall = (-(spot * normalPdf(d1) * sigma) / (2 * Math.sqrt(t)) - r * strike * discount * normalCdf(d2)) / 365;
  const thetaPut = (-(spot * normalPdf(d1) * sigma) / (2 * Math.sqrt(t)) + r * strike * discount * normalCdf(-d2)) / 365;
  const rhoCall = strike * t * discount * normalCdf(d2) / 100;
  const rhoPut = -strike * t * discount * normalCdf(-d2) / 100;
  return {
    price,
    delta,
    gamma,
    vega,
    theta: type === "call" ? thetaCall : thetaPut,
    rho: type === "call" ? rhoCall : rhoPut,
  };
}

function drawLineChart(svg, points, options = {}) {
  const width = 640;
  const height = 220;
  const pad = 34;
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys, 0);
  const maxY = Math.max(...ys);
  const xScale = (x) => pad + ((x - minX) / (maxX - minX || 1)) * (width - pad * 2);
  const yScale = (y) => height - pad - ((y - minY) / (maxY - minY || 1)) * (height - pad * 2);
  const path = points.map((p, i) => `${i === 0 ? "M" : "L"} ${xScale(p.x).toFixed(1)} ${yScale(p.y).toFixed(1)}`).join(" ");
  const zeroY = yScale(0);
  svg.innerHTML = `
    <line class="grid-line" x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}"></line>
    <line class="grid-line" x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}"></line>
    <line class="axis" x1="${pad}" y1="${zeroY}" x2="${width - pad}" y2="${zeroY}"></line>
    <path class="line ${options.alt ? "alt" : ""}" d="${path}"></path>
    <text class="chart-label" x="${pad}" y="${height - 8}">${options.leftLabel || minX.toFixed(0)}</text>
    <text class="chart-label" x="${width - pad - 52}" y="${height - 8}">${options.rightLabel || maxX.toFixed(0)}</text>
    <text class="chart-label" x="${pad}" y="18">${options.topLabel || currency(maxY, 0)}</text>
  `;
}

function drawBarChart(svg, bars) {
  const width = 640;
  const height = 220;
  const pad = 34;
  const maxAbs = Math.max(...bars.map((b) => Math.abs(b.value)), 1);
  const zero = height / 2;
  const gap = 22;
  const barWidth = (width - pad * 2 - gap * (bars.length - 1)) / bars.length;
  const scale = (height / 2 - pad) / maxAbs;
  const nodes = bars.map((bar, i) => {
    const x = pad + i * (barWidth + gap);
    const h = Math.abs(bar.value) * scale;
    const y = bar.value < 0 ? zero : zero - h;
    const cls = bar.value < 0 ? "bar negative" : "bar";
    return `
      <rect class="${cls}" x="${x}" y="${y}" width="${barWidth}" height="${h}" rx="5"></rect>
      <text class="chart-label" x="${x}" y="${height - 12}">${bar.label}</text>
      <text class="chart-label" x="${x}" y="${bar.value < 0 ? y + h + 18 : y - 8}">${currency(bar.value, 0)}</text>
    `;
  }).join("");
  svg.innerHTML = `<line class="axis" x1="${pad}" y1="${zero}" x2="${width - pad}" y2="${zero}"></line>${nodes}`;
}

function updateBarrier() {
  const style = $("barrier-type").value;
  const spot = number("barrier-spot");
  const strike = number("barrier-strike");
  const barrier = number("barrier-level");
  const maturity = number("barrier-maturity");
  const rate = number("barrier-rate");
  const surfaceVol = interpolateVol(maturity, strike / spot) * 100;
  const vanilla = blackScholes("call", spot, strike, rate, surfaceVol, maturity).price;
  const distance = style === "down-out" ? (spot - barrier) / spot : (barrier - spot) / spot;
  const clampedDistance = clamp(distance, 0, 1);
  const survival = clamp(0.35 + 2.6 * clampedDistance, 0, 0.98);
  const knockedOut = distance <= 0;
  const price = knockedOut ? 0 : vanilla * survival;
  $("barrier-vol-readout").textContent = pct(surfaceVol, 2);
  $("barrier-price").textContent = currency(price, 2);
  $("barrier-vanilla").textContent = currency(vanilla, 2);
  $("barrier-survival").textContent = pct(knockedOut ? 0 : survival * 100, 1);
  $("barrier-distance").textContent = pct(distance * 100, 1);
  const report = $("report-barrier-price");
  if (report) report.textContent = currency(price, 2);
}

function updateAsian() {
  const type = $("asian-type").value;
  const spot = number("asian-spot");
  const strike = number("asian-strike");
  const rate = number("asian-rate");
  const vol = number("asian-volatility");
  const maturity = number("asian-maturity");
  const observations = Math.max(1, number("asian-observations"));
  const adjustedVol = vol / Math.sqrt(3) * Math.sqrt((observations + 1) / observations);
  const asian = blackScholes(type, spot, strike, rate, adjustedVol, maturity).price;
  const euro = blackScholes(type, spot, strike, rate, vol, maturity).price;
  const discount = euro === 0 ? 0 : (1 - asian / euro) * 100;
  $("asian-price").textContent = currency(asian, 2);
  $("asian-adj-vol").textContent = pct(adjustedVol, 2);
  $("asian-euro-ref").textContent = currency(euro, 2);
  $("asian-discount").textContent = pct(discount, 1);
  const report = $("report-asian-price");
  if (report) report.textContent = currency(asian, 2);
}

function updateOptions() {
  const type = $("option-type").value;
  const spot = number("spot");
  const strike = number("strike");
  const maturity = number("maturity");
  const moneyness = strike / spot;
  const surfaceVol = interpolateVol(maturity, moneyness) * 100;
  const rate = $("option-data-mode").value === "surface" ? interpolateYield(maturity) * 100 : number("rate");
  const vol = $("option-data-mode").value === "surface" ? surfaceVol : number("volatility");
  $("surface-vol-readout").textContent = pct(surfaceVol, 2);
  $("market-data-asof").textContent = `Sample market data as of ${marketData.as_of}`;
  const result = blackScholes(type, spot, strike, rate, vol, maturity);
  $("option-price").textContent = currency(result.price, 2);
  $("delta").textContent = result.delta.toFixed(4);
  $("gamma").textContent = result.gamma.toFixed(4);
  $("vega").textContent = result.vega.toFixed(4);
  $("theta").textContent = result.theta.toFixed(4);
  $("rho").textContent = result.rho.toFixed(4);
  $("moneyness").textContent = pct((spot / strike) * 100, 1);

  const points = [];
  for (let i = 0; i <= 16; i += 1) {
    const s = spot * (0.8 + i * 0.025);
    const localVol = $("option-data-mode").value === "surface" ? interpolateVol(maturity, strike / s) * 100 : vol;
    points.push({ x: s, y: blackScholes(type, s, strike, rate, localVol, maturity).price });
  }
  drawLineChart($("option-chart"), points, {
    leftLabel: currency(spot * 0.8, 0),
    rightLabel: currency(spot * 1.2, 0),
  });
}

function bondPrice(face, couponRate, ytm, years, frequency) {
  const c = face * (couponRate / 100) / frequency;
  const y = ytm / 100 / frequency;
  const n = Math.max(1, Math.round(years * frequency));
  let price = 0;
  let weighted = 0;
  let convex = 0;
  for (let i = 1; i <= n; i += 1) {
    const cash = i === n ? c + face : c;
    const df = 1 / Math.pow(1 + y, i);
    const pv = cash * df;
    const time = i / frequency;
    price += pv;
    weighted += time * pv;
    convex += time * (time + 1 / frequency) * pv;
  }
  const macaulay = weighted / price;
  const modified = macaulay / (1 + y);
  return { price, modified, convexity: convex / price / Math.pow(1 + y, 2) };
}

function updateStructuredAutocallable() {
  const mode = $("structured-mode").value;
  const notional = number("structured-notional");
  const maturity = Math.max(0.25, number("structured-maturity"));
  const observations = Math.max(1, number("structured-observations"));
  const coupon = number("structured-coupon") / 100;
  const couponBarrier = number("structured-coupon-barrier") / 100;
  const autocallBarrier = number("structured-autocall-barrier") / 100;
  const protectionBarrier = number("structured-protection-barrier") / 100;
  const memoryEnabled = $("structured-memory").value === "yes";
  const correlation = clamp(number("structured-correlation"), -0.95, 0.95);
  const allSpots = parseNumberList($("structured-spots").value, [100, 95, 90]);
  const allVols = parseNumberList($("structured-vols").value, [22, 24, 26]);
  const spots = mode === "single" ? [allSpots[0]] : allSpots;
  const vols = mode === "single" ? [allVols[0]] : allVols;
  const avgVol = (vols.reduce((a, b) => a + b, 0) / vols.length) / 100;
  const basketPenalty = mode === "single" ? 0 : clamp((spots.length - 1) * 0.035 * (1 - correlation), 0, 0.22);
  const riskVol = avgVol * Math.sqrt(maturity) * (1 + basketPenalty);
  const expectedWorstFinal = clamp(1 - 0.18 * riskVol - basketPenalty, 0.45, 1.25);
  const autocallDistance = autocallBarrier - 1;
  const autocallProb = clamp(0.58 - 1.45 * autocallDistance - 0.75 * riskVol - basketPenalty, 0.03, 0.92);
  const couponProb = clamp(0.97 - Math.max(couponBarrier - expectedWorstFinal, 0) * 1.8 - 0.25 * riskVol, 0.05, 0.99);
  const breachProb = clamp(0.02 + Math.max(protectionBarrier - expectedWorstFinal, 0) * 1.9 + 0.42 * riskVol + basketPenalty * 0.65, 0.01, 0.75);
  const expectedCoupons = observations * couponProb * (memoryEnabled ? 1.08 : 0.92);
  const memoryValue = memoryEnabled ? notional * coupon * observations * 0.08 * (1 - couponProb) : 0;
  const couponPv = notional * coupon * expectedCoupons * Math.exp(-marketData.risk_free_rate * maturity * 0.5);
  const protectionLoss = notional * breachProb * Math.max(1 - expectedWorstFinal, 0.08);
  const earlyRedemptionValue = notional * autocallProb * 0.012;
  const price = notional + couponPv + memoryValue + earlyRedemptionValue - protectionLoss;

  $("structured-price").textContent = currency(price, 0);
  $("structured-autocall-prob").textContent = pct(autocallProb * 100, 1);
  $("structured-breach-prob").textContent = pct(breachProb * 100, 1);
  $("structured-coupons").textContent = `${expectedCoupons.toFixed(1)}x`;
  $("structured-worst-final").textContent = pct(expectedWorstFinal * 100, 1);
  $("structured-basket-penalty").textContent = pct(basketPenalty * 100, 1);
  $("structured-memory-value").textContent = currency(memoryValue, 0);

  const payoffPoints = [];
  for (let i = 0; i <= 18; i += 1) {
    const level = 0.4 + i * 0.05;
    const protectedRedemption = level >= protectionBarrier ? notional : notional * level;
    const couponCount = level >= couponBarrier ? observations : (memoryEnabled && level >= couponBarrier * 0.92 ? observations * 0.5 : 0);
    const payoff = protectedRedemption + notional * coupon * couponCount;
    payoffPoints.push({ x: level * 100, y: payoff });
  }
  drawLineChart($("structured-chart"), payoffPoints, {
    leftLabel: "40%",
    rightLabel: "130%",
    topLabel: "Payoff",
  });

  const report = $("report-structured-price");
  if (report) report.textContent = currency(price, 0);
}

function updateBonds() {
  const face = number("bond-face");
  const coupon = number("coupon-rate");
  const years = number("bond-years");
  const frequency = Math.max(1, number("bond-frequency"));
  const curveYield = interpolateYield(years) * 100;
  const ytm = $("bond-rate-mode").value === "curve" ? curveYield : number("ytm");
  $("curve-yield-readout").textContent = pct(curveYield, 2);
  const base = bondPrice(face, coupon, ytm, years, frequency);
  const down = bondPrice(face, coupon, ytm - 1, years, frequency);
  const up = bondPrice(face, coupon, ytm + 1, years, frequency);
  $("bond-price").textContent = currency(base.price, 2);
  $("bond-duration").textContent = `${base.modified.toFixed(2)}y`;
  $("bond-convexity").textContent = base.convexity.toFixed(2);
  $("bond-dv01").textContent = currency(base.price * base.modified * 0.0001, 2);
  $("bond-premium").textContent = currency(base.price - face, 2);
  $("bond-price-down").textContent = currency(down.price, 2);
  $("bond-pnl-down").textContent = currency(down.price - base.price, 2);
  $("bond-price-base").textContent = currency(base.price, 2);
  $("bond-price-up").textContent = currency(up.price, 2);
  $("bond-pnl-up").textContent = currency(up.price - base.price, 2);
}

function updateForwards() {
  const spot = number("forward-spot");
  const years = number("forward-years");
  const curveRate = interpolateYield(years) * 100;
  const r = ($("forward-rate-mode").value === "curve" ? curveRate : number("forward-rate")) / 100;
  const income = ($("forward-rate-mode").value === "curve" ? marketData.dividend_yield * 100 : number("income-yield")) / 100;
  const carry = number("carry-cost") / 100;
  const net = r + carry - income;
  const fwd = spot * Math.exp(net * years);
  $("forward-rate").value = $("forward-rate-mode").value === "curve" ? curveRate.toFixed(2) : $("forward-rate").value;
  $("income-yield").value = $("forward-rate-mode").value === "curve" ? (marketData.dividend_yield * 100).toFixed(2) : $("income-yield").value;
  $("forward-price").textContent = currency(fwd, 2);
  $("net-carry").textContent = pct(net * 100, 2);
  $("forward-up").textContent = currency(spot * 1.1 - fwd, 2);
  $("forward-down").textContent = currency(spot * 0.9 - fwd, 2);
  $("forward-breakeven").textContent = currency(fwd, 2);

  const points = [];
  for (let i = 0; i <= 16; i += 1) {
    const terminal = spot * (0.75 + i * 0.04);
    points.push({ x: terminal, y: terminal - fwd });
  }
  drawLineChart($("forward-chart"), points, {
    leftLabel: currency(spot * 0.75, 0),
    rightLabel: currency(spot * 1.39, 0),
    topLabel: "Payoff",
    alt: true,
  });
}

function updateSwaps() {
  const notional = number("swap-notional");
  const fixed = number("fixed-rate") / 100;
  const par = number("par-rate") / 100;
  const discount = number("discount-rate") / 100;
  const years = number("swap-years");
  const frequency = Math.max(1, number("swap-frequency"));
  const periods = Math.round(years * frequency);
  let annuity = 0;
  for (let i = 1; i <= periods; i += 1) {
    annuity += (1 / frequency) * Math.exp(-discount * (i / frequency));
  }
  const spread = par - fixed;
  const value = notional * spread * annuity;
  const dv01 = notional * annuity * 0.0001;
  $("swap-value").textContent = currency(value, 0);
  $("swap-annuity").textContent = annuity.toFixed(3);
  $("swap-spread").textContent = pct(spread * 100, 2);
  $("swap-dv01").textContent = currency(dv01, 0);
  $("swap-shock").textContent = currency(-dv01 * 100, 0);
}

function updateCDS() {
  const notional = number("cds-notional");
  const spread = number("cds-spread") / 10000;
  const hazard = number("cds-hazard") / 100;
  const recovery = number("cds-recovery") / 100;
  const discount = number("cds-discount") / 100;
  const maturity = number("cds-maturity");
  const lgd = 1 - recovery;
  const defaultProb = 1 - Math.exp(-hazard * maturity);
  const discountFactor = Math.exp(-discount * maturity / 2);
  const expectedLossPv = notional * lgd * defaultProb * discountFactor;
  const premiumAnnuity = Array.from({ length: Math.max(1, Math.round(maturity * 4)) }, (_, i) => {
    const t = (i + 1) / 4;
    return 0.25 * Math.exp(-(discount + hazard) * t);
  }).reduce((a, b) => a + b, 0);
  const premiumPv = notional * spread * premiumAnnuity;
  const fairSpread = premiumAnnuity === 0 ? 0 : expectedLossPv / (notional * premiumAnnuity) * 10000;
  const value = expectedLossPv - premiumPv;
  $("cds-value").textContent = currency(value, 0);
  $("cds-el").textContent = currency(expectedLossPv, 0);
  $("cds-premium").textContent = currency(premiumPv, 0);
  $("cds-fair-spread").textContent = `${fairSpread.toFixed(0)} bps`;
  const report = $("report-cds-value");
  if (report) report.textContent = currency(value, 0);
}

function updateSurfacePage() {
  const maturity = number("surface-maturity");
  const moneynessGrid = marketData.vol_surface.moneyness;
  const points = moneynessGrid.map((m) => ({ x: m * 100, y: interpolateVol(maturity, m) * 100 }));
  drawLineChart($("surface-page-chart"), points, {
    leftLabel: "80%",
    rightLabel: "120%",
    topLabel: "Volatility",
    alt: true,
  });
  $("surface-page-label").textContent = `${maturity.toFixed(2)}Y maturity`;
  $("surface-atm-vol").textContent = pct(interpolateVol(maturity, 1) * 100, 2);
  $("surface-vol-80").textContent = pct(interpolateVol(maturity, 0.8) * 100, 2);
  $("surface-vol-100").textContent = pct(interpolateVol(maturity, 1.0) * 100, 2);
  $("surface-vol-120").textContent = pct(interpolateVol(maturity, 1.2) * 100, 2);
}

function updatePortfolio() {
  const equityShock = number("equity-shock");
  const volShock = number("vol-shock");
  const rateShock = number("rate-shock");
  const creditShock = number("credit-shock");
  const equityBook = marketData.portfolio.find((p) => p.book === "Equity Options");
  const ratesBook = marketData.portfolio.find((p) => p.book === "Rates");
  const creditBook = marketData.portfolio.find((p) => p.book === "Credit");
  const equity = equityBook.delta_notional * (equityShock / 100) + equityBook.vega_notional * volShock;
  const rates = ratesBook.dv01 * rateShock;
  const credit = creditBook.cs01 * creditShock;
  const total = equity + rates + credit;
  $("pnl-equity").textContent = currency(equity, 0);
  $("pnl-rates").textContent = currency(rates, 0);
  $("pnl-credit").textContent = currency(credit, 0);
  $("portfolio-pnl").textContent = currency(total, 0);
  drawBarChart($("portfolio-chart"), [
    { label: "Equity", value: equity },
    { label: "Rates", value: rates },
    { label: "Credit", value: credit },
  ]);
}

function updateReport() {
  const option = $("option-price");
  const bond = $("bond-price");
  const structured = $("structured-price");
  const portfolio = $("portfolio-pnl");
  if ($("report-option-price") && option) $("report-option-price").textContent = option.textContent;
  if ($("report-structured-price") && structured) $("report-structured-price").textContent = structured.textContent;
  if ($("report-bond-price") && bond) $("report-bond-price").textContent = bond.textContent;
  if ($("report-portfolio-pnl") && portfolio) $("report-portfolio-pnl").textContent = portfolio.textContent;
  if ($("report-asof")) $("report-asof").textContent = `Sample data as of ${marketData.as_of}`;
}

function updateAll() {
  updateOptions();
  updateBarrier();
  updateAsian();
  updateStructuredAutocallable();
  updateBonds();
  updateForwards();
  updateSwaps();
  updateCDS();
  updatePortfolio();
  updateSurfacePage();
  updateReport();
}

function setPage(name) {
  document.querySelectorAll(".page").forEach((page) => page.classList.toggle("active", page.id === name));
  document.querySelectorAll(".nav-link").forEach((link) => link.classList.toggle("selected", link.dataset.tab === name));
  if (history.replaceState) {
    history.replaceState(null, "", `#${name}`);
  }
}

function setupNavigation() {
  document.querySelectorAll("[data-tab]").forEach((element) => {
    element.addEventListener("click", (event) => {
      event.preventDefault();
      setPage(element.dataset.tab);
    });
  });

  $("toggleMenu").addEventListener("click", () => {
    $("navPanel").classList.toggle("hidden");
    $("toggleMenu").textContent = $("navPanel").classList.contains("hidden") ? "Show Menu" : "Hide Menu";
  });

  if ($("print-report")) {
    $("print-report").addEventListener("click", () => window.print());
  }

  const initial = window.location.hash ? window.location.hash.slice(1) : "overview";
  if ($(initial)) setPage(initial);
}

async function loadMarketData() {
  try {
    const response = await fetch("data/sample_market_data.json", { cache: "no-store" });
    if (response.ok) {
      marketData = await response.json();
    }
  } catch (error) {
    marketData = embeddedMarketData;
  }
}

function setupInputs() {
  document.querySelectorAll("input, select").forEach((input) => {
    input.addEventListener("input", updateAll);
    input.addEventListener("change", updateAll);
  });
}

async function boot() {
  await loadMarketData();
  setupNavigation();
  setupInputs();
  $("spot").value = marketData.spot.toFixed(0);
  $("barrier-spot").value = marketData.spot.toFixed(0);
  $("asian-spot").value = marketData.spot.toFixed(0);
  $("forward-spot").value = marketData.spot.toFixed(0);
  $("rate").value = (marketData.risk_free_rate * 100).toFixed(2);
  $("barrier-rate").value = (marketData.risk_free_rate * 100).toFixed(2);
  $("asian-rate").value = (marketData.risk_free_rate * 100).toFixed(2);
  $("forward-rate").value = (marketData.risk_free_rate * 100).toFixed(2);
  $("income-yield").value = (marketData.dividend_yield * 100).toFixed(2);
  updateAll();
}

boot();
