const $ = (id) => document.getElementById(id);

const defaults = {
  option: { type: "call", spot: 100, strike: 100, rate: 5, volatility: 20, maturity: 1 },
  bond: { face: 1000, coupon: 5, ytm: 4.5, years: 5, frequency: 2 },
  forward: { spot: 100, rate: 5, income: 2, carry: 0, years: 1 },
  swap: { notional: 1000000, fixed: 4.2, par: 4.5, discount: 4.4, years: 5, frequency: 2 },
  portfolio: { equity: -10, vol: 5, rate: 50, credit: 75 },
};

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
    <path class="line" d="${path}"></path>
    <text class="chart-label" x="${pad}" y="${height - 8}">${options.leftLabel || minX.toFixed(0)}</text>
    <text class="chart-label" x="${width - pad - 30}" y="${height - 8}">${options.rightLabel || maxX.toFixed(0)}</text>
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
  svg.innerHTML = `
    <line class="axis" x1="${pad}" y1="${zero}" x2="${width - pad}" y2="${zero}"></line>
    ${nodes}
  `;
}

function updateOptions() {
  const type = $("option-type").value;
  const spot = number("spot");
  const strike = number("strike");
  const rate = number("rate");
  const vol = number("volatility");
  const maturity = number("maturity");
  const result = blackScholes(type, spot, strike, rate, vol, maturity);
  $("option-price").textContent = currency(result.price, 2);
  $("delta").textContent = result.delta.toFixed(4);
  $("gamma").textContent = result.gamma.toFixed(4);
  $("vega").textContent = result.vega.toFixed(4);
  $("theta").textContent = result.theta.toFixed(4);
  $("rho").textContent = result.rho.toFixed(4);
  $("moneyness").textContent = pct((spot / strike) * 100, 1);
  $("hero-option-price").textContent = currency(result.price, 2);

  const points = [];
  for (let i = 0; i <= 16; i += 1) {
    const s = spot * (0.8 + i * 0.025);
    points.push({ x: s, y: blackScholes(type, s, strike, rate, vol, maturity).price });
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

function updateBonds() {
  const face = number("bond-face");
  const coupon = number("coupon-rate");
  const ytm = number("ytm");
  const years = number("bond-years");
  const frequency = Math.max(1, number("bond-frequency"));
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
  $("hero-duration").textContent = `${base.modified.toFixed(2)}y`;
}

function updateForwards() {
  const spot = number("forward-spot");
  const r = number("forward-rate") / 100;
  const income = number("income-yield") / 100;
  const carry = number("carry-cost") / 100;
  const years = number("forward-years");
  const net = r + carry - income;
  const fwd = spot * Math.exp(net * years);
  $("forward-price").textContent = currency(fwd, 2);
  $("net-carry").textContent = pct(net * 100, 2);
  $("forward-up").textContent = currency(spot * 1.1 - fwd, 2);
  $("forward-down").textContent = currency(spot * 0.9 - fwd, 2);
  $("forward-breakeven").textContent = currency(fwd, 2);
  $("hero-forward").textContent = currency(fwd, 2);

  const points = [];
  for (let i = 0; i <= 16; i += 1) {
    const terminal = spot * (0.75 + i * 0.04);
    points.push({ x: terminal, y: terminal - fwd });
  }
  drawLineChart($("forward-chart"), points, {
    leftLabel: currency(spot * 0.75, 0),
    rightLabel: currency(spot * 1.39, 0),
    topLabel: "Payoff",
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

function updatePortfolio() {
  const equityShock = number("equity-shock");
  const volShock = number("vol-shock");
  const rateShock = number("rate-shock");
  const creditShock = number("credit-shock");
  const equity = 1850 * equityShock + 1200 * volShock;
  const rates = -134 * rateShock;
  const credit = -42.9 * creditShock;
  const total = equity + rates + credit;
  $("pnl-equity").textContent = currency(equity, 0);
  $("pnl-rates").textContent = currency(rates, 0);
  $("pnl-credit").textContent = currency(credit, 0);
  $("portfolio-pnl").textContent = currency(total, 0);
  $("hero-stress-pnl").textContent = currency(total, 0);
  drawBarChart($("portfolio-chart"), [
    { label: "Equity", value: equity },
    { label: "Rates", value: rates },
    { label: "Credit", value: credit },
  ]);
}

function setTab(name) {
  document.querySelectorAll(".tab").forEach((tab) => {
    const active = tab.dataset.tab === name;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === name);
  });
}

function resetDemo() {
  $("option-type").value = defaults.option.type;
  $("spot").value = defaults.option.spot;
  $("strike").value = defaults.option.strike;
  $("rate").value = defaults.option.rate;
  $("volatility").value = defaults.option.volatility;
  $("maturity").value = defaults.option.maturity;

  $("bond-face").value = defaults.bond.face;
  $("coupon-rate").value = defaults.bond.coupon;
  $("ytm").value = defaults.bond.ytm;
  $("bond-years").value = defaults.bond.years;
  $("bond-frequency").value = defaults.bond.frequency;

  $("forward-spot").value = defaults.forward.spot;
  $("forward-rate").value = defaults.forward.rate;
  $("income-yield").value = defaults.forward.income;
  $("carry-cost").value = defaults.forward.carry;
  $("forward-years").value = defaults.forward.years;

  $("swap-notional").value = defaults.swap.notional;
  $("fixed-rate").value = defaults.swap.fixed;
  $("par-rate").value = defaults.swap.par;
  $("discount-rate").value = defaults.swap.discount;
  $("swap-years").value = defaults.swap.years;
  $("swap-frequency").value = defaults.swap.frequency;

  $("equity-shock").value = defaults.portfolio.equity;
  $("vol-shock").value = defaults.portfolio.vol;
  $("rate-shock").value = defaults.portfolio.rate;
  $("credit-shock").value = defaults.portfolio.credit;

  updateAll();
}

function updateAll() {
  updateOptions();
  updateBonds();
  updateForwards();
  updateSwaps();
  updatePortfolio();
}

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => setTab(tab.dataset.tab));
});

document.querySelectorAll("[data-jump]").forEach((button) => {
  button.addEventListener("click", () => {
    setTab(button.dataset.jump);
    document.querySelector(".workspace").scrollIntoView({ behavior: "smooth", block: "start" });
  });
});

document.querySelectorAll("input, select").forEach((input) => {
  input.addEventListener("input", updateAll);
  input.addEventListener("change", updateAll);
});

$("reset-demo").addEventListener("click", resetDemo);

updateAll();
