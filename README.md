<p align="center">
  <img src="https://github.com/user-attachments/assets/54dd4b1b-a332-4109-a768-ca1c4e695fe6" alt="DerivaPro logo" width="300">
</p>

<h1 align="center">DerivaPro</h1>

<p align="center">
  <strong>Financial Instruments Pricing, Evaluation, and Risk Management Platform</strong>
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.x-blue">
  <img alt="Flask" src="https://img.shields.io/badge/Flask-Web%20App-black">
  <img alt="QuantLib" src="https://img.shields.io/badge/QuantLib-Pricing%20Engines-green">
  <img alt="Status" src="https://img.shields.io/badge/Status-Active%20Development-orange">
  <img alt="Scope" src="https://img.shields.io/badge/Scope-Multi--Asset%20Analytics-purple">
</p>

---

## 📌 Overview

**DerivaPro** is a browser-based quantitative finance application for pricing, evaluating, and risk-managing financial instruments. It combines Flask-based workflows, pricing engines, market data extraction, model documentation, scenario analysis, sensitivity analysis, and report-style outputs into one extensible platform.

The current version is a working analytical application with broad instrument coverage. The next development stages focus on turning it into a production-quality platform with stronger security, persistent storage, user-scoped workflows, automated testing, portfolio-level risk management, and professional reporting.

---

## ⚡ DerivaPro Lite Demo

A no-install static demo is available on GitHub Pages:

```text
https://derivapro.github.io/derivapro-v1/demo/
```

The demo source lives under [`demo/`](demo/). It runs entirely in the browser with plain HTML, CSS, and JavaScript.

You can also open the local file directly after cloning the repository:

```text
demo/index.html
```

The Lite demo includes simplified interactive modules for European options, fixed-rate bonds, forwards, swaps, portfolio stress, and packaged sample market data. It is designed for a quick glance and basic user operation only. It does not run the Flask backend, QuantLib, live market data feeds, AI assessment, user persistence, or production report generation.

The full Flask application remains the source for production analytics, market data workflows, model-governance features, and future portfolio-level risk management.

---

## 🎯 Platform Vision

DerivaPro is being developed toward a full browser-launched platform where users can:

| Workflow | Platform Goal |
|---|---|
| Price instruments | Run analytical, lattice, Monte Carlo, and QuantLib-backed valuation workflows. |
| Evaluate models | Review Greeks, sensitivities, convergence behavior, scenarios, and model assumptions. |
| Manage risk | Move from instrument-level analytics to portfolio-level exposure, stress testing, VaR, and risk ladders. |
| Document methodology | Maintain model notes, governance content, monitoring pages, and validation-style documentation. |
| Generate reports | Produce professional outputs for pricing review, model validation, and risk communication. |

---

## ✅ Current Capabilities

DerivaPro already includes active workflows across several major financial product areas.

| Category | Current Coverage |
|---|---|
| **Equity Options** | European and American options, Black-Scholes, binomial/lattice models, Monte Carlo workflows, Greeks, convergence, sensitivity, and scenario analysis. |
| **Exotic Options** | Barrier options, Asian options, and autocallable-related workflows. |
| **Fixed Income** | Non-callable fixed-rate bonds, fixed-rate amortizing bonds, floating-rate bonds, and floating-rate amortizing bonds. |
| **Interest Rate Derivatives** | Swaps, swaptions, term structure analytics, market-rate extraction, and rates API utilities. |
| **Credit Derivatives** | Credit default swaps, synthetic CDO analytics, and credit-linked notes. |
| **Volatility Products** | Volatility surface construction, variance swaps, and volatility swaps. |
| **Futures and Forwards** | Pricing, sensitivity analysis, and scenario-style analysis. |
| **Prepayment Analytics** | Simple calculator-style workflow and v2 data-driven modeling workflow. |
| **AI-Assisted Assessment** | Optional Azure/OpenAI-compatible narrative assessment workflows through environment configuration. |

---

## 🚦 Development Status

| Area | Status | Notes |
|---|---:|---|
| Flask application factory and blueprint architecture | ✅ Done | Modular route registration through `derivapro/routes`. |
| Product-level pricing workflows | ✅ Done / active | Current workflows span options, rates, credit, fixed income, volatility, forwards, and prepayment. |
| Markdown-backed model documentation | ✅ Done / active | Route-level Markdown content supports explanations, governance notes, and user guidance. |
| Environment-based secrets/configuration | 🟡 In progress | `.env.example` exists; configuration should continue moving away from hardcoded values. |
| Structured logging | 🟡 In progress | `logging_config.py` exists; remaining ad hoc logging should be normalized. |
| Monte Carlo modernization | 🟡 In progress | Both legacy and v2 Monte Carlo modules exist; consolidation is a roadmap item. |
| Prepayment modeling workflow | 🟡 In progress | Calculator-style and v2 data-driven tracks exist and need clearer product boundaries. |
| Report generation | 🟡 In progress | Report-style pages exist; production-quality PDF/report generation remains planned. |
| Portfolio-level risk | 🔵 Planned | Current workflows are mostly instrument-level; portfolio aggregation is the next major product step. |
| Database persistence and user identity | 🔵 Planned | Current state is mostly session/file based. |
| Automated tests and CI | 🔵 Planned | Pricing regression tests and route tests are needed before production use. |

---

## 🧭 Development Roadmap

The roadmap follows the development plan for turning DerivaPro from an analytical prototype into a production-grade platform.

### Phase 0: Security and Stability

- Keep credentials and secrets out of source code.
- Use environment-driven configuration for API keys, Flask secrets, debug mode, and logging level.
- Continue replacing ad hoc `print()` calls with structured logging.
- Reduce fixed-name static plot artifacts to avoid concurrent-user collisions.
- Review browser forms for CSRF protection.

### Phase 1: Deployment Foundation

- Add a production WSGI path, such as Gunicorn.
- Add Docker and Docker Compose for reproducible local launch.
- Lock dependencies with a reproducible dependency process.
- Add basic CI for linting and future test execution.

### Phase 2: Persistence and User Workflows

- Add database persistence for users, instruments, pricing results, analysis results, plots, and reports.
- Add authentication and user-scoped sessions.
- Move result passing away from browser session storage and toward database-backed result IDs.

### Phase 3: Maintainability and Service Layer

- Split large route handlers into smaller action-specific functions.
- Introduce a service layer for pricing, analysis, market data, and AI assessment.
- Cache repeated market data calls.
- Consolidate legacy/v2 modules where practical.
- Add type hints to the model layer.

### Phase 4: Testing and Model Regression

- Add unit tests for Black-Scholes, lattice, Monte Carlo, bond, swap, and credit models.
- Add Flask route integration tests.
- Add benchmark pricing tests, convergence checks, and put-call-parity tests.
- Use CI to prevent accidental pricing regressions.

### Phase 5: Portfolio-Level Risk Management

- Add portfolios and positions.
- Aggregate Greeks and exposures across positions.
- Add portfolio stress testing.
- Add historical, parametric, and Monte Carlo VaR/CVaR.
- Add DV01 and tenor risk ladders for rate-sensitive books.

### Phase 6: Reporting and Model Governance

- Build production-quality report generation.
- Store generated reports with pricing and analysis metadata.
- Expand model governance workflows.
- Complete prepayment model validation and monitoring.

### Phase 7: Performance and User Experience

- Move long-running analytics to background tasks.
- Add task progress/status tracking for heavy simulations and model training.
- Improve UI consistency and responsive layout.
- Add stronger client-side validation and clearer error states.

---

## 🧩 Planned Instrument Extensions

The following additions align with the long-term multi-asset platform vision.

| Priority | Category | Candidate Additions | Why It Fits |
|---:|---|---|---|
| 1 | **FX** | FX forwards, FX vanilla options, FX barrier options, cross-currency swaps | Reuses existing forwards, options, and swaps patterns. |
| 2 | **Interest Rate Options** | Caps, floors, collars, CMS products, OIS/SOFR swaps | Builds on term structure and swaption infrastructure. |
| 3 | **Fixed Income Extensions** | Callable bonds, putable bonds, TIPS/inflation-linked bonds, convertible bonds | Extends current bond analytics into more realistic desk workflows. |
| 4 | **Structured Products** | Reverse convertibles, principal-protected notes, CPPI structures | Combines option, credit, and fixed-income components already present. |
| 5 | **Additional Exotics** | Digital/binary options, lookback options, Bermuda options, spread options, quanto options | Expands derivatives coverage after core engines are tested. |
| 6 | **XVA** | CVA, DVA, FVA, MVA | Adds OTC valuation adjustment and counterparty risk capabilities. |
| 7 | **Securitized Products** | MBS, ABS, CLO-style workflows | Longer-term extension connected to the prepayment modeling work. |
| 8 | **Commodities** | Commodity forwards, futures options, energy derivatives | Extends the platform toward broader multi-asset coverage. |

---

## 🏗️ Architecture

DerivaPro uses a conventional Flask application structure with a clear separation between routes, models, templates, and static assets.

```text
derivapro-v1/
|-- README.md
|-- requirements.txt
|-- setup.py
|-- run.py
`-- derivapro/
    |-- __init__.py
    |-- config.py
    |-- logging_config.py
    |-- secret_key_utils.py
    |-- routes/
    |-- models/
    |-- templates/
    |-- static/
    `-- legacy/
```

### Main application areas

| Path | Purpose |
|---|---|
| `derivapro/routes/` | Flask blueprints, page handlers, and workflow endpoints. |
| `derivapro/models/` | Pricing engines, market data helpers, analytics logic, and model utilities. |
| `derivapro/templates/` | Jinja templates for browser-facing pages. |
| `derivapro/static/` | CSS, images, generated plots, and generated artifacts. |
| `derivapro/legacy/` | Retired implementation artifacts retained for reference. |

### Key model modules

| Module | Focus |
|---|---|
| `market_data.py` | Market data helpers. |
| `mdls_vanilla_options.py` | Black-Scholes style pricing and Greeks. |
| `mdls_lattice_trees.py` / `mdls_binomial_tree.py` | Lattice and binomial option models. |
| `mdls_monte_carlo.py` / `mdls_monte_carlo_v2.py` | Monte Carlo pricing and simulation engines. |
| `mdls_bonds.py` | Fixed-income analytics. |
| `mdls_credit.py` | Credit derivatives analytics. |
| `mdls_swaps.py` / `swaps.py` / `swaptions.py` | Rates and swap analytics. |
| `mdls_term_structure.py` | Yield curve and term structure modeling. |
| `mdls_prepayment.py` / `mdls_prepayment_v2.py` | Prepayment workflows. |

---

## ⚙️ Setup and Local Development

### 1. Create and activate a virtual environment

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a local `.env` file in the project root. Do not commit `.env`.

Use `.env.example` as the starting point:

```env
OpenAI_API_Key="xxx"
Base_URL="https://your-azure-openai-endpoint"
Model="your-model-name"
API_Version="2025-04-01-preview"
Auth_headers="your-subscription-key-header"
FRED_API_Key="xxx"
SECRET_KEY="replace_with_a_secure_random_value"
FLASK_ENV=development
FLASK_DEBUG=true
LOG_LEVEL=INFO
```

To generate a local secret key:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### 4. Run the app

```bash
python run.py
```

Then open the local Flask URL shown in the terminal, typically:

```text
http://127.0.0.1:5000
```

---

## 🔌 External Services

DerivaPro can use external services for market data and AI-assisted assessments:

| Service | Usage |
|---|---|
| Yahoo Finance / `yfinance` | Equity and option-market data workflows. |
| FRED / Treasury / SOFR data | Rates and term-structure workflows. |
| Azure/OpenAI-compatible APIs | Optional narrative assessments and model commentary. |

External service availability, credentials, quotas, and network access can affect runtime behavior.

---

## 🛡️ Operational Notes

This repository is actively evolving. Before treating it as production-ready, complete at least the following:

- Verify no credentials or secrets are committed.
- Add CSRF protection for browser forms.
- Add persistent storage and user-level isolation.
- Add automated pricing regression tests.
- Replace development-server deployment with a WSGI deployment path.
- Review generated static artifacts and cleanup rules.

---

## 🤝 Contributing Notes

For low-risk development:

- Make one focused change at a time.
- Preserve existing pricing behavior unless intentionally changing it.
- Update routes, models, templates, and documentation together when renaming workflows.
- Treat `legacy/` and active pricing modules carefully before deleting older code.
- Add tests when changing pricing logic, market data logic, or model assumptions.

---

## 🧠 Product Direction

The long-term goal is a multi-asset pricing and risk platform that users can launch in a browser, configure through forms, save instruments and portfolios, run analyses, and produce professional model/risk reports.

The next major milestone is to move from instrument-level analytics to persistent, user-scoped, portfolio-level risk management.
