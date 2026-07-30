# DerivaPro
## Development Plan
*Financial Instruments Pricing, Evaluation & Risk Management Platform*

May 18, 2026

## 1. Executive Summary

DerivaPro is a Flask-based quantitative analytics platform with broad coverage across derivatives and structured products. The current version includes vanilla and exotic options (Black-Scholes, binomial trees, Monte Carlo with Sobol sequences and LSMC), interest rate derivatives (swaps, swaptions), credit derivatives (CDS, synthetic CDO, CLN), non-callable bonds, volatility surfaces, term structure modeling, and a machine-learning prepayment probability pipeline with Azure OpenAI assessment integration.

This development plan transforms DerivaPro from a working prototype into a production-grade, browser-based platform where financial professionals can price instruments, run model validation workflows, and manage risk across a multi-asset portfolio. The plan is organized into seven phases over approximately 26 weeks.

## 2. Current Instrument Coverage

The table below summarises the financial instruments currently supported in DerivaPro.

| Category | Instruments Currently Supported |
| --- | --- |
| Equity Options | European Options (Black-Scholes, CRR, Jarrow-Rudd, Trinomial, Monte Carlo); American Options (CRR, JR, Trinomial, LSMC Monte Carlo, Binomial CRR with dividends) |
| Exotic Options | Barrier Options; Asian Options (QuantLib); Autocallable Notes (Monte Carlo) |
| Fixed Income | Non-Callable Fixed Rate Bonds; Fixed Rate Amortizing Bonds; Floating Rate Bonds; Floating Rate Amortizing Bonds |
| Interest Rate Derivatives | Plain Vanilla Swaps (analytical + mark-to-market); European Swaptions (Black / Bachelier / Hull-White MC); Futures; Forwards |
| Credit Derivatives | Credit Default Swaps (QuantLib); Synthetic CDO; Credit-Linked Notes (fixed and floating) |
| Volatility Products | Variance Swaps; Volatility Swaps; Implied Volatility Surface |
| Market Curves | Yield Term Structure (bootstrapping, Nelson-Siegel, Svensson); SOFR / Treasury / FRED market data |
| Prepayment Models | Logistic Regression prepayment probability; ML pipeline (Random Forest, GBM, LightGBM) with feature selection and model registry |

## 3. Proposed New Instrument Categories

The following financial instruments and categories are recommended for addition in future development phases. Each builds on existing pricing engines, minimising duplication of effort.

### 3.1 FX Products

Foreign exchange products are a natural extension of the existing futures/forwards and swaps engines. Proposed additions:

- FX Forwards and FX Futures — extend the existing Forwards model with FX carry / interest rate parity
- FX Vanilla Options — Garman-Kohlhagen model, a direct extension of Black-Scholes
- FX Barrier and Asian Options — reuse the existing Monte Carlo engine with an FX spot process
- FX Swaps and Cross-Currency Basis Swaps — extend the Swaps model with dual-currency discounting

### 3.2 Interest Rate Options

Closely related to the swaptions engine already in the platform:

- Interest Rate Caps and Floors — collections of caplets/floorlets priced with Black or Bachelier models
- Interest Rate Collars — cap + floor combination
- Constant Maturity Swaps (CMS) and CMS Spread Options
- Overnight Index Swaps (OIS) with SOFR discounting — SOFR data infrastructure is already present

### 3.3 Fixed Income Extensions

- Callable and Putable Bonds — extend the QuantLib bond engine with embedded optionality via interest rate trees
- Inflation-Linked Bonds / TIPS — real yield and breakeven inflation analytics
- Mortgage-Backed Securities (MBS) and Asset-Backed Securities (ABS) — natural companion to the existing prepayment model
- Convertible Bonds — equity + fixed income hybrid, priced with binomial tree

### 3.4 Additional Exotic Options

- Digital / Binary Options — cash-or-nothing and asset-or-nothing, analytical and Monte Carlo
- Lookback Options — floating and fixed strike, analytical Conze-Viswanathan and Monte Carlo
- Bermuda Options — early exercise on discrete dates, extend the existing lattice tree engine
- Spread Options — Margrabe's formula and Monte Carlo for two-asset spreads
- Compound Options — options on options
- Quanto Options — payoff in a different currency

### 3.5 Structured Products

- Reverse Convertibles — bond + short put; common retail structured product
- Principal-Protected Notes — bond + call option structure
- Equity-Linked Notes and Deposits
- CPPI (Constant Proportion Portfolio Insurance) — dynamic allocation strategy with payoff simulation
- Leveraged Notes and Capital-at-Risk Products

### 3.6 XVA — Valuation Adjustments

Increasingly required for OTC derivatives pricing and risk management. The existing Monte Carlo engine provides the simulation infrastructure; the credit model provides the hazard rate framework.

- CVA (Credit Valuation Adjustment) — cost of counterparty default risk
- DVA (Debit Valuation Adjustment) — own-credit benefit
- FVA (Funding Valuation Adjustment) — cost of funding uncollateralized positions
- MVA (Margin Valuation Adjustment) — cost of posting initial margin

### 3.7 Commodity Derivatives

- Commodity Forwards and Futures — extend the cost-of-carry model with convenience yield and storage costs
- Commodity Options — Black-76 model for options on futures
- Energy Derivatives — electricity, natural gas; mean-reverting spot price processes

## 4. Development Phases

### Phase 0 — Security and Stability (Weeks 1–2)

These items are blockers. Nothing else ships until they are resolved.

**0.1 Rotate and externalize credentials**

An Azure OpenAI API key is currently hardcoded in source code and is therefore present in the git commit history. Steps: (1) rotate the key immediately in the Azure portal; (2) move all credentials to a .env file loaded via python-dotenv; (3) add .env to .gitignore; (4) create a .env.example template with placeholder values. Apply the same treatment to the Azure endpoint, API version, and model identifier. `Remove any api keys`

**0.2 Secure the Flask secret key**

config.py currently contains SECRET_KEY = 'your_secret_key'. Flask signs session cookies with this value; anyone who knows it can forge sessions. Replace with a cryptographically random 32-byte value stored in the environment. Introduce a proper config hierarchy with DevelopmentConfig and ProductionConfig classes.

**0.3 Gate debug mode on environment**

Remove debug=True from run.py and the application factory. Control via the FLASK_ENV environment variable so that the Werkzeug interactive debugger is never exposed in production.

**0.4 Rename _NEW files and remove importlib workarounds**

`mdls_monte_carlo_NEW.py and mdls_binomial_tree_model_NEW.py are loaded via importlib.util.spec_from_file_location because their filenames cannot be imported as standard Python modules. Rename them (e.g. mdls_monte_carlo_v2.py, mdls_binomial_crr.py) and replace all importlib blocks with standard imports. Retire the old mdls_monte_carlo.py and mdls_prepayment.py by merging any unique logic into their v2 counterparts.`

**0.5 Replace print() with structured logging**

Create a logging configuration module. Remove all print() calls across routes and models — especially the statement that prints the API key inside ask_gpt(). Use Python's standard logging module with level controlled by a LOG_LEVEL environment variable.

**0.6 Fix concurrent-user plot filename collision**

`Routes currently save plots with fixed filenames (e.g. vanilla_convergence_plot.png). Two simultaneous users overwrite each other's plots. Apply the UUID-based naming scheme already used in the term structure routes to all other plot-generating routes.`

**Deliverable:** App runs in development with no secrets in source, no importlib hacks, structured logging, and concurrent-user plot safety.

### Phase 1 — Deployment Infrastructure (Weeks 3–4)

**1.1 Add a production WSGI server**

Replace the development app.run() entry point with gunicorn. Add to requirements.txt and provide a Procfile: web: gunicorn "derivapro:create_app()" --workers 4 --timeout 120. The 120-second timeout accommodates long-running Monte Carlo analyses.

**1.2 Dockerize the application**

Write a Dockerfile and docker-compose.yml. The compose file defines at minimum the Flask/gunicorn container and (after Phase 2) a PostgreSQL container. Goal: docker compose up launches the full stack on any machine.

**1.3 Pin dependency versions**

Switch requirements.txt from floor-only pins (>=) to exact versions generated by pip-compile. Add lightgbm, which is used in the prepayment module but missing from requirements. This makes deployments reproducible.

**1.4 Continuous Integration pipeline**

Add a GitHub Actions workflow that runs on every push: install dependencies, run linting (ruff or flake8), and — after Phase 4 — run the test suite. A passing CI badge signals the branch is deployable.

**Deliverable:** docker compose up launches the app on any machine. CI catches regressions automatically.

### Phase 2 — Data Persistence and Authentication (Weeks 5–7)

Currently the application has no database. Results live in Flask sessions (lost on tab close) and plot files on disk (overwritten by concurrent users).

**2.1 Add a relational database**

Use SQLAlchemy with SQLite for development and PostgreSQL for production, configurable via the DATABASE_URL environment variable. Initial schema:

| Table | Purpose |
| --- | --- |
| users | Identity, password hash, role |
| instruments | Saved instrument configurations (ticker, model params, dates) |
| pricing_results | Computed prices and Greeks, linked to an instrument and user |
| analysis_results | Sensitivity, scenario, and convergence outputs |
| plots | Plot metadata (filename, type) linked to analysis results |
| reports | Generated report metadata and file paths |
| portfolios | Named collections of positions per user |
| positions | Individual instrument positions with notional / quantity |

**2.2 User authentication**

Implement username/password authentication using Flask-Login and Flask-Bcrypt. Sessions are scoped per user so one user's results never interfere with another's. HTTPS via an nginx reverse proxy or cloud load balancer is required in production.

**2.3 Migrate the model registry to the database**

The prepayment ML pipeline saves trained models as pickle files to a local folder. Store model metadata in the database and model binary files at a configurable path (local in development, S3-compatible object storage in production). Replace pickle with joblib for safer scikit-learn serialization.

**2.4 Replace session-based result passing**

Form data and results are currently passed between pages via Flask session and URL query parameters. With a database, results are written to pricing_results and retrieved by ID. The session only needs to hold a last_result_id reference.

**Deliverable:** Multiple users can run analyses simultaneously without collision. Results persist across browser sessions. Users can return to previously saved instrument configurations.

### Phase 3 — Code Refactoring (Weeks 8–10)

**3.1 Decompose large route handlers**

The american_options and european_options view functions are each approximately 900+ lines long. Each action block becomes its own function, and a dispatch table replaces the if/elif chain. Near-identical sensitivity and scenario logic across routes is consolidated into shared helpers in a new derivapro/services/analysis.py module.

**3.2 Introduce a service layer**

Routes should not contain business logic. Add a derivapro/services/ directory:

- services/pricing.py — calls the correct pricing model, returns a standardised result dict
- services/analysis.py — sensitivity, scenario, and convergence analysis logic
- services/market_data.py — wraps StockData with caching
- services/ai_assessment.py — wraps the Azure OpenAI call with retry and error handling

**3.3 Cache market data fetches**

Wrap yfinance and FRED API calls with Flask-Caching (memory backend in development, Redis in production) keyed on ticker, start date, and end date with a configurable TTL. This eliminates duplicate API calls within a single request and across sensitivity analysis loops.

**3.4 Fix SmoothnessTest inheritance**

SmoothnessTest in mdls_vanilla_options.py has an identical __init__ to BlackScholes and creates BlackScholes instances internally. It should subclass BlackScholes directly.

**3.5 Add type hints to models**

Add Python type annotations to the model layer, beginning with BlackScholes, LatticeModel, and the Monte Carlo engine. Run mypy in strict mode as part of CI.

**3.6 Remove dead code**

Several hundred lines of commented-out code exist in the route files. Git preserves history — delete them cleanly.

**3.7 Add CSRF protection**

Add Flask-WTF's CSRFProtect to the application factory. This is a one-line addition that automatically protects all POST forms.

**Deliverable:** No view function longer than ~100 lines. Service layer is independently testable. Type checker and CSRF protection pass in CI.

### Phase 4 — Test Suite (Weeks 10–12)

**4.1 Pricing model unit tests**

| Test | What It Checks |
| --- | --- |
| Black-Scholes call/put against closed-form reference values | Pricing correctness |
| Put-call parity: C - P = S - K·exp(-rT) across all models | Internal consistency |
| Delta in (0,1) for calls; gamma > 0; vega > 0 | Greek sign and magnitude sanity |
| Monte Carlo converges to Black-Scholes as num_paths increases | MC engine calibration |
| CRR tree converges to Black-Scholes as num_steps increases | Lattice convergence |
| CDS fair spread round-trip pricing | Credit model correctness |
| Bond price equals par when coupon rate equals YTM | Fixed income basic validation |

**4.2 Route integration tests**

Use Flask's test client to verify that each route returns HTTP 200 for valid inputs and a structured error response (not a 500) for invalid inputs such as an unrecognised ticker, a malformed date, or a missing required field.

**4.3 Prepayment ML pipeline tests**

Verify that model_training returns the expected metrics structure, that train/test split parameters are preserved across metadata serialisation/deserialisation, and that register_model and deregister_model leave the registry in a consistent state.

**Deliverable:** pytest suite with coverage above 80% on the models layer. CI runs the suite on every push and blocks merges on failure.

### Phase 5 — Risk Management Features (Weeks 13–18)

This phase delivers the core platform value: aggregation across a portfolio.

**5.1 Portfolio and Position Manager**

A portfolio is a named collection of saved instruments. Add a Portfolio page where users can add any priced instrument as a position with notional/quantity, view positions, and delete or modify them. Requires the portfolios and positions database tables from Phase 2.

**5.2 Aggregated Greeks Dashboard**

A portfolio summary page showing net Delta, Gamma, Vega, Theta, and Rho across all positions, grouped by asset class. A bar chart of the largest Greek contributors shows a risk manager immediately where exposure is concentrated.

**5.3 Value at Risk (VaR) and CVaR**

Add a RiskEngine service that computes:

- Historical VaR — reprice all positions under the last N days of observed market moves
- Parametric VaR — portfolio Greeks + variance-covariance matrix of risk factors (delta-gamma approximation)
- Monte Carlo VaR — simulate forward P&L distribution using the existing Monte Carlo engine

Report VaR at configurable confidence levels (95%, 99%) and holding periods (1-day, 10-day).

**5.4 Portfolio-Level Stress Testing**

Extend the existing per-instrument scenario analysis to the full portfolio. A user defines a scenario (e.g. rates +200bps, equities -20%, credit spreads +100bps, volatility +10%), applies it simultaneously to all positions, and sees total portfolio P&L impact with a breakdown by position and asset class.

**5.5 DV01 / Risk Ladder**

For rate-sensitive portfolios (bonds, swaps, swaptions), generate a DV01 ladder: the sensitivity of portfolio value to a 1 basis point move at each tenor on the yield curve. The YieldTermStructure model already provides the underlying curve.

**5.6 CVA / DVA — Initial Implementation**

Using the existing credit model (hazard rates from the CDS pricer) and the Monte Carlo simulation engine, compute Credit Valuation Adjustment and Debit Valuation Adjustment for OTC derivative positions. This forms the foundation for the full XVA suite.

**Deliverable:** Users can manage a multi-instrument portfolio, view aggregated Greeks, run portfolio-level stress tests, and compute VaR and initial CVA/DVA.

### Phase 6 — Reporting and Model Governance (Weeks 18–22)

**6.1 Implement proper PDF report generation**

The current download_report route produces a nearly blank PDF. Replace with a template-driven pipeline: collect pricing results, sensitivity plots, scenario tables, AI assessments, and model governance text into a ReportTemplate dataclass, then render to PDF using WeasyPrint or a structured ReportLab template. Store generated PDFs in the database with a download link.

**6.2 Complete the prepayment model validation workflow**

Implement the currently-stubbed ValidationPerformanceTesting class:

- Classification metrics: AUC-ROC, Gini coefficient, KS statistic
- Regression metrics: RMSE, MAE, R-squared
- Stability tests: Population Stability Index (PSI) to detect data drift
- Backtesting: compare model-predicted prepayment probabilities against realised rates

**6.3 Cross-validation for the ML pipeline**

Replace the single train/test split with k-fold cross-validation. For time-series financial data, offer out-of-time validation: train on earlier periods, test on later periods. Report mean and standard deviation of each metric across folds.

**6.4 Model Governance workflow**

Wire the existing model_governance.md files into a structured governance lifecycle:

- Conceptual soundness documentation and AI-assisted review
- Model performance benchmarks with pass/fail thresholds
- Ongoing monitoring dashboard: distribution shift detection over time
- Sign-off and attestation records stored in the database with timestamp and user identity
- Full audit trail of model versions, validation results, and approvals

**Deliverable:** Users can run a complete model validation lifecycle — train, validate, govern, register, and generate a formal report — without leaving the browser.

### Phase 7 — Performance and UX (Weeks 22–26)

**7.1 Async task queue for long-running computations**

Monte Carlo with many paths, lattice trees with many steps, and ML model training can each take 10–60 seconds. Add Celery with a Redis broker: long-running analyses submit a task and return a task ID immediately; the frontend polls a /task-status/<id> endpoint or receives a push notification via Flask-SocketIO. A progress bar replaces the blank wait.

**7.2 Scheduled market data refresh**

Add a Celery Beat job that refreshes cached market data (yield curves, volatility surfaces, equity spot prices) on a configurable schedule. Display a 'data as of' timestamp on all pricing pages with a manual refresh button.

**7.3 UI modernisation**

- Migrate from the current custom CSS to Bootstrap 5 or Tailwind CSS for a responsive layout
- Add client-side form validation to reduce unnecessary server round-trips
- Fix the malformed navigation HTML in base.html (unclosed li tag near the Prepayment section)
- Add breadcrumb navigation for deep pages (e.g. Products > Options > Vanilla Options > European)

**7.4 Excel and CSV export**

Add export buttons to all results tables. Pricing results, Greeks, sensitivity tables, scenario outputs, and risk reports should be downloadable as Excel (.xlsx) or CSV using openpyxl. High value, low effort.

**7.5 REST API layer**

Expose a REST API (Flask-RESTful or FastAPI as a companion service) so pricing and risk engines can be called programmatically from Excel, Python scripts, or other internal systems. Authenticate with API keys stored in the database.

**Deliverable:** Concurrent users without blocking. Long analyses show progress feedback. Responsive UI. Results exportable. Pricing engines accessible via API.

## 5. Instrument Addition Roadmap

The table below maps each proposed new instrument to a development phase, its primary dependency on existing code, and a relative effort estimate.

| Phase | Instrument / Category | Key Dependency | Effort |
| --- | --- | --- | --- |
| 5 | FX Forwards and FX Options (Garman-Kohlhagen) | Extend existing Forwards engine | Low |
| 5 | Interest Rate Caps and Floors | Extend Swaptions engine | Low |
| 5 | Interest Rate Collars | Caps + Floors (above) | Very Low |
| 5 | Overnight Index Swaps (OIS / SOFR) | SOFR data already integrated | Low |
| 5 | Digital / Binary Options | Extend MC + Black-Scholes engines | Low |
| 6 | Callable and Putable Bonds | QuantLib bond engine + lattice | Medium |
| 6 | FX Barrier and Asian Options | MC engine with FX spot process | Medium |
| 6 | Cross-Currency Swaps | Swaps engine, dual-currency discount | Medium |
| 6 | Lookback Options | Conze-Viswanathan + Monte Carlo | Medium |
| 6 | Bermuda Options | Extend lattice tree engine | Medium |
| 6 | Spread Options | Margrabe's formula + two-asset MC | Medium |
| 6 | Inflation-Linked Bonds (TIPS) | Real yield / breakeven analytics | Medium |
| 7 | CVA / DVA (full XVA) | Phase 5 CVA + credit model | High |
| 7 | Reverse Convertibles | Bond + short put engines | Medium |
| 7 | Principal-Protected Notes | Bond + call option engines | Medium |
| 7 | MBS / ABS | Extend prepayment to cash flow projection | High |
| 7 | Convertible Bonds | Equity + fixed income, binomial tree | High |
| 7 | Commodity Options (Black-76) | Options on futures, extend BS | Low |
| 7 | Commodity Forwards / Futures | Cost-of-carry + convenience yield | Low |
| Future | CPPI | Dynamic allocation simulation | Medium |
| Future | Compound Options | Options on options, analytical | Medium |
| Future | Quanto Options | Cross-currency payoff adjustment | Medium |
| Future | CLO (Collateralised Loan Obligation) | Complex structuring, extend CDO | Very High |

## 6. Technology Stack

| Layer | Technology |
| --- | --- |
| Web Framework | Flask (current); gunicorn for production WSGI |
| Database | SQLAlchemy ORM; SQLite (development); PostgreSQL (production) |
| Authentication | Flask-Login, Flask-Bcrypt |
| Caching | Flask-Caching; Redis (production) |
| Async Tasks | Celery with Redis broker; Celery Beat for scheduled jobs |
| Quantitative Models | NumPy, SciPy, QuantLib, scikit-learn, LightGBM |
| Market Data | yfinance, FRED API (fredapi), Treasury API |
| AI Assessment | Azure OpenAI (GPT-4o) |
| PDF Generation | WeasyPrint or ReportLab |
| Excel Export | openpyxl |
| Testing | pytest, Flask test client, mypy |
| CI/CD | GitHub Actions |
| Containerisation | Docker, docker-compose |
| Frontend | Bootstrap 5 or Tailwind CSS; MathJax (current, keep) |
| Security | Flask-WTF (CSRF); python-dotenv; Flask-Talisman (HTTP headers) |

## 7. Summary Timeline

| Phase | Name | Weeks | Key Deliverable |
| --- | --- | --- | --- |
| 0 | Security and Stability | 1–2 | No secrets in source; structured logging; plot safety |
| 1 | Deployment Infrastructure | 3–4 | Docker launch; CI pipeline |
| 2 | Data Persistence and Auth | 5–7 | Multi-user database; login; persistent results |
| 3 | Code Refactoring | 8–10 | Thin routes; service layer; type hints; CSRF |
| 4 | Test Suite | 10–12 | 80%+ model coverage; CI-gated merges |
| 5 | Risk Management Features | 13–18 | Portfolio aggregation; VaR; stress testing; CVA |
| 6 | Reporting and Model Governance | 18–22 | PDF reports; model validation lifecycle; attestation |
| 7 | Performance and UX | 22–26 | Async tasks; Excel export; REST API; responsive UI |

*Phases 0–1 make the app deployable. Phases 2–4 make it maintainable and multi-user. Phases 5–7 deliver platform-level value. Each phase produces a usable increment. Phase 0 is the only hard prerequisite before any other work begins.*
