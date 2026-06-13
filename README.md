<p align="center">
<img src="https://github.com/user-attachments/assets/54dd4b1b-a332-4109-a768-ca1c4e695fe6" alt="DerivaPro logo_final" width="300">
</p>

# DerivaPro

DerivaPro is a Flask-based quantitative analytics application for derivative pricing, risk analysis, market data extraction, and model-validation style reporting. The platform exposes multiple pricing and analytics workflows through modular Flask blueprints, with supporting pricing engines implemented under the `derivapro/models` package.

The application combines:
- pricing workflows for vanilla, exotic, credit, rates, and volatility products
- sensitivity, scenario, convergence, and risk-based P&L style analyses
- market data retrieval and derived analytics
- Markdown-backed explanatory content for model documentation and user guidance
- optional AI-assisted narrative assessments through Azure OpenAI configuration supplied via environment variables

---

## Core Capabilities

### Product and analytics coverage

The current codebase includes routes, templates, and model support for:
- **Vanilla options**
  - European option pricing
  - American option pricing
  - sensitivity analysis
  - convergence analysis
  - scenario analysis
  - reporting and model governance views
- **Exotic options**
  - barrier options
  - Asian options
  - autocallable-related workflows
- **Futures and forwards**
  - pricing and sensitivity analysis
- **Credit derivatives**
  - credit default swaps
  - synthetic CDO analytics
  - credit-linked notes
- **Rates and term structure**
  - swaps and swaptions
  - term structure extraction and analytics
  - rates-related API utilities
- **Volatility products**
  - volatility surface construction
  - variance and volatility swap analytics
- **Prepayment analytics**
  - a simpler calculator-style workflow
  - a separate v2 data-driven workflow
- **Bond analytics**
  - non-call fixed and floating bond variants, including amortizing structures

### Cross-cutting features

Across many of the above workflows, the platform supports:
- option Greeks and related sensitivities
- plot generation and rendering in the Flask UI
- session-backed workflow state
- Markdown-based supporting documentation
- AI-generated assessments where configured
- export/report style pages and generated narrative views

---

## Architecture Overview

DerivaPro uses a standard Flask application-factory structure.

### Application entry points

- `run.py` provides the local application entry point.
- `derivapro/__init__.py` initializes the Flask app and registers blueprints.
- `derivapro/config.py` contains application configuration.

### Main package layout

- `derivapro/routes/` contains Flask blueprints and page handlers.
- `derivapro/models/` contains pricing engines, data helpers, and analytics logic.
- `derivapro/templates/` contains Jinja templates for all user-facing pages.
- `derivapro/static/` contains CSS, generated artifacts, and saved plots.
- `derivapro/legacy/` contains retired or archived implementation files retained for reference.

### Content model

Many pages are backed by Markdown files stored alongside route modules. These are rendered into templates to provide:
- conceptual soundness notes
- workflow instructions
- model governance content
- end-user educational context

---

## Repository Structure

A simplified high-level view of the repository:
- `README.md` - project overview and setup notes
- `requirements.txt` - Python dependencies
- `setup.py` - package setup metadata
- `run.py` - local runtime entry point
- `derivapro/`
  - `__init__.py`
  - `config.py`
  - `routes/`
  - `models/`
  - `templates/`
  - `static/`
  - `legacy/`

Key route modules include:
- `routes/vanilla_options.py`
- `routes/exotic_options.py`
- `routes/futures_forwards.py`
- `routes/credit_derivatives.py`
- `routes/swaps.py`
- `routes/swaptions.py`
- `routes/term_structure.py`
- `routes/volatility_surface.py`
- `routes/variance_swaps.py`
- `routes/prepayment.py`
- `routes/prepayment_v2.py`
- `routes/bonds.py`

Key model modules include:
- `models/market_data.py`
- `models/mdls_vanilla_options.py`
- `models/mdls_monte_carlo_v2.py`
- `models/mdls_binomial_tree.py`
- `models/mdls_lattice_trees.py`
- `models/mdls_asian_options.py`
- `models/mdls_autocallables.py`
- `models/mdls_credit.py`
- `models/mdls_futures_forwards.py`
- `models/mdls_term_structure.py`
- `models/mdls_variance_volatility_swaps.py`
- `models/mdls_prepayment.py`
- `models/mdls_prepayment_v2.py`

Legacy / archived modules may also exist under:
- `derivapro/legacy/`

---

## Pricing Engine Notes

### Vanilla options

The vanilla options workflow is the most feature-rich area of the application.

It combines multiple pricing approaches, including:
- Black-Scholes style pricing and Greeks from `mdls_vanilla_options.py`
- lattice-based pricing through `mdls_lattice_trees.py`
- Monte Carlo support through the v2 engine in `mdls_monte_carlo_v2.py`
- American binomial pricing through `models/mdls_binomial_tree.py`

### Binomial tree implementation

The current active American binomial tree implementation is:
- `derivapro/models/mdls_binomial_tree.py`

This module provides `BinomialTreeEngineCRR`, which is used by the American vanilla options workflow. It supports:
- CRR-style binomial pricing
- variable internal step handling
- discrete dividend handling
- Greeks by finite differences
- exercise-boundary style output helpers

Older binomial development artifacts have been moved out of the active runtime path and retained under:
- `derivapro/legacy/`

### Monte Carlo implementation

The active Monte Carlo implementation is:
- `derivapro/models/mdls_monte_carlo_v2.py`

This module provides the current engine-based Monte Carlo support used by the application for active pricing workflows, including vanilla and exotic option use cases.

The older Monte Carlo implementation has been retired from the active runtime path and moved to:
- `derivapro/legacy/`

### Prepayment implementation

Two separate prepayment tracks exist today:
- `mdls_prepayment.py` supporting the simpler calculator-style workflow
- `mdls_prepayment_v2.py` supporting a more data-oriented v2 workflow

These are not currently interchangeable and should be treated as separate product paths unless explicitly consolidated.

---

## Web UI and User Workflow

The front end is rendered with Jinja templates under `derivapro/templates/`.

Notable UI characteristics:
- a shared navigation structure through `base.html`
- form-driven analytics pages per asset class or product type
- conditional UI logic for model-specific parameters
- embedded explanatory Markdown content
- session-driven display of analysis results and generated plots

The American options UI currently uses the user-facing label:
- `Binomial Tree`

Generated plots are written to the static area and rendered in templates. Recent updates introduced unique plot filenames for active plot-generation paths to reduce concurrent-user overwrite collisions for filesystem-backed images.

---

## Environment Variables

Several externalized settings are read from environment variables. The following names are currently used by parts of the application and should be preserved as-is unless the code is intentionally updated everywhere:
- `OpenAI_API_Key`
- `Base_URL`
- `Model`
- `API_Version`
- `Auth_headers`
- `FRED_API_Key`

These values are expected to be supplied through a local `.env` file or equivalent environment configuration.

### Example `.env`

```env .env
OpenAI_API_Key=your_openai_or_azure_key
Base_URL=your_azure_openai_endpoint
Model=your_model_name
API_Version=your_api_version
Auth_headers=api-key
FRED_API_Key=your_fred_api_key
```

> Note: `.env` should remain excluded from version control.

---

## Setup and Local Development

### 1. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a local `.env` file in the project root and populate the required variables.

### 4. Run the application

```powershell
python run.py
```

The Flask app should then be available locally based on the configured runtime settings.

---

## Dependencies and External Services

The project depends on a mix of web, quantitative, and data libraries. Based on the codebase, important dependencies include:
- Flask
- NumPy
- SciPy
- Matplotlib
- pandas / market-data related dependencies where applicable
- QuantLib
- Markdown rendering utilities
- dotenv support
- OpenAI / Azure OpenAI client libraries
- Yahoo Finance-backed market data access through the application’s market data helpers

Please refer to `requirements.txt` for the exact pinned dependency list in the repository.

---

## Reporting and Model Validation Style Features

The platform includes reporting and model-risk-style supporting pages in several areas, especially under the vanilla options workflow. These include pages and handlers for:
- model performance
- model governance
- ongoing monitoring
- conceptual soundness
- report rendering and download generation

These features are oriented toward explanatory analytics and internal validation-style workflows rather than being a standalone enterprise reporting framework.

---

## Operational and Security Notes

This repository has undergone partial hardening, but some operational/security cleanup items may still remain depending on the current branch state. Typical examples include:
- debug-mode cleanup
- stronger secret-key handling
- removal of non-essential sensitive logging
- further normalization of legacy import patterns
- remaining concurrent-user artifact safety review in selected flows

Accordingly, this repository should be treated as an actively evolving application rather than a fully hardened production deployment artifact without further review.

---
