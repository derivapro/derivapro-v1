<p align="center">
  <img src="https://github.com/user-attachments/assets/54dd4b1b-a332-4109-a768-ca1c4e695fe6" alt="DerivaPro logo_final" width="300">
</p>

---

## Overview

- **DerivaPro** is a Flask-based analytical dashboard that exposes quantitative pricing and risk tooling through a collection of blueprints registered in a standard application factory (`create_app`).

- The production entry point simply instantiates the factory and calls `app.run`, so a newcomer can launch the site with `python run.py` after dependencies are installed.

- Configuration is minimal—currently just a placeholder `SECRET_KEY`—so environment-specific secrets will need to be supplied separately.

---

## Project Layout and Navigation

- All routes live under `derivapro/routes`, and the factory wires them into the app: vanilla and exotic options, futures/forwards, swaps, credit products, volatility tools, prepayment, and term structure analytics are exposed as individual blueprints.

- The navigation bar in `base.html` mirrors that organization and is reused by every page, providing quick links to each analytics area alongside a MathJax-enabled content pane.

- The default landing page renders Markdown from `routes/home.md`, which doubles as end-user documentation and highlights coverage across asset classes.

---

## Backend Models and Data Helpers

- `derivapro/models` houses the pricing engines.  
  For example, `StockData` wraps Yahoo Finance for quotes, calendar logic, and implied-volatility extraction—making it the go-to helper for any market input.

- Core option analytics live in `mdls_vanilla_options.py`, which implements Black–Scholes greeks and a `SmoothnessTest` utility for plotting sensitivities over parameter ranges.

- Monte Carlo simulations (including barrier support and finite-difference greeks) sit in `mdls_monte_carlo.py`, pulling term and spot data through `StockData` and offering reusable path-generation logic.

- Credit products leverage QuantLib; the `CreditDefaultSwap` class in `mdls_credit.py` constructs schedules, hazard curves, and produces NPV, fair spread, expected loss, and sensitivity plots.

---

## Blueprints, Views, and Analytics Flows

- The **vanilla options** blueprint is the busiest: it dynamically loads Monte Carlo and binomial engines, exposes multiple model choices in the European option workflow, calculates greeks, persists results in the session, and even pipes outputs into Azure OpenAI for narrative assessments.

- The accompanying `european_options.html` template demonstrates how Markdown explanations, form validation, and conditional UI (e.g., Monte Carlo parameter toggles and client-side checks) are integrated.

- Many pages embed pre-authored guidance, such as the vanilla dashboard’s PDF primer delivered through `vanilla_options.html`.

- Other blueprints follow a consistent pattern:  
  load a local Markdown tutorial, handle form submissions, compute pricing/greeks/sensitivities, and save plots into `static`.  
  Examples include:
  - **Futures/Forwards** routes: sensitivity, scenario, and risk-based P&L analyses.  
  - **Credit Derivatives** routes: QuantLib-driven CDS and CDO workflows.  
  - **Volatility Surface** blueprint: fetches option chains, builds a 3D implied-vol grid, and allows CSV exports.
