---
name: Project Context
alwaysApply: true
description: Background, architecture, and current progress — read at the start of every session
---

# Project: DerivaPro-v1

## Overview
DerivaPro is a Flask-based, browser-facing quantitative finance platform for multi-asset pricing, model evaluation, reporting, and portfolio risk workflows. It uses SQLAlchemy/Alembic, Flask-Login, Flask-WTF, QuantLib, NumPy/SciPy, scikit-learn, Jinja templates, and configurable LLM providers.

## Architecture / Key Decisions
- 2026-07-24 — Protect same-origin AJAX centrally through a CSRF meta token and Fetch/XMLHttpRequest wrappers in `base.html`, while retaining explicit hidden tokens in every POST/AJAX form for defense in depth.
- 2026-07-24 — Validate development-plan work in small, independently verifiable stages and avoid major pricing-model changes during platform hardening.

## Current Status
The app compiles and starts successfully in `.venv`; 83 routes register, public product GET pages render, authenticated saved-results/history/portfolio/report pages render, and the SQLite database is at migration head `c1a2f3b4d5e6`. Phase 2 persistence/authentication is substantially implemented, including user-scoped results, analysis history, portfolios, report downloads, and a DB-backed prepayment registry. CSRF initialization and template coverage now pass the form audit, including explicit tokens on all detected POST/AJAX forms and rates API requests. Portfolio management, Greek aggregation, ReportLab PDF generation, and selected CSV/XLSX exports are implemented but still need completeness and financial-convention validation.

## Next Steps
- Complete Stage A Phase 2 hardening: enable SQLite foreign-key enforcement, fix future `AnalysisResult` linkage, repair existing mismatched rows, and isolate/authenticate the full prepayment-v2 workflow.
- Declare direct dependencies such as `curl_cffi`, add production cookie settings, and prevent unsupported S3 configuration from being recorded as active storage.
- Validate Greek aggregation units, complete report listing and analysis/risk exports, then add automated route/model/cross-user regression tests and CI.
- Run staged valid/invalid POST tests for every pricing and analysis workflow, including plot, persistence, PDF, and export assertions.

## Known Issues / Gotchas
- SQLite reports `PRAGMA foreign_keys = 0`; declared foreign keys are not currently enforced.
- All four existing `analysis_results` rows link to pricing results associated with different instrument rows, despite matching user/product types.
- Prepayment uploads use shared original filenames and many preprocessing endpoints are anonymous, allowing cross-user filename collisions; configured S3 storage is not implemented.
- Type hints, dead-code cleanup, PDF reporting coverage, and CSV/XLSX coverage are partial rather than complete; no real pytest suite or GitHub Actions workflow exists.
- `curl_cffi` is imported directly but absent from `requirements.txt`; dependencies remain broadly unpinned.
- Use `.venv/Scripts/python.exe` for audits. The global Python environment lacks required packages, and `scripts/print_routes.py` should be run as `python -m scripts.print_routes`.
- Runtime plots and historical `.pkl` artifacts remain under `derivapro/static/` without a retention/cleanup policy.