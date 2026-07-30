# 🔍 DerivaPro Phase 2 — Audit & Remediation Plan

> **Audit scope:** Baseline application integrity, Phase 2 persistence/authentication, and selected completed roadmap items
> **Audit date:** July 2026
> **Approach:** Apply and validate changes in small, independently reversible stages. Avoid major pricing-model changes during platform hardening.

---

## 📑 Contents

1. [Purpose](#purpose)
2. [Stage 1–2 Audit Result](#stage-12-audit-result)
3. [Confirmed Phase 2 Issues](#confirmed-phase-2-issues) — 🔴 Critical · 🟠 High priority
4. [Other Roadmap Findings](#other-roadmap-findings)
5. [Staged Remediation Plan](#staged-remediation-plan) — Stages A–D
6. [Baseline Validation Commands](#baseline-validation-commands)
7. [Current Status Matrix](#current-status-matrix)
8. [Progress Log](#progress-log)
9. [Recommended Next Action](#recommended-next-action)

---

## Purpose

This document records the current audit findings and provides a staged implementation checklist. Complete one batch at a time, run its validation steps, and record the outcome before proceeding.

---

## Stage 1–2 Audit Result

The baseline and Phase 2 review was completed **without changing backend application logic**.

### ✅ Confirmed working

| ✔ | Item                                                             |
| -- | ---------------------------------------------------------------- |
| ✅ | All Python files compile                                         |
| ✅ | `create_app()` starts successfully using `.venv`             |
| ✅ | 83 routes register                                               |
| ✅ | Database is at Alembic migration head`c1a2f3b4d5e6`            |
| ✅ | Public product pages load successfully on GET                    |
| ✅ | Protected pages load successfully for an authenticated test user |
| ✅ | Saved-results and analysis-history queries are user-scoped       |
| ✅ | Portfolio routes enforce user ownership                          |
| ✅ | Report downloads enforce user ownership                          |
| ✅ | New model artifacts use`joblib`                                |
| ✅ | Current SQLite database passes`PRAGMA integrity_check`         |
| ✅ | CSRF initialization and detected form coverage pass the audit    |

> **Conclusion:** Phase 2 is substantially implemented, but it is not yet complete or enterprise-grade. The integrity and isolation issues below should be resolved before comprehensive product POST testing.

---

## Confirmed Phase 2 Issues

### 🔴 Critical

#### 1. SQLite foreign keys are not enforced

The database returned:

```text
PRAGMA foreign_keys = 0
```

The schema declares foreign keys, but SQLite is not enforcing them at runtime. This allows orphaned or inconsistent relationships to be written.

> **Required outcome:** Every application-managed SQLite connection enables `PRAGMA foreign_keys=ON`.

#### 2. Existing analysis-result links are inconsistent

The audit found:

```text
analysis_pricing_mismatch = 4
```

All four existing `AnalysisResult` rows reference:

- one instrument through `analysis_results.instrument_id`; and
- a different instrument through the linked `pricing_results.instrument_id`.

The users and product types match, but the relational linkage does not.

**Root cause:** Analysis workflows create a new instrument row, then link the analysis to the latest pricing result belonging to another instrument row.

> **Required outcome:** If an analysis has a `pricing_result_id`, its `instrument_id` and `user_id` must match the linked pricing result.

#### 3. Prepayment uploads are not isolated by user

Uploads currently use the sanitized original filename in a shared directory:

```python
filename = secure_filename(file.filename)
filepath = os.path.join(self.upload_folder, filename)
```

Two users uploading `loan_data.csv` can target the same path and overwrite one another.

> **Required outcome:** Store uploads under a user-specific location and use a collision-safe server filename while retaining the original filename as metadata.

#### 4. Much of prepayment-v2 is accessible anonymously

Only training, registration, deregistration, and selected registry operations currently require authentication. Upload, preprocessing, feature selection, deletion, and performance operations rely on session-held file paths but are accessible anonymously.

> **Required outcome:** Protect the complete prepayment-v2 workflow with `@login_required` and verify that every referenced upload/artifact belongs to the current user.

### 🟠 High priority

#### 5. S3 model storage is configured but not implemented

Configuration includes:

- `PREPAYMENT_MODEL_STORAGE_BACKEND`
- `PREPAYMENT_S3_BUCKET`
- `PREPAYMENT_S3_PREFIX`
- `PREPAYMENT_S3_ENDPOINT_URL`
- `PREPAYMENT_S3_REGION`

However, `derivapro/utils/model_storage.py` only reads and writes local files. Selecting `s3` can therefore result in a local artifact being recorded as S3-backed.

> **Required outcome:** Until S3 operations are implemented and tested, fail application startup or model registration when the configured backend is not `local`.

#### 6. Phase 2.4 is only partially complete

Database-backed pricing and analysis records exist, but sessions still hold state including:

- European option form data;
- prepayment upload paths;
- feature-selection results; and
- other workflow state.

Lightweight session references are acceptable, but persistent result payloads and ownership-sensitive paths should move to database-backed IDs or user-scoped records.

#### 7. Authentication needs production hardening

Current gaps include:

- no password-strength policy;
- no login or password-reset rate limiting;
- security-question password reset;
- logout through GET;
- no explicit production cookie security settings;
- no account lockout; and
- no expiring password-reset tokens.

---

## Other Roadmap Findings

| #   | Item                           | Status                                |
| --- | ------------------------------ | ------------------------------------- |
| 3.5 | Type hints                     | ⚠️ Partial                          |
| 3.6 | Dead-code removal              | ⚠️ Partial                          |
| 3.7 | CSRF protection                | ✅ Functionally complete              |
| 5.1 | Portfolio and Position Manager | ✅ Implemented                        |
| 5.2 | Aggregated Greeks Dashboard    | ✅ Implemented — validation required |
| 6.1 | PDF reporting                  | ⚠️ Partially complete               |
| 7.4 | CSV/XLSX export                | ⚠️ Partially complete               |

### 3.5 Type hints — ⚠️ Partial

Core models have meaningful type annotations, but the roadmap item is not complete:

- several model methods remain untyped;
- most route functions remain untyped;
- `mypy` is not installed or configured;
- no strict type-check command exists in CI; and
- duplicate signatures suggest mixed typed/untyped or stale implementations.

**Status:** In progress.

### 3.6 Dead-code removal — ⚠️ Partial

Legacy, debug, commented, and historical artifacts remain, including:

- `derivapro/legacy/mdls_monte_carlo.py`
- `derivapro/legacy/mdls_prepayment.py`
- `derivapro/legacy/mdls_binomial_tree_model_NEW.py`
- debug helper scripts
- historical `.pkl` artifacts under `derivapro/static/`

**Status:** In progress.

### 3.7 CSRF protection — ✅ Functionally complete

- `CSRFProtect` is initialized in the application factory.
- All detected POST/AJAX forms contain explicit CSRF fields.
- Rates API requests explicitly send CSRF tokens.
- `base.html` centrally protects same-origin Fetch and XMLHttpRequest calls.

**Validation update:** A targeted regression test now confirms that an anonymous prepayment POST with a valid CSRF token reaches authentication enforcement and redirects to login. Broader product POST coverage remains in Stage D.

### 5.1 Portfolio and Position Manager — ✅ Implemented

The application supports:

- portfolio creation;
- adding a saved result as a position;
- updating positions;
- deleting positions; and
- user ownership checks.

### 5.2 Aggregated Greeks Dashboard — ✅ Implemented, validation required

Portfolio Greeks are aggregated by portfolio and product type. The current multiplier is selected as:

```python
position.notional if position.notional is not None else position.quantity
```

Quantity is ignored whenever notional is provided. This may or may not be correct depending on whether each stored Greek represents exposure per unit, per contract, per currency unit of notional, or total instrument exposure.

> **Required outcome:** Define and document Greek units by product type, then centralize the exposure multiplier convention before certifying portfolio totals.

### 6.1 PDF reporting — ⚠️ Partially complete

**Implemented:**

- structured `ReportTemplate`;
- ReportLab rendering;
- generated PDF bytes;
- database persistence; and
- user-scoped downloads.

**Incomplete:**

- the central Reports page remains a placeholder;
- reporting is concentrated in vanilla options;
- report linkage may inherit analysis/pricing instrument mismatches;
- PDFs are stored both on disk and in the database;
- no retention or cleanup policy exists; and
- user-scoped report listing is not implemented.

### 7.4 CSV/XLSX export — ⚠️ Partially complete

**Implemented for:**

- saved pricing results;
- portfolio list;
- portfolio positions; and
- portfolio Greek summaries.

**Still required for full roadmap completion:**

- analysis history;
- sensitivity data;
- scenario tables;
- convergence results;
- individual pricing-page result tables;
- risk reports; and
- generated report metadata.

---

## Staged Remediation Plan

### Operating rules

1. Complete one batch before beginning the next.
2. Back up the SQLite database before any data repair or migration.
3. Do not alter pricing-model behavior during Phase 2 hardening.
4. Run compile, startup, migration, GET smoke, and targeted regression checks after every batch.
5. Do not proceed when a batch leaves the application in a failing state.
6. Commit each validated batch separately so it can be reverted safely.

---

### 🅰️ Stage A — Phase 2 Integrity and Isolation

#### Batch A1 — Low-risk platform hardening

- [X] Add `curl_cffi` as a direct dependency in `requirements.txt`.
- [X] Enable SQLite foreign-key enforcement for every SQLAlchemy connection.
- [X] Add explicit production cookie settings.
- [X] Require authentication for the complete prepayment-v2 workflow.
- [X] Reject unsupported prepayment storage backends instead of recording false S3 state.

**A1 validation gate**

- [X] `python -m compileall -q derivapro migrations run.py scripts` passes using `.venv`.
- [X] `create_app()` succeeds in development.
- [X] 84 routes register; the deliberate additional route is the token-based password reset route.
- [X] `flask db current` remains at migration head.
- [X] `PRAGMA foreign_keys` returns `1` inside an application DB connection.
- [X] Anonymous prepayment GET routes redirect to login, and a targeted POST with valid CSRF redirects to login rather than failing CSRF validation.
- [X] Authenticated prepayment-v2 GET renders successfully.
- [X] The CSRF regression script passes when run as a module.

> Validation update (2026-07-29): `redis>=5.0.0` is declared, production-mode `create_app()` succeeds, the configured session cookie emits `Secure`, `HttpOnly`, and `SameSite=Lax`, and a Redis Cloud cache write/read/delete round trip succeeds.

#### Batch A2 — User-isolated prepayment files

- [X] Introduce a user-specific upload root, such as `uploads/user_<id>/`.
- [X] Generate a collision-safe stored filename using a UUID.
- [X] Retain the sanitized original filename only as display/metadata.
- [X] Validate that session upload paths resolve inside the current user's allowed directory.
- [X] Use user-safe UUID-based filenames for temporary and registered model artifacts.
- [X] Constrain model artifact save, load, and deletion operations to the current user's resolved temporary or registry directory.
- [X] Add a maximum upload size and explicit CSV validation.

**A2 validation gate**

- [X] Two users can upload files with the same original name without collision.
- [X] Cross-user upload paths and tampered temporary/registered artifact paths are rejected without reading or deleting another path.
- [X] Path traversal and resolved-path escapes are rejected by centralized allowed-root validation.
- [X] Existing single-user model training and temporary registry persistence remain functional.

#### Batch A3 — Correct future analysis linkage

- [ ] Stop creating an unrelated instrument row when an analysis belongs to an existing pricing result; all 11 vanilla analysis paths are migrated, while exotic analysis paths remain.
- [ ] Resolve the current user-owned pricing result first; vanilla routes and exotic product lookup are complete, but exotic persistence migration remains.
- [X] Set `AnalysisResult.instrument_id` from `pricing_result.instrument_id` when linked, with same-user enforcement.
- [X] If no pricing result exists, preserve a standalone analysis with an owned instrument and null `pricing_result_id`.
- [X] Centrally align and validate linked `Plot` and `Report` ownership, pricing-result, analysis-result, and instrument relationships.
- [X] Reject missing, cross-user, and conflicting instrument/pricing/analysis links through model event listeners.

**Required invariant** — for every analysis with a linked pricing result:

```text
analysis.user_id == pricing_result.user_id
analysis.instrument_id == pricing_result.instrument_id
```

**A3 validation gate**

- [ ] New European option analyses satisfy the invariant in product POST tests; all four persistence paths are statically migrated.
- [ ] New American option analyses satisfy the invariant in product POST tests; all seven persistence paths are statically migrated.
- [ ] New barrier, Asian, and autocallable analyses satisfy the invariant; product lookup is scoped, but persistence migration and POST tests remain.
- [X] Centralized tests reject cross-user and conflicting pricing/analysis links.
- [X] Centralized tests validate linked and standalone analysis, plot, and report invariants.
- [ ] Analysis history and reports still render after route-level A3 changes.

#### Batch A4 — Repair existing data

- [ ] Back up `instance/derivapro.db`; no backup file is currently present.
- [X] Produce a dry-run report listing every proposed row update.
- [X] Encode the policy: adopt the pricing instrument for same-user rows; clear cross-user pricing links.
- [ ] Apply the repair using the auditable maintenance script.
- [X] Re-run consistency queries; four analysis/pricing instrument mismatches remain, while the other five counters are zero.

**A4 validation gate** — expected result:

```text
analysis_pricing_mismatch = 0
pricing_user_mismatch = 0
analysis_user_mismatch = 0
position_portfolio_mismatch = 0
position_instrument_mismatch = 0
position_pricing_mismatch = 0
```

---

### 🅱️ Stage B — Authentication Hardening

- [X] Make logout POST-only and CSRF-protected.
- [X] Configure production cookies with `SESSION_COOKIE_SECURE=True`.
- [X] Set `SESSION_COOKIE_HTTPONLY=True`.
- [X] Set an appropriate `SESSION_COOKIE_SAMESITE` policy.
- [X] Add minimum password requirements.
- [X] Add login and password-reset throttling (in-memory only; not suitable for multi-worker production deployment).
- [X] Add baseline security headers through an application response hook; CSP/Talisman is not configured.
- [X] Replace the active security-question reset flow with expiring, password-change-invalidated tokens; delivery remains an on-page development link until email exists.
- [X] Add audit logging for authentication-sensitive actions without logging secrets.

**Stage B validation gate**

- [ ] Valid registration/login/change-password flows work.
- [ ] Invalid credentials return safe, non-enumerating messages.
- [ ] Unsafe `next` redirects are rejected.
- [ ] Logout cannot be triggered through GET.
- [ ] Rate limits work as configured.
- [X] Production session cookies carry `Secure`, `HttpOnly`, and `SameSite=Lax`; configuration tests also cover the equivalent remember-cookie settings.

---

### 🅲 Stage C — Completed-Feature Validation

#### C1 — Greek aggregation conventions

- [ ] Define Greek units for every persisted product type.
- [ ] Define whether quantity and notional multiply or replace one another.
- [ ] Add a centralized exposure calculation helper.
- [ ] Validate signs and scaling using hand-calculated portfolio examples.
- [ ] Add unit labels and assumptions to the UI/export/report.

#### C2 — Reporting completion

- [ ] Replace the Reports placeholder with a user-scoped report list.
- [ ] Add report metadata and download links.
- [ ] Decide on DB-only or object-storage-backed PDF persistence.
- [ ] Remove unnecessary duplicate disk storage.
- [ ] Define retention and cleanup behavior.
- [ ] Extend reports beyond vanilla options where required.

#### C3 — Export completion

- [ ] Export analysis history.
- [ ] Export sensitivity data.
- [ ] Export scenario tables.
- [ ] Export convergence results.
- [ ] Export individual pricing results.
- [ ] Export portfolio/risk reports.
- [ ] Export generated report metadata.
- [ ] Reject unsupported export formats instead of silently treating them as CSV.

#### C4 — Type hints and dead code

- [ ] Add and configure `mypy`.
- [ ] Define a realistic strictness boundary and ratchet it upward.
- [ ] Remove duplicate/stale implementations after comparison.
- [ ] Remove or archive debug helper scripts.
- [ ] Remove obsolete legacy modules after verifying no active imports.
- [ ] Remove historical `.pkl` files from runtime static storage.

---

### 🅳 Stage D — Product Workflow Regression Tests

*Begin this stage after the Phase 2 integrity batches are complete.*

#### D1 — Route smoke tests

- [ ] Anonymous public GET routes return expected 200 responses.
- [ ] Protected routes redirect anonymous users to login.
- [ ] Authenticated protected routes render successfully.
- [ ] Missing record IDs return 404 rather than exposing another user's record.

#### D2 — Pricing POST tests

For each product workflow:

- [ ] Submit valid inputs.
- [ ] Assert a non-error response.
- [ ] Assert expected result fields are present.
- [ ] Assert authenticated results persist with the correct user/instrument.
- [ ] Submit missing and malformed inputs.
- [ ] Assert structured 4xx handling instead of 500 errors.

#### D3 — Analysis and plot tests

For every sensitivity, scenario, convergence, and RBPL workflow:

- [ ] Submit valid inputs.
- [ ] Assert analysis output is non-empty.
- [ ] Assert generated plot files exist where applicable.
- [ ] Assert filenames are unique.
- [ ] Assert `AnalysisResult`, `Plot`, and linked pricing records satisfy ownership/instrument invariants.
- [ ] Assert invalid inputs do not leave partial DB rows or orphan files.

**Targeted hardening tests added (2026-07-29):**

- anonymous prepayment POST with a valid CSRF token redirects to login;
- production session and remember-cookie configuration is secure;
- unsupported prepayment storage backends fail application startup;
- authenticated prepayment model training reaches temporary artifact and registry persistence;
- two users can upload the same original filename without collision;
- cross-user upload session paths are rejected and cleared;
- tampered temporary artifact paths cannot be loaded; and
- external registered artifact paths cannot be deleted or falsely deactivated.

Current targeted result: `15 passed` across `tests/test_platform_hardening.py` and `tests/test_analysis_linkage.py`. The suite uses isolated temporary SQLite databases and artifact directories; seven A3 tests cover linked/standalone analyses, cross-user rejection, and plot/report inheritance and conflict handling.

#### D4 — Cross-user tests

- [ ] Users cannot view each other's saved results.
- [ ] Users cannot view each other's analysis history.
- [ ] Users cannot add another user's result to a portfolio.
- [ ] Users cannot update/delete another user's positions.
- [ ] Users cannot download another user's reports.
- [ ] Users cannot access another user's prepayment upload or model registry.
- [ ] Users cannot export another user's records.

#### D5 — PDF and export tests

- [ ] Generated PDF begins with a valid PDF signature and has meaningful content.
- [ ] Report rows link to the correct user, instrument, pricing result, and analysis.
- [ ] CSV output has expected headers and rows.
- [ ] XLSX output opens successfully and has expected sheet content.
- [ ] Empty result sets produce a valid, intentional export response.

---

## Baseline Validation Commands

Use the project virtual environment on Windows:

```powershell
.venv\Scripts\python.exe -m compileall -q derivapro migrations run.py scripts
.venv\Scripts\python.exe -c "from derivapro import create_app; app=create_app(); print(len(list(app.url_map.iter_rules())))"
$env:FLASK_APP = "run.py"
.venv\Scripts\python.exe -m flask db current
.venv\Scripts\python.exe -m flask db heads
.venv\Scripts\python.exe -m flask db check
.venv\Scripts\python.exe -m pip check
.venv\Scripts\python.exe -m scripts.print_routes
```

**Expected baseline:**

- compilation succeeds;
- app creation succeeds;
- 84 routes register, including the deliberate token-based password-reset route;
- current migration equals head `c1a2f3b4d5e6` until a new migration is deliberately added;
- no dependency conflicts are reported.

---

## Current Status Matrix

| Area                            | Status                                           |
| ------------------------------- | ------------------------------------------------ |
| Baseline startup                | ✅ Pass                                          |
| Database migrations             | ✅ Pass                                          |
| Phase 2 DB schema               | ⚠️ Mostly complete                             |
| Authentication                  | ⚠️ Implemented; hardening required             |
| User-scoped saved results       | ✅ Pass                                          |
| User-scoped analysis history    | ✅ Pass                                          |
| Prepayment user isolation       | ✅ A2 targeted gate passed                       |
| Analysis relational consistency | ⚠️ Future-write guards pass; A3 routes/A4 pending |
| Portfolio manager               | ✅ Implemented                                   |
| Aggregated Greeks               | ⚠️ Implemented; convention validation required |
| PDF reporting                   | ⚠️ Partial                                     |
| CSV/XLSX export                 | ⚠️ Partial                                     |
| CSRF                            | ✅ Pass                                          |
| Type hints                      | ⚠️ Partial                                     |
| Dead-code removal               | ⚠️ Partial                                     |
| Full product correctness        | ⬜ Not yet verified                              |
| Automated tests                 | ⚠️ Initial targeted suite: 15 passing tests      |
| CI                              | 🔴 Missing                                       |

---

## Progress Log

*Update this section after each independently validated batch.*

| Date    | Batch                      | Result         | Commit/Reference | Notes                                                                                                                                                                                                                     |
| ------- | -------------------------- | -------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-07 | Baseline and Phase 2 audit | ✅ Completed   | —               | Findings documented; no backend pricing-model changes made.                                                                                                                                                               |
| 2026-07 | A1                         | ✅ Completed   | pending commit   | Development and production startup pass; FK=1, 84 routes, CSRF/auth enforcement, secure cookies, Redis Cloud caching, and unsupported-backend rejection are validated.                                                    |
| 2026-07 | A2                         | ✅ Completed   | pending commit   | User-scoped uploads and model directories, UUID filenames, resolved allowed-root enforcement, cross-user/tampered-path rejection, and normal persistence are validated by the 8-test suite.                               |
| 2026-07 | A3                         | ⚠️ Partial     | pending commit   | Seven centralized invariant tests pass; all 11 vanilla paths use pricing-first resolution and exotic lookup is product-scoped. Exotic persistence migration and product POST/render tests remain. |
| 2026-07 | A4                         | ❌ Not applied | `3ce1504e`     | Dry-run script exists, but no backup is present and the live database still reports`analysis_pricing_mismatch = 4`.                                                                                                     |
| 2026-07 | B                          | ⚠️ Partial   | `3ce1504e`     | Production startup and cookie configuration now pass. Remaining validation covers auth flows, safe redirects, logout, and throttling; throttling is process-local, CSP is absent, and reset delivery is development-only. |
|         | C                          | ⬜ Not started |                  |                                                                                                                                                                                                                           |
|         | D                          | ⬜ Not started |                  |                                                                                                                                                                                                                           |

---

## Recommended next action

**Finish exotic A3 persistence before A4 or Stage C:**

1. apply pricing-first instrument resolution to four autocallable analysis branches and make their plots inherit analysis pricing links;
2. repeat the migration for four Asian and four barrier analysis branches;
3. add focused exotic persistence tests and run valid European/American/exotic POST and history/report render checks;
4. confirm new linked analyses, plots, and reports satisfy ownership/instrument invariants without creating unused instruments; and
5. only after A3 passes, back up the live database and perform the reviewed A4 repair.
