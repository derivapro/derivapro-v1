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

**Remaining validation:** Include CSRF-protected POSTs in route regression tests.

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

- [ ] Add `curl_cffi` as a direct dependency in `requirements.txt`.
- [ ] Enable SQLite foreign-key enforcement for every SQLAlchemy connection.
- [ ] Add explicit production cookie settings.
- [ ] Require authentication for the complete prepayment-v2 workflow.
- [ ] Reject unsupported prepayment storage backends instead of recording false S3 state.

**A1 validation gate**

- [ ] `python -m compileall -q derivapro migrations run.py scripts` passes using `.venv`.
- [ ] `create_app()` succeeds.
- [ ] 83 expected routes still register, unless a deliberate route change is documented.
- [ ] `flask db current` remains at migration head.
- [ ] `PRAGMA foreign_keys` returns `1` inside an application DB connection.
- [ ] Anonymous requests to all prepayment-v2 workflow routes redirect to login.
- [ ] Authenticated prepayment-v2 GET renders successfully.
- [ ] No CSRF regressions occur.

#### Batch A2 — User-isolated prepayment files

- [ ] Introduce a user-specific upload root, such as `uploads/user_<id>/`.
- [ ] Generate a collision-safe stored filename using a UUID.
- [ ] Retain the sanitized original filename only as display/metadata.
- [ ] Validate that session paths resolve inside the current user's allowed directory.
- [ ] Use user-safe UUID-based filenames for temporary and registered model artifacts.
- [ ] Prevent deletion endpoints from removing files outside the current user's directory.
- [ ] Add a maximum upload size and explicit CSV validation.

**A2 validation gate**

- [ ] Two users can upload files with the same original name without collision.
- [ ] One user cannot read, preprocess, register, or delete another user's data/artifacts.
- [ ] Path traversal attempts are rejected.
- [ ] Existing single-user upload and model-training behavior remains functional.

#### Batch A3 — Correct future analysis linkage

- [ ] Stop creating an unrelated instrument row when an analysis belongs to an existing pricing result.
- [ ] Resolve the current user-owned pricing result first.
- [ ] Set `AnalysisResult.instrument_id` from `pricing_result.instrument_id` when linked.
- [ ] If no pricing result exists, create a standalone instrument and leave `pricing_result_id` null.
- [ ] Apply the same rule to associated `Plot` and `Report` rows.
- [ ] Centralize this linkage behavior to avoid route-by-route drift.

**Required invariant** — for every analysis with a linked pricing result:

```text
analysis.user_id == pricing_result.user_id
analysis.instrument_id == pricing_result.instrument_id
```

**A3 validation gate**

- [ ] New European option analyses satisfy the invariant.
- [ ] New American option analyses satisfy the invariant.
- [ ] New barrier, Asian, and autocallable analyses satisfy the invariant.
- [ ] Cross-user result IDs cannot be linked.
- [ ] Analysis history and reports still render.

#### Batch A4 — Repair existing data

- [ ] Back up `instance/derivapro.db`.
- [ ] Produce a dry-run report listing every proposed row update.
- [ ] Decide whether each existing analysis should adopt the linked pricing result's instrument or have `pricing_result_id` cleared.
- [ ] Apply the repair in a versioned migration or auditable maintenance script.
- [ ] Re-run all consistency queries.

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

- [ ] Make logout POST-only and CSRF-protected.
- [ ] Configure production cookies with `SESSION_COOKIE_SECURE=True`.
- [ ] Set `SESSION_COOKIE_HTTPONLY=True`.
- [ ] Set an appropriate `SESSION_COOKIE_SAMESITE` policy.
- [ ] Add minimum password requirements.
- [ ] Add login and password-reset throttling.
- [ ] Add security headers, preferably through Flask-Talisman.
- [ ] Replace security-question reset with expiring, single-use reset tokens when email infrastructure is available.
- [ ] Add audit logging for authentication-sensitive actions without logging secrets.

**Stage B validation gate**

- [ ] Valid registration/login/change-password flows work.
- [ ] Invalid credentials return safe, non-enumerating messages.
- [ ] Unsafe `next` redirects are rejected.
- [ ] Logout cannot be triggered through GET.
- [ ] Rate limits work as configured.
- [ ] Production cookies carry the expected flags.

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
- 83 routes register unless intentionally changed;
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
| Prepayment user isolation       | 🔴 Fail                                          |
| Analysis relational consistency | 🔴 Fail                                          |
| Portfolio manager               | ✅ Implemented                                   |
| Aggregated Greeks               | ⚠️ Implemented; convention validation required |
| PDF reporting                   | ⚠️ Partial                                     |
| CSV/XLSX export                 | ⚠️ Partial                                     |
| CSRF                            | ✅ Pass                                          |
| Type hints                      | ⚠️ Partial                                     |
| Dead-code removal               | ⚠️ Partial                                     |
| Full product correctness        | ⬜ Not yet verified                              |
| Automated tests                 | 🔴 Missing                                       |
| CI                              | 🔴 Missing                                       |

---

## Progress Log

*Update this section after each independently validated batch.*

| Date    | Batch                      | Result         | Commit/Reference | Notes                                                       |
| ------- | -------------------------- | -------------- | ---------------- | ----------------------------------------------------------- |
| 2026-07 | Baseline and Phase 2 audit | ✅ Completed   | —               | Findings documented; no backend pricing-model changes made. |
|         | A1                         | ⬜ Not started |                  |                                                             |
|         | A2                         | ⬜ Not started |                  |                                                             |
|         | A3                         | ⬜ Not started |                  |                                                             |
|         | A4                         | ⬜ Not started |                  |                                                             |
|         | B                          | ⬜ Not started |                  |                                                             |
|         | C                          | ⬜ Not started |                  |                                                             |
|         | D                          | ⬜ Not started |                  |                                                             |

---

## Recommended next action

**Start with Stage A, Batch A1 only:**

1. declare `curl_cffi` as a direct dependency;
2. enable SQLite foreign-key enforcement;
3. configure production session-cookie security;
4. authenticate the full prepayment-v2 workflow; and
5. fail safely for unsupported model-storage backends.

Validate A1 completely before proceeding to user-specific upload paths or analysis-link repairs.
