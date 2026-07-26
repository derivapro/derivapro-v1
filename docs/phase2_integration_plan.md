# Phase 2 Integration Review Plan

This plan covers the review and integration of the remote branch:

```text
origin/Phase-2---Data-Persistence-&-Auth
```

As of July 26, 2026, the branch is substantially ahead of `main` and includes authentication, database persistence, migrations, portfolio routes, saved-result workflows, report utilities, CSRF handling, UI updates, and additional runtime artifacts. It should be reviewed intentionally before merging.

## Goals

- Preserve the current `main` product workflows, including the structured autocallable/Phoenix workflow and DerivaPro Lite demo.
- Validate that authentication, persistence, portfolio, and reporting changes are production-directionally sound.
- Avoid merging generated artifacts, uploaded data, temporary models, debug scripts, or environment-specific files unless they are deliberately needed.
- Establish a repeatable verification path for future platform phases.

## Review Sequence

1. Create a local review branch from `main`.
2. Merge or rebase the Phase 2 branch into that review branch.
3. Resolve conflicts with priority on preserving current `main` behavior.
4. Review application startup, route registration, configuration, and extension initialization.
5. Review database models and migrations.
6. Review authentication, authorization, CSRF, and user-scoped data behavior.
7. Review portfolio, saved-results, analysis-history, and export/report workflows.
8. Review static/runtime artifacts and update `.gitignore` where needed.
9. Run local app smoke tests and targeted route checks.
10. Merge to `main` only after the branch can run locally and the high-risk workflows pass.

## High-Risk Areas To Inspect

| Area | Review Focus |
|---|---|
| App startup | Flask app factory, extensions, database initialization, config defaults. |
| Authentication | Registration, login, password reset, session behavior, user isolation. |
| CSRF | All browser forms and AJAX-style POST flows. |
| Database | Migrations, model relationships, nullable fields, default values, local vs production database config. |
| Persistence | Saved pricing results, analysis history, portfolio ownership, report metadata. |
| Products | Existing vanilla, exotic, structured autocallable, rates, credit, fixed-income, and prepayment workflows. |
| Static files | Generated plots, uploaded CSV files, model registry files, reports, temp models. |
| Dependencies | New packages in `requirements.txt`, install reliability, version compatibility. |
| Debug files | `_debug_*.py`, local scripts, and temporary diagnostics. |

## Minimum Acceptance Criteria

- `git status` is clean except intentional local-only files.
- The app imports and starts locally.
- Core public routes return successfully.
- Auth routes render and support a basic register/login/logout flow.
- Database migrations can initialize a clean local database.
- Existing pricing workflows still render.
- Structured autocallable pricing form renders with defaults and preserves submitted values.
- Portfolio pages render and enforce user-specific access.
- Saved results/history pages render and do not expose cross-user data.
- Runtime-generated plots, uploads, temp models, and reports are not committed unless intentionally curated.
- README and setup instructions describe any new environment variables or database setup.

## Suggested Local Commands

```bash
git checkout main
git pull --ff-only origin main
git checkout -b review/phase2-integration
git merge origin/Phase-2---Data-Persistence-\&-Auth
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m compileall -q derivapro
python scripts/print_routes.py
python run.py
```

If database migrations are active on the review branch, add the migration command after confirming the selected local database configuration.

## Recommended Outcome

Do not merge the Phase 2 branch as a blind fast-forward. Treat it as a review branch, split out or discard generated/runtime files where practical, fix startup or route regressions, and then merge a cleaned version into `main`.
