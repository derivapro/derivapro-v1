"""Repair AnalysisResult rows whose linked PricingResult points to another instrument.

By default this script performs a dry run. Use --apply to update rows so each linked
analysis adopts the linked pricing result's instrument_id. A timestamped SQLite
backup is created before applying changes.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from derivapro import create_app
from sqlalchemy import inspect

from derivapro.extensions import db
from derivapro.models.db_models import AnalysisResult, PricingResult


def _sqlite_db_path() -> Path | None:
    uri = db.engine.url
    if uri.drivername != "sqlite" or not uri.database or uri.database == ":memory:":
        return None
    return Path(uri.database)


def _mismatched_rows() -> list[tuple[AnalysisResult, PricingResult]]:
    table_names = set(inspect(db.engine).get_table_names())
    if not {"analysis_results", "pricing_results"}.issubset(table_names):
        return []

    return (
        db.session.query(AnalysisResult, PricingResult)
        .join(PricingResult, AnalysisResult.pricing_result_id == PricingResult.id)
        .filter(
            AnalysisResult.pricing_result_id.isnot(None),
            db.or_(
                AnalysisResult.user_id != PricingResult.user_id,
                AnalysisResult.instrument_id != PricingResult.instrument_id,
            ),
        )
        .order_by(AnalysisResult.id)
        .all()
    )


def _print_report(rows: list[tuple[AnalysisResult, PricingResult]]) -> None:
    print(f"analysis_pricing_mismatch={len(rows)}")
    for analysis, pricing in rows:
        action = "CLEAR_PRICING_LINK"
        new_instrument_id = analysis.instrument_id
        if analysis.user_id == pricing.user_id:
            action = "ADOPT_PRICING_INSTRUMENT"
            new_instrument_id = pricing.instrument_id
        print(
            "analysis_id={analysis_id} pricing_result_id={pricing_id} "
            "analysis_user_id={analysis_user_id} pricing_user_id={pricing_user_id} "
            "old_instrument_id={old_instrument_id} new_instrument_id={new_instrument_id} "
            "pricing_instrument_id={pricing_instrument_id} action={action}".format(
                analysis_id=analysis.id,
                pricing_id=pricing.id,
                analysis_user_id=analysis.user_id,
                pricing_user_id=pricing.user_id,
                old_instrument_id=analysis.instrument_id,
                new_instrument_id=new_instrument_id,
                pricing_instrument_id=pricing.instrument_id,
                action=action,
            )
        )


def _backup_database() -> Path | None:
    db_path = _sqlite_db_path()
    if db_path is None or not db_path.exists():
        return None
    backup_path = db_path.with_suffix(
        f".backup_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}{db_path.suffix}"
    )
    shutil.copy2(db_path, backup_path)
    return backup_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Apply the repair")
    args = parser.parse_args()

    app = create_app()
    with app.app_context():
        rows = _mismatched_rows()
        _print_report(rows)

        if not args.apply:
            print("dry_run=True")
            return

        backup_path = _backup_database()
        if backup_path:
            print(f"backup={backup_path}")

        for analysis, pricing in rows:
            if analysis.user_id == pricing.user_id:
                analysis.instrument_id = pricing.instrument_id
            else:
                analysis.pricing_result_id = None

        db.session.commit()
        print("applied=True")
        _print_report(_mismatched_rows())


if __name__ == "__main__":
    main()
