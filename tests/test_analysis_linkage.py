import pytest

from derivapro.extensions import db
from derivapro.models.db_models import (
    AnalysisResult,
    Instrument,
    Plot,
    PricingResult,
    Report,
    User,
)


def _create_user(username):
    user = User(username=username)
    user.set_password("ValidTestPassword1!")
    db.session.add(user)
    db.session.flush()
    return user


def _create_instrument(user_id, product_type="european_option"):
    instrument = Instrument(
        user_id=user_id,
        product_type=product_type,
        ticker="TEST",
        model_name="Test Model",
    )
    db.session.add(instrument)
    db.session.flush()
    return instrument


def _create_pricing_result(user_id, instrument_id):
    pricing_result = PricingResult(
        user_id=user_id,
        instrument_id=instrument_id,
        price=10.0,
        result_json={"price": 10.0},
    )
    db.session.add(pricing_result)
    db.session.flush()
    return pricing_result


def test_linked_analysis_adopts_pricing_instrument(app, user):
    with app.app_context():
        pricing_instrument = _create_instrument(user)
        unrelated_instrument = _create_instrument(user)
        pricing_result = _create_pricing_result(
            user,
            pricing_instrument.id,
        )

        analysis = AnalysisResult(
            user_id=user,
            instrument_id=unrelated_instrument.id,
            pricing_result_id=pricing_result.id,
            analysis_type="sensitivity",
            result_json={"status": "ok"},
        )
        db.session.add(analysis)
        db.session.flush()

        assert analysis.instrument_id == pricing_instrument.id
        assert analysis.user_id == pricing_result.user_id
        assert analysis.instrument_id == pricing_result.instrument_id


def test_standalone_analysis_remains_unlinked(app, user):
    with app.app_context():
        instrument = _create_instrument(user)

        analysis = AnalysisResult(
            user_id=user,
            instrument_id=instrument.id,
            pricing_result_id=None,
            analysis_type="standalone_scenario",
            result_json={"status": "ok"},
        )
        db.session.add(analysis)
        db.session.flush()

        assert analysis.instrument_id == instrument.id
        assert analysis.pricing_result_id is None


def test_cross_user_pricing_link_is_rejected(app, user):
    with app.app_context():
        second_user = _create_user("analysis-second-user")
        first_user_instrument = _create_instrument(user)
        second_user_instrument = _create_instrument(second_user.id)
        second_user_pricing = _create_pricing_result(
            second_user.id,
            second_user_instrument.id,
        )

        analysis = AnalysisResult(
            user_id=user,
            instrument_id=first_user_instrument.id,
            pricing_result_id=second_user_pricing.id,
            analysis_type="sensitivity",
            result_json={"status": "invalid"},
        )
        db.session.add(analysis)

        with pytest.raises(
            ValueError,
            match="Linked pricing result must belong to the same user",
        ):
            db.session.flush()

        db.session.rollback()


def test_plot_inherits_analysis_pricing_result(app, user):
    with app.app_context():
        instrument = _create_instrument(user)
        pricing_result = _create_pricing_result(user, instrument.id)
        analysis = AnalysisResult(
            user_id=user,
            instrument_id=instrument.id,
            pricing_result_id=pricing_result.id,
            analysis_type="sensitivity",
            result_json={"status": "ok"},
        )
        db.session.add(analysis)
        db.session.flush()

        plot = Plot(
            user_id=user,
            analysis_result_id=analysis.id,
            pricing_result_id=None,
            plot_type="sensitivity",
            filename="sensitivity.png",
            filepath="plots/sensitivity.png",
        )
        db.session.add(plot)
        db.session.flush()

        assert plot.pricing_result_id == pricing_result.id


def test_plot_conflicting_pricing_result_is_rejected(app, user):
    with app.app_context():
        first_instrument = _create_instrument(user)
        second_instrument = _create_instrument(user)
        first_pricing = _create_pricing_result(user, first_instrument.id)
        second_pricing = _create_pricing_result(user, second_instrument.id)
        analysis = AnalysisResult(
            user_id=user,
            instrument_id=first_instrument.id,
            pricing_result_id=first_pricing.id,
            analysis_type="sensitivity",
            result_json={"status": "ok"},
        )
        db.session.add(analysis)
        db.session.flush()

        plot = Plot(
            user_id=user,
            analysis_result_id=analysis.id,
            pricing_result_id=second_pricing.id,
            plot_type="sensitivity",
            filename="conflict.png",
            filepath="plots/conflict.png",
        )
        db.session.add(plot)

        with pytest.raises(
            ValueError,
            match="Plot pricing result must match the linked analysis result",
        ):
            db.session.flush()

        db.session.rollback()


def test_report_inherits_analysis_links(app, user):
    with app.app_context():
        pricing_instrument = _create_instrument(user)
        unrelated_instrument = _create_instrument(user)
        pricing_result = _create_pricing_result(
            user,
            pricing_instrument.id,
        )
        analysis = AnalysisResult(
            user_id=user,
            instrument_id=pricing_instrument.id,
            pricing_result_id=pricing_result.id,
            analysis_type="validation",
            result_json={"status": "ok"},
        )
        db.session.add(analysis)
        db.session.flush()

        report = Report(
            user_id=user,
            instrument_id=unrelated_instrument.id,
            pricing_result_id=None,
            analysis_result_id=analysis.id,
            report_type="model_validation",
            filename="report.pdf",
            filepath="reports/report.pdf",
            pdf_data=b"%PDF-test",
        )
        db.session.add(report)
        db.session.flush()

        assert report.instrument_id == pricing_instrument.id
        assert report.pricing_result_id == pricing_result.id
        assert report.analysis_result_id == analysis.id


def test_report_conflicting_pricing_result_is_rejected(app, user):
    with app.app_context():
        first_instrument = _create_instrument(user)
        second_instrument = _create_instrument(user)
        first_pricing = _create_pricing_result(user, first_instrument.id)
        second_pricing = _create_pricing_result(user, second_instrument.id)
        analysis = AnalysisResult(
            user_id=user,
            instrument_id=first_instrument.id,
            pricing_result_id=first_pricing.id,
            analysis_type="validation",
            result_json={"status": "ok"},
        )
        db.session.add(analysis)
        db.session.flush()

        report = Report(
            user_id=user,
            instrument_id=first_instrument.id,
            pricing_result_id=second_pricing.id,
            analysis_result_id=analysis.id,
            report_type="model_validation",
            filename="conflict.pdf",
            filepath="reports/conflict.pdf",
            pdf_data=b"%PDF-test",
        )
        db.session.add(report)

        with pytest.raises(
            ValueError,
            match="Report pricing result must match the linked analysis result",
        ):
            db.session.flush()

        db.session.rollback()