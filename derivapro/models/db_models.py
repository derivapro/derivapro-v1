from datetime import datetime

from flask_login import UserMixin
from sqlalchemy import event
from sqlalchemy.orm import object_session

from ..extensions import bcrypt, db


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    security_question = db.Column(db.String(255), nullable=True)
    security_answer_hash = db.Column(db.String(255), nullable=True)
    role = db.Column(db.String(50), nullable=False, default="user")
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    instruments = db.relationship("Instrument", back_populates="user", lazy=True)
    pricing_results = db.relationship("PricingResult", back_populates="user", lazy=True)
    portfolios = db.relationship("Portfolio", back_populates="user", lazy=True)
    positions = db.relationship("Position", back_populates="user", lazy=True)
    plots = db.relationship("Plot", back_populates="user", lazy=True)
    reports = db.relationship("Report", back_populates="user", lazy=True)
    prepayment_models = db.relationship(
        "PrepaymentModelRegistry", back_populates="user", lazy=True
    )

    def set_password(self, password: str) -> None:
        self.password_hash = bcrypt.generate_password_hash(password).decode("utf-8")

    def check_password(self, password: str) -> bool:
        return bcrypt.check_password_hash(self.password_hash, password)

    def set_security_answer(self, answer: str) -> None:
        self.security_answer_hash = bcrypt.generate_password_hash(
            answer.strip().lower()
        ).decode("utf-8")

    def check_security_answer(self, answer: str) -> bool:
        if not self.security_answer_hash:
            return False
        return bcrypt.check_password_hash(
            self.security_answer_hash, answer.strip().lower()
        )

    def __repr__(self):
        return f"<User {self.username}>"


class Instrument(db.Model):
    __tablename__ = "instruments"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)

    product_type = db.Column(db.String(100), nullable=False)
    ticker = db.Column(db.String(50), nullable=True)
    model_name = db.Column(db.String(100), nullable=True)

    start_date = db.Column(db.String(20), nullable=True)
    end_date = db.Column(db.String(20), nullable=True)

    params_json = db.Column(db.JSON, nullable=True)

    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    user = db.relationship("User", back_populates="instruments")
    pricing_results = db.relationship(
        "PricingResult", back_populates="instrument", lazy=True
    )
    positions = db.relationship("Position", back_populates="instrument", lazy=True)
    reports = db.relationship("Report", back_populates="instrument", lazy=True)

    def __repr__(self):
        return f"<Instrument {self.product_type} {self.ticker}>"


class PricingResult(db.Model):
    __tablename__ = "pricing_results"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    instrument_id = db.Column(
        db.Integer, db.ForeignKey("instruments.id"), nullable=False
    )

    price = db.Column(db.Float, nullable=True)
    delta = db.Column(db.Float, nullable=True)
    gamma = db.Column(db.Float, nullable=True)
    vega = db.Column(db.Float, nullable=True)
    theta = db.Column(db.Float, nullable=True)
    rho = db.Column(db.Float, nullable=True)

    result_json = db.Column(db.JSON, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    user = db.relationship("User", back_populates="pricing_results")
    instrument = db.relationship("Instrument", back_populates="pricing_results")
    analysis_results = db.relationship(
        "AnalysisResult", back_populates="pricing_result", lazy=True
    )
    plots = db.relationship("Plot", back_populates="pricing_result", lazy=True)
    reports = db.relationship("Report", back_populates="pricing_result", lazy=True)
    positions = db.relationship("Position", back_populates="pricing_result", lazy=True)

    def __repr__(self):
        return f"<PricingResult instrument_id={self.instrument_id} price={self.price}>"


class AnalysisResult(db.Model):
    __tablename__ = "analysis_results"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    instrument_id = db.Column(
        db.Integer, db.ForeignKey("instruments.id"), nullable=False
    )
    pricing_result_id = db.Column(
        db.Integer, db.ForeignKey("pricing_results.id"), nullable=True
    )

    analysis_type = db.Column(db.String(100), nullable=False)
    result_json = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    user = db.relationship("User")
    instrument = db.relationship("Instrument")
    pricing_result = db.relationship("PricingResult", back_populates="analysis_results")
    plots = db.relationship("Plot", back_populates="analysis_result", lazy=True)
    reports = db.relationship("Report", back_populates="analysis_result", lazy=True)

    def __repr__(self):
        return f"<AnalysisResult type={self.analysis_type} instrument_id={self.instrument_id}>"


class Plot(db.Model):
    __tablename__ = "plots"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    analysis_result_id = db.Column(
        db.Integer, db.ForeignKey("analysis_results.id"), nullable=True
    )
    pricing_result_id = db.Column(
        db.Integer, db.ForeignKey("pricing_results.id"), nullable=True
    )

    plot_type = db.Column(db.String(100), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(500), nullable=False)

    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    user = db.relationship("User", back_populates="plots")
    analysis_result = db.relationship("AnalysisResult", back_populates="plots")
    pricing_result = db.relationship("PricingResult", back_populates="plots")

    def __repr__(self):
        return f"<Plot type={self.plot_type} filename={self.filename}>"


class Report(db.Model):
    __tablename__ = "reports"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    instrument_id = db.Column(
        db.Integer, db.ForeignKey("instruments.id"), nullable=True
    )
    pricing_result_id = db.Column(
        db.Integer, db.ForeignKey("pricing_results.id"), nullable=True
    )
    analysis_result_id = db.Column(
        db.Integer, db.ForeignKey("analysis_results.id"), nullable=True
    )

    report_type = db.Column(db.String(100), nullable=False)
    filename = db.Column(db.String(255), nullable=False)
    filepath = db.Column(db.String(500), nullable=False)
    pdf_data = db.Column(db.LargeBinary, nullable=True)

    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    user = db.relationship("User", back_populates="reports")
    instrument = db.relationship("Instrument", back_populates="reports")
    pricing_result = db.relationship("PricingResult", back_populates="reports")
    analysis_result = db.relationship("AnalysisResult", back_populates="reports")

    def __repr__(self):
        return f"<Report type={self.report_type} filename={self.filename}>"


class Portfolio(db.Model):
    __tablename__ = "portfolios"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)

    name = db.Column(db.String(150), nullable=False)
    description = db.Column(db.Text, nullable=True)

    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    user = db.relationship("User", back_populates="portfolios")
    positions = db.relationship("Position", back_populates="portfolio", lazy=True)

    def __repr__(self):
        return f"<Portfolio {self.name}>"


class Position(db.Model):
    __tablename__ = "positions"

    id = db.Column(db.Integer, primary_key=True)
    portfolio_id = db.Column(db.Integer, db.ForeignKey("portfolios.id"), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    instrument_id = db.Column(
        db.Integer, db.ForeignKey("instruments.id"), nullable=False
    )
    pricing_result_id = db.Column(
        db.Integer, db.ForeignKey("pricing_results.id"), nullable=True
    )

    quantity = db.Column(db.Float, nullable=False, default=1.0)
    notional = db.Column(db.Float, nullable=True)

    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    portfolio = db.relationship("Portfolio", back_populates="positions")
    user = db.relationship("User", back_populates="positions")
    instrument = db.relationship("Instrument", back_populates="positions")
    pricing_result = db.relationship("PricingResult", back_populates="positions")

    def __repr__(self):
        return (
            f"<Position portfolio_id={self.portfolio_id} "
            f"instrument_id={self.instrument_id} quantity={self.quantity}>"
        )


class PrepaymentModelRegistry(db.Model):
    __tablename__ = "prepayment_model_registry"

    __table_args__ = (
        db.Index("ix_prepayment_model_registry_user_active", "user_id", "is_active"),
        db.Index("ix_prepayment_model_registry_user_temp", "user_id", "is_temporary"),
    )

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)

    dataset_name = db.Column(db.String(255), nullable=True)
    model_type = db.Column(db.String(100), nullable=False)
    model_name = db.Column(db.String(100), nullable=False)
    task_type = db.Column(db.String(50), nullable=False)
    target_variable = db.Column(db.String(255), nullable=False)

    feature_columns_json = db.Column(db.JSON, nullable=True)
    hyperparameters_json = db.Column(db.JSON, nullable=True)
    metrics_json = db.Column(db.JSON, nullable=True)
    preprocessing_json = db.Column(db.JSON, nullable=True)

    artifact_path = db.Column(db.String(500), nullable=False)
    artifact_filename = db.Column(db.String(255), nullable=False)
    storage_backend = db.Column(db.String(50), nullable=False, default="local")

    is_active = db.Column(db.Boolean, nullable=False, default=True)
    is_temporary = db.Column(db.Boolean, nullable=False, default=False)
    registered_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    user = db.relationship("User", back_populates="prepayment_models")

    def __repr__(self):
        return (
            f"<PrepaymentModelRegistry id={self.id} "
            f"user_id={self.user_id} "
            f"model_name={self.model_name} "
            f"model_type={self.model_type} "
            f"is_active={self.is_active} "
            f"is_temporary={self.is_temporary}>"
        )


def _get_record_session(target):
    return object_session(target) or db.session


def _get_pricing_result_for_link(target):
    pricing_result_id = getattr(target, "pricing_result_id", None)
    if not pricing_result_id:
        return None

    pricing_result = _get_record_session(target).get(
        PricingResult,
        pricing_result_id,
    )
    if pricing_result is None:
        raise ValueError("Linked pricing result does not exist.")

    if target.user_id != pricing_result.user_id:
        raise ValueError(
            "Linked pricing result must belong to the same user as the persisted record."
        )

    return pricing_result


def _get_analysis_result_for_link(target):
    analysis_result_id = getattr(target, "analysis_result_id", None)
    if not analysis_result_id:
        return None

    analysis_result = _get_record_session(target).get(
        AnalysisResult,
        analysis_result_id,
    )
    if analysis_result is None:
        raise ValueError("Linked analysis result does not exist.")

    if target.user_id != analysis_result.user_id:
        raise ValueError(
            "Linked analysis result must belong to the same user as the persisted record."
        )

    return analysis_result


def _validate_instrument_owner(target):
    instrument_id = getattr(target, "instrument_id", None)
    if not instrument_id:
        return

    instrument = _get_record_session(target).get(Instrument, instrument_id)
    if instrument is None:
        raise ValueError("Linked instrument does not exist.")

    if target.user_id != instrument.user_id:
        raise ValueError(
            "Linked instrument must belong to the same user as the persisted record."
        )


@event.listens_for(AnalysisResult, "before_insert")
@event.listens_for(AnalysisResult, "before_update")
def align_analysis_with_pricing_result(mapper, connection, target):
    pricing_result = _get_pricing_result_for_link(target)
    if pricing_result is not None:
        target.instrument_id = pricing_result.instrument_id

    _validate_instrument_owner(target)


@event.listens_for(Plot, "before_insert")
@event.listens_for(Plot, "before_update")
def align_plot_links(mapper, connection, target):
    analysis_result = _get_analysis_result_for_link(target)

    if analysis_result is not None:
        if (
            target.pricing_result_id is not None
            and target.pricing_result_id != analysis_result.pricing_result_id
        ):
            raise ValueError(
                "Plot pricing result must match the linked analysis result."
            )
        target.pricing_result_id = analysis_result.pricing_result_id

    _get_pricing_result_for_link(target)


@event.listens_for(Report, "before_insert")
@event.listens_for(Report, "before_update")
def align_report_links(mapper, connection, target):
    analysis_result = _get_analysis_result_for_link(target)

    if analysis_result is not None:
        if (
            target.pricing_result_id is not None
            and target.pricing_result_id != analysis_result.pricing_result_id
        ):
            raise ValueError(
                "Report pricing result must match the linked analysis result."
            )

        target.pricing_result_id = analysis_result.pricing_result_id
        target.instrument_id = analysis_result.instrument_id

    pricing_result = _get_pricing_result_for_link(target)
    if pricing_result is not None:
        target.instrument_id = pricing_result.instrument_id

    _validate_instrument_owner(target)