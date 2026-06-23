from datetime import datetime

from flask_login import UserMixin

from ..extensions import bcrypt, db


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    role = db.Column(db.String(50), nullable=False, default="user")
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    instruments = db.relationship("Instrument", back_populates="user", lazy=True)
    pricing_results = db.relationship("PricingResult", back_populates="user", lazy=True)

    def set_password(self, password: str) -> None:
        self.password_hash = bcrypt.generate_password_hash(password).decode("utf-8")

    def check_password(self, password: str) -> bool:
        return bcrypt.check_password_hash(self.password_hash, password)

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

    def __repr__(self):
        return f"<PricingResult instrument_id={self.instrument_id} price={self.price}>"


class AnalysisResult(db.Model):
    __tablename__ = "analysis_results"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id"), nullable=False)
    instrument_id = db.Column(db.Integer, db.ForeignKey("instruments.id"), nullable=False)
    pricing_result_id = db.Column(
        db.Integer, db.ForeignKey("pricing_results.id"), nullable=True
    )

    analysis_type = db.Column(db.String(100), nullable=False)
    result_json = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    user = db.relationship("User")
    instrument = db.relationship("Instrument")
    pricing_result = db.relationship("PricingResult", back_populates="analysis_results")

    def __repr__(self):
        return f"<AnalysisResult type={self.analysis_type} instrument_id={self.instrument_id}>"