import pytest

from derivapro import create_app
from derivapro.config import DevelopmentConfig
from derivapro.extensions import db
from derivapro.models.db_models import User


@pytest.fixture
def app(tmp_path, monkeypatch):
    database_path = tmp_path / "test.db"
    upload_root = tmp_path / "uploads"
    temp_model_dir = tmp_path / "temp_models"
    model_registry_dir = tmp_path / "model_registry"

    monkeypatch.setattr(DevelopmentConfig, "TESTING", True)
    monkeypatch.setattr(DevelopmentConfig, "DEBUG", False)
    monkeypatch.setattr(DevelopmentConfig, "SECRET_KEY", "test-secret-key")
    monkeypatch.setattr(
        DevelopmentConfig,
        "SQLALCHEMY_DATABASE_URI",
        f"sqlite:///{database_path.as_posix()}",
    )
    monkeypatch.setattr(DevelopmentConfig, "CACHE_TYPE", "SimpleCache")
    monkeypatch.setattr(DevelopmentConfig, "WTF_CSRF_ENABLED", True, raising=False)
    monkeypatch.setattr(
        DevelopmentConfig,
        "PREPAYMENT_UPLOAD_ROOT",
        str(upload_root),
    )
    monkeypatch.setattr(
        DevelopmentConfig,
        "PREPAYMENT_TEMP_MODEL_DIR",
        str(temp_model_dir),
    )
    monkeypatch.setattr(
        DevelopmentConfig,
        "PREPAYMENT_MODEL_REGISTRY_DIR",
        str(model_registry_dir),
    )
    monkeypatch.setattr(
        DevelopmentConfig,
        "PREPAYMENT_MODEL_STORAGE_BACKEND",
        "local",
    )

    monkeypatch.setenv("FLASK_ENV", "development")

    application = create_app()

    with application.app_context():
        db.create_all()

    yield application

    with application.app_context():
        db.session.remove()
        db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def user(app):
    with app.app_context():
        test_user = User(username="test-user")
        test_user.set_password("ValidTestPassword1!")
        db.session.add(test_user)
        db.session.commit()
        user_id = test_user.id

    return user_id


@pytest.fixture
def authenticated_client(app, client, user):
    with client.session_transaction() as session:
        session["_user_id"] = str(user)
        session["_fresh"] = True

    return client