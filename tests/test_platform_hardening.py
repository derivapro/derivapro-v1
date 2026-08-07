import re
from io import BytesIO
from pathlib import Path

import joblib
import pandas as pd
import pytest
from werkzeug.datastructures import FileStorage

from derivapro import create_app
from derivapro.config import DevelopmentConfig, ProductionConfig
from derivapro.extensions import db
from derivapro.models.db_models import PrepaymentModelRegistry, User
from derivapro.models.mdls_prepayment_v2 import PrepaymentDataUploader
import derivapro.routes.prepayment_v2 as prepayment_routes


class FakeValidation:
    def __init__(self, filepath):
        self.filepath = filepath
        self.target = "target"
        self.task = "classification"
        self.X_train = pd.DataFrame({"feature": [1.0]})
        self.trained_model = {"model": "test-model"}
        self.last_training_method = "random_forest"
        self.last_training_params = {"n_estimators": 1}
        self.last_training_results = {
            "model": "Random Forest",
            "metrics": {"Accuracy": 1.0},
        }

    def model_training(self, method, hyperparameters):
        return self.last_training_results


def _csrf_token(client):
    response = client.get("/")
    match = re.search(
        r'<meta name="csrf-token" content="([^"]+)">',
        response.get_data(as_text=True),
    )
    assert match is not None
    return match.group(1)


def _prepare_training_file(app, client, user_id):
    upload_dir = Path(app.config["PREPAYMENT_UPLOAD_ROOT"]) / f"user_{user_id}"
    upload_dir.mkdir(parents=True)
    dataset_path = upload_dir / "training.csv"
    dataset_path.write_text(
        "feature,target\n1.0,0\n",
        encoding="utf-8",
    )

    with client.session_transaction() as session:
        session["uploaded_data_file_path"] = str(dataset_path)
        session["uploaded_filename"] = "training.csv"

    return dataset_path



def _create_second_user(app):
    with app.app_context():
        second_user = User(username="second-user")
        second_user.set_password("SecondValidPassword1!")
        db.session.add(second_user)
        db.session.commit()
        return second_user.id
    
def test_anonymous_prepayment_post_with_valid_csrf_redirects(client):
    token = _csrf_token(client)

    response = client.post(
        "/prepayment-v2/delete_upload",
        data={"csrf_token": token},
    )

    assert response.status_code == 302
    assert "/auth/login" in response.headers["Location"]


def test_production_cookie_configuration(monkeypatch):
    monkeypatch.setenv("FLASK_ENV", "production")
    monkeypatch.setattr(ProductionConfig, "CACHE_TYPE", "SimpleCache")
    application = create_app()

    assert application.config["SESSION_COOKIE_SECURE"] is True
    assert application.config["SESSION_COOKIE_HTTPONLY"] is True
    assert application.config["SESSION_COOKIE_SAMESITE"] == "Lax"
    assert application.config["REMEMBER_COOKIE_SECURE"] is True
    assert application.config["REMEMBER_COOKIE_HTTPONLY"] is True
    assert application.config["REMEMBER_COOKIE_SAMESITE"] == "Lax"


def test_unsupported_storage_backend_rejects_startup(monkeypatch):
    monkeypatch.setenv("FLASK_ENV", "development")
    monkeypatch.setattr(
        DevelopmentConfig,
        "PREPAYMENT_MODEL_STORAGE_BACKEND",
        "s3",
    )

    with pytest.raises(RuntimeError, match="Only 'local' is implemented"):
        create_app()


def test_model_training_persists_temporary_registry_entry(
    app,
    authenticated_client,
    user,
    monkeypatch,
):
    monkeypatch.setattr(
        prepayment_routes,
        "Validation",
        FakeValidation,
    )
    _prepare_training_file(app, authenticated_client, user)
    token = _csrf_token(authenticated_client)

    response = authenticated_client.post(
        "/prepayment-v2/prepayment-model-validator/model_training",
        json={"method": "random_forest", "hyperparameters": {}},
        headers={"X-CSRFToken": token},
    )

    assert response.status_code == 200
    assert response.get_json()["success"] is True

    with app.app_context():
        entry = PrepaymentModelRegistry.query.one()
        artifact_path = Path(entry.artifact_path)
        assert entry.user_id == user
        assert entry.is_temporary is True
        assert entry.storage_backend == "local"

    assert artifact_path.is_file()
    assert artifact_path.parent.name == f"user_{user}"




def test_two_users_upload_same_original_filename_without_collision(app, user):
    second_user_id = _create_second_user(app)
    upload_root = Path(app.config["PREPAYMENT_UPLOAD_ROOT"])
    first_uploader = PrepaymentDataUploader(upload_root / f"user_{user}")
    second_uploader = PrepaymentDataUploader(
        upload_root / f"user_{second_user_id}"
    )

    first_result = first_uploader.save_uploaded_file(
        FileStorage(
            stream=BytesIO(b"feature,target\n1,0\n"),
            filename="loan_data.csv",
            content_type="text/csv",
        )
    )
    second_result = second_uploader.save_uploaded_file(
        FileStorage(
            stream=BytesIO(b"feature,target\n2,1\n"),
            filename="loan_data.csv",
            content_type="text/csv",
        )
    )

    first_path = Path(first_result["filepath"]).resolve()
    second_path = Path(second_result["filepath"]).resolve()

    assert first_result["success"] is True
    assert second_result["success"] is True
    assert first_result["filename"] == "loan_data.csv"
    assert second_result["filename"] == "loan_data.csv"
    assert first_path != second_path
    assert first_path.parent.name == f"user_{user}"
    assert second_path.parent.name == f"user_{second_user_id}"
    assert first_path.read_text(encoding="utf-8") != second_path.read_text(
        encoding="utf-8"
    )



def test_cross_user_upload_session_path_is_rejected(
    app,
    authenticated_client,
    user,
):
    second_user_id = _create_second_user(app)
    second_user_directory = (
        Path(app.config["PREPAYMENT_UPLOAD_ROOT"])
        / f"user_{second_user_id}"
    )
    second_user_directory.mkdir(parents=True)
    second_user_file = second_user_directory / "private.csv"
    second_user_file.write_text(
        "feature,target\n2,1\n",
        encoding="utf-8",
    )

    with authenticated_client.session_transaction() as session:
        session["uploaded_data_file_path"] = str(second_user_file)
        session["uploaded_filename"] = "private.csv"

    response = authenticated_client.get(
        "/prepayment-v2/prepayment-model-validator/get_columns"
    )

    assert response.status_code == 200
    assert response.get_json() == {
        "success": False,
        "error": "Uploaded file not found",
    }
    assert second_user_file.is_file()

    with authenticated_client.session_transaction() as session:
        assert "uploaded_data_file_path" not in session
        assert "uploaded_filename" not in session



def test_register_model_rejects_tampered_temp_artifact_path(
    app,
    authenticated_client,
    user,
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        prepayment_routes,
        "Validation",
        FakeValidation,
    )
    _prepare_training_file(app, authenticated_client, user)

    outside_artifact = tmp_path / "outside-temp-model.joblib"
    joblib.dump(
        {
            "model": {"model": "outside"},
            "results": {
                "model": "Random Forest",
                "metrics": {"Accuracy": 1.0},
            },
            "method": "random_forest",
            "params": {"n_estimators": 1},
        },
        outside_artifact,
    )

    with app.app_context():
        entry = PrepaymentModelRegistry(
            user_id=user,
            dataset_name="training.csv",
            model_type="random_forest",
            model_name="Random Forest",
            task_type="classification",
            target_variable="target",
            feature_columns_json=["feature"],
            hyperparameters_json={"n_estimators": 1},
            metrics_json={"Accuracy": 1.0},
            preprocessing_json={},
            artifact_path=str(outside_artifact),
            artifact_filename=outside_artifact.name,
            storage_backend="local",
            is_active=False,
            is_temporary=True,
        )
        db.session.add(entry)
        db.session.commit()

    token = _csrf_token(authenticated_client)
    response = authenticated_client.post(
        "/prepayment-v2/prepayment-model-validator/register_model",
        headers={"X-CSRFToken": token},
    )

    assert response.status_code == 200
    assert response.get_json() == {
        "success": False,
        "error": "Failed to load trained model. Please train again.",
    }
    assert outside_artifact.is_file()



def test_deregister_model_rejects_external_artifact_path(
    app,
    authenticated_client,
    user,
    tmp_path,
):
    _prepare_training_file(app, authenticated_client, user)
    outside_artifact = tmp_path / "outside-registered-model.joblib"
    joblib.dump({"model": "must-not-delete"}, outside_artifact)

    with app.app_context():
        entry = PrepaymentModelRegistry(
            user_id=user,
            dataset_name="training.csv",
            model_type="random_forest",
            model_name="Random Forest",
            task_type="classification",
            target_variable="target",
            feature_columns_json=["feature"],
            hyperparameters_json={"n_estimators": 1},
            metrics_json={"Accuracy": 1.0},
            preprocessing_json={},
            artifact_path=str(outside_artifact),
            artifact_filename=outside_artifact.name,
            storage_backend="local",
            is_active=True,
            is_temporary=False,
        )
        db.session.add(entry)
        db.session.commit()
        entry_id = entry.id

    token = _csrf_token(authenticated_client)
    response = authenticated_client.post(
        "/prepayment-v2/prepayment-model-validator/deregister_model",
        headers={"X-CSRFToken": token},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["success"] is False
    assert "outside the allowed storage directory" in payload["error"]
    assert outside_artifact.is_file()

    with app.app_context():
        entry = db.session.get(PrepaymentModelRegistry, entry_id)
        assert entry.is_active is True