import os
from pathlib import Path

import joblib


def _validated_artifact_path(artifact_path, allowed_root):
    if not artifact_path or not allowed_root:
        raise ValueError("Artifact path and allowed root are required.")

    resolved_root = Path(allowed_root).resolve()
    resolved_path = Path(artifact_path).resolve()

    try:
        is_allowed = os.path.commonpath([resolved_path, resolved_root]) == str(
            resolved_root
        )
    except ValueError:
        is_allowed = False

    if not is_allowed or resolved_path == resolved_root:
        raise ValueError("Artifact path is outside the allowed storage directory.")

    return resolved_path


def save_model_artifact(model_obj, artifact_path, allowed_root):
    resolved_path = _validated_artifact_path(artifact_path, allowed_root)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_obj, resolved_path)
    return str(resolved_path)


def load_model_artifact(artifact_path, allowed_root):
    resolved_path = _validated_artifact_path(artifact_path, allowed_root)
    return joblib.load(resolved_path)


def delete_model_artifact(artifact_path, allowed_root):
    resolved_path = _validated_artifact_path(artifact_path, allowed_root)
    if resolved_path.exists():
        resolved_path.unlink()
        return True
    return False
