import os
import joblib


def save_model_artifact(model_obj, artifact_path):
    os.makedirs(os.path.dirname(artifact_path), exist_ok=True)
    joblib.dump(model_obj, artifact_path)
    return artifact_path


def load_model_artifact(artifact_path):
    return joblib.load(artifact_path)


def delete_model_artifact(artifact_path):
    if artifact_path and os.path.exists(artifact_path):
        os.remove(artifact_path)
        return True
    return False
