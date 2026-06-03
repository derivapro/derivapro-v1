from pathlib import Path
import secrets


ENV_FILE = Path(".env")


def ensure_local_secret_key() -> str:
    """
    Ensure SECRET_KEY exists in local .env for development use.

    Returns the existing or newly generated SECRET_KEY value.
    Only appends SECRET_KEY if it is missing.
    Does not modify any other .env entries.
    """
    if ENV_FILE.exists():
        contents = ENV_FILE.read_text(encoding="utf-8")
        for line in contents.splitlines():
            stripped = line.strip()
            if stripped.startswith("SECRET_KEY="):
                return stripped.split("=", 1)[1]
    else:
        contents = ""

    secret_key = secrets.token_hex(32)

    with ENV_FILE.open("a", encoding="utf-8") as env_file:
        if contents and not contents.endswith("\n"):
            env_file.write("\n")
        env_file.write(f"SECRET_KEY={secret_key}\n")

    return secret_key