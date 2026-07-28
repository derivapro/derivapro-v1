import logging
import re
import time
from collections import defaultdict, deque
from urllib.parse import urlsplit

from flask import Blueprint, current_app, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from ..extensions import db
from ..models.db_models import User

logger = logging.getLogger(__name__)

auth_bp = Blueprint("auth", __name__)
_RATE_LIMITS: dict[str, deque[float]] = defaultdict(deque)


def _is_safe_redirect_url(target: str | None) -> bool:
    if not target:
        return False

    parsed = urlsplit(target)
    return not parsed.netloc and not parsed.scheme and target.startswith("/")


def _client_key(action: str, username: str = "") -> str:
    remote_addr = request.headers.get("X-Forwarded-For", request.remote_addr or "")
    remote_addr = remote_addr.split(",", 1)[0].strip()
    return f"{action}:{remote_addr}:{username.strip().lower()}"


def _rate_limited(action: str, username: str = "") -> bool:
    attempts = current_app.config["AUTH_RATE_LIMIT_ATTEMPTS"]
    window = current_app.config["AUTH_RATE_LIMIT_WINDOW_SECONDS"]
    now = time.monotonic()
    bucket = _RATE_LIMITS[_client_key(action, username)]

    while bucket and now - bucket[0] > window:
        bucket.popleft()

    if len(bucket) >= attempts:
        return True

    bucket.append(now)
    return False


def _clear_rate_limit(action: str, username: str = "") -> None:
    _RATE_LIMITS.pop(_client_key(action, username), None)


def _password_policy_error(password: str) -> str | None:
    min_length = current_app.config["PASSWORD_MIN_LENGTH"]
    if len(password) < min_length:
        return f"Password must be at least {min_length} characters long."
    if not re.search(r"[A-Z]", password):
        return "Password must include at least one uppercase letter."
    if not re.search(r"[a-z]", password):
        return "Password must include at least one lowercase letter."
    if not re.search(r"\d", password):
        return "Password must include at least one number."
    if not re.search(r"[^A-Za-z0-9]", password):
        return "Password must include at least one symbol."
    return None


def _reset_serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(current_app.config["SECRET_KEY"], salt="password-reset")


def _make_reset_token(user: User) -> str:
    return _reset_serializer().dumps({"user_id": user.id, "password_hash": user.password_hash})


def _load_reset_token(token: str) -> User | None:
    max_age = current_app.config["PASSWORD_RESET_TOKEN_MAX_AGE_SECONDS"]
    try:
        payload = _reset_serializer().loads(token, max_age=max_age)
    except (BadSignature, SignatureExpired):
        return None

    user = db.session.get(User, payload.get("user_id"))
    if user is None or user.password_hash != payload.get("password_hash"):
        return None
    return user


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("index.index"))

    error = None

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        password_error = _password_policy_error(password) if password else None

        if (
            not username
            or not password
            or not confirm_password
        ):
            error = "All fields are required."
        elif password != confirm_password:
            error = "Passwords do not match."
        elif password_error:
            error = password_error
        elif User.query.filter_by(username=username).first():
            error = "Username already exists."
        else:
            user = User(username=username)
            user.set_password(password)

            db.session.add(user)
            db.session.commit()

            logger.info("User registered: user_id=%s", user.id)
            flash("Registration successful. Please log in.", "success")
            return redirect(url_for("auth.login"))

    return render_template("auth/register.html", error=error)


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index.index"))

    error = None

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")

        if _rate_limited("login", username):
            logger.warning("Login throttled for username=%s", username)
            error = "Too many login attempts. Please try again later."
        else:
            user = User.query.filter_by(username=username).first()

            if user is None or not user.check_password(password):
                logger.warning("Failed login attempt for username=%s", username)
                error = "Invalid username or password."
            else:
                _clear_rate_limit("login", username)
                login_user(user)
                logger.info("User logged in: user_id=%s", user.id)
                flash("Logged in successfully.", "success")
                next_page = request.args.get("next")
                if not _is_safe_redirect_url(next_page):
                    next_page = url_for("index.index")
                return redirect(next_page)

    return render_template("auth/login.html", error=error)


@auth_bp.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if current_user.is_authenticated:
        return redirect(url_for("index.index"))

    error = None
    reset_link = None

    if request.method == "POST":
        username = request.form.get("username", "").strip()

        if _rate_limited("password_reset", username):
            logger.warning("Password reset throttled for username=%s", username)
            error = "Too many password reset attempts. Please try again later."
        elif not username:
            error = "Username is required."
        else:
            user = User.query.filter_by(username=username).first()
            if user is not None:
                token = _make_reset_token(user)
                reset_link = url_for("auth.reset_password_with_token", token=token)
                logger.info("Password reset token issued: user_id=%s", user.id)
            flash(
                "If the account exists, a password reset link has been generated.",
                "success",
            )

    return render_template("auth/forgot_password.html", error=error, reset_link=reset_link)


@auth_bp.route("/reset-password/<username>", methods=["GET"])
def reset_password(username):
    flash("Password resets now require an expiring reset token.", "error")
    return redirect(url_for("auth.forgot_password"))


@auth_bp.route("/reset-password-token/<token>", methods=["GET", "POST"])
def reset_password_with_token(token):
    if current_user.is_authenticated:
        return redirect(url_for("index.index"))

    user = _load_reset_token(token)
    if user is None:
        flash("Invalid or expired password reset link.", "error")
        return redirect(url_for("auth.forgot_password"))

    error = None

    if request.method == "POST":
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        password_error = _password_policy_error(password) if password else None

        if _rate_limited("password_reset_submit", user.username):
            error = "Too many password reset attempts. Please try again later."
        elif not password or not confirm_password:
            error = "All fields are required."
        elif password != confirm_password:
            error = "Passwords do not match."
        elif password_error:
            error = password_error
        else:
            user.set_password(password)
            db.session.commit()
            _clear_rate_limit("password_reset", user.username)
            _clear_rate_limit("password_reset_submit", user.username)
            logger.info("Password reset completed: user_id=%s", user.id)
            flash("Password reset successful. Please log in.", "success")
            return redirect(url_for("auth.login"))

    return render_template(
        "auth/reset_password.html",
        error=error,
        username=user.username,
        token=token,
    )


@auth_bp.route("/change-password", methods=["GET", "POST"])
@login_required
def change_password():
    error = None

    if request.method == "POST":
        current_password = request.form.get("current_password", "")
        new_password = request.form.get("new_password", "")
        confirm_password = request.form.get("confirm_password", "")
        password_error = _password_policy_error(new_password) if new_password else None

        if not current_password or not new_password or not confirm_password:
            error = "All fields are required."
        elif not current_user.check_password(current_password):
            logger.warning("Failed password change verification: user_id=%s", current_user.id)
            error = "Current password is incorrect."
        elif new_password != confirm_password:
            error = "New passwords do not match."
        elif password_error:
            error = password_error
        else:
            current_user.set_password(new_password)
            db.session.commit()
            logger.info("Password changed: user_id=%s", current_user.id)
            flash("Password changed successfully.", "success")
            return redirect(url_for("index.index"))

    return render_template("auth/change_password.html", error=error)


@auth_bp.route("/logout", methods=["POST"])
@login_required
def logout():
    user_id = current_user.id
    logout_user()
    logger.info("User logged out: user_id=%s", user_id)
    flash("Logged out successfully.", "success")
    return redirect(url_for("index.index"))
