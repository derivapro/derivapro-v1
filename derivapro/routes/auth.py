from datetime import datetime
from urllib.parse import urlsplit

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user
from sqlalchemy import or_

from ..extensions import db
from ..models.db_models import User

auth_bp = Blueprint("auth", __name__)


def _is_safe_redirect_url(target: str | None) -> bool:
    if not target:
        return False

    parsed = urlsplit(target)
    return not parsed.netloc and not parsed.scheme and target.startswith("/")


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("index.index"))

    error = None

    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        organization = request.form.get("organization", "").strip()
        intended_use = request.form.get("intended_use", "").strip()
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        security_question = request.form.get("security_question", "").strip()
        security_answer = request.form.get("security_answer", "").strip()
        accepted_terms = request.form.get("accepted_terms") == "on"

        if (
            not full_name
            or not email
            or not organization
            or not intended_use
            or not username
            or not password
            or not confirm_password
            or not security_question
            or not security_answer
        ):
            error = "All fields are required."
        elif "@" not in email or "." not in email:
            error = "Please enter a valid email address."
        elif password != confirm_password:
            error = "Passwords do not match."
        elif not accepted_terms:
            error = "You must accept the terms and disclaimer to create an account."
        elif User.query.filter_by(username=username).first():
            error = "Username already exists."
        elif User.query.filter_by(email=email).first():
            error = "Email is already registered."
        else:
            user = User(
                username=username,
                full_name=full_name,
                email=email,
                organization=organization,
                intended_use=intended_use,
                security_question=security_question,
                accepted_terms=True,
                accepted_terms_at=datetime.utcnow(),
            )
            user.set_password(password)
            user.set_security_answer(security_answer)

            db.session.add(user)
            db.session.commit()

            flash("Registration successful. Please log in.", "success")
            return redirect(url_for("auth.login"))

    return render_template("auth/register.html", error=error, form_data=request.form)


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index.index"))

    error = None

    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        password = request.form.get("password", "")

        user = User.query.filter(
            or_(
                User.username == username,
                User.email == username,
            )
        ).first()

        if user is None or not user.check_password(password):
            error = "Invalid username or password."
        else:
            login_user(user)
            flash("Logged in successfully.", "success")
            next_page = request.args.get("next")
            if not _is_safe_redirect_url(next_page):
                next_page = url_for("index.index")
            return redirect(next_page)

    return render_template("auth/login.html", error=error)


@auth_bp.route("/terms", methods=["GET"])
def terms():
    return render_template("auth/terms.html")


@auth_bp.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if current_user.is_authenticated:
        return redirect(url_for("index.index"))

    error = None

    if request.method == "POST":
        username = request.form.get("username", "").strip()

        if not username:
            error = "Username is required."
        else:
            user = User.query.filter_by(username=username).first()
            if user is None or not user.security_question:
                error = "Unable to process password reset for this account."
            else:
                return redirect(url_for("auth.reset_password", username=user.username))

    return render_template("auth/forgot_password.html", error=error)


@auth_bp.route("/reset-password/<username>", methods=["GET", "POST"])
def reset_password(username):
    if current_user.is_authenticated:
        return redirect(url_for("index.index"))

    user = User.query.filter_by(username=username).first()

    if user is None or not user.security_question:
        flash("Invalid password reset request.", "error")
        return redirect(url_for("auth.forgot_password"))

    error = None

    if request.method == "POST":
        security_answer = request.form.get("security_answer", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not security_answer or not password or not confirm_password:
            error = "All fields are required."
        elif not user.check_security_answer(security_answer):
            error = "Incorrect security answer."
        elif password != confirm_password:
            error = "Passwords do not match."
        else:
            user.set_password(password)
            db.session.commit()
            flash("Password reset successful. Please log in.", "success")
            return redirect(url_for("auth.login"))

    return render_template(
        "auth/reset_password.html",
        error=error,
        username=user.username,
        security_question=user.security_question,
    )


@auth_bp.route("/change-password", methods=["GET", "POST"])
@login_required
def change_password():
    error = None

    if request.method == "POST":
        current_password = request.form.get("current_password", "")
        new_password = request.form.get("new_password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not current_password or not new_password or not confirm_password:
            error = "All fields are required."
        elif not current_user.check_password(current_password):
            error = "Current password is incorrect."
        elif new_password != confirm_password:
            error = "New passwords do not match."
        else:
            current_user.set_password(new_password)
            db.session.commit()
            flash("Password changed successfully.", "success")
            return redirect(url_for("index.index"))

    return render_template("auth/change_password.html", error=error)


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out successfully.", "success")
    return redirect(url_for("index.index"))
