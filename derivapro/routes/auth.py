from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required, login_user, logout_user

from ..extensions import db
from ..models.db_models import User

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("index.index"))

    error = None

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not password or not confirm_password:
            error = "All fields are required."
        elif password != confirm_password:
            error = "Passwords do not match."
        elif User.query.filter_by(username=username).first():
            error = "Username already exists."
        else:
            user = User(username=username)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
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

        user = User.query.filter_by(username=username).first()

        if user is None or not user.check_password(password):
            error = "Invalid username or password."
        else:
            login_user(user)
            flash("Logged in successfully.", "success")
            next_page = request.args.get("next")
            return redirect(next_page or url_for("index.index"))

    return render_template("auth/login.html", error=error)


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    flash("Logged out successfully.", "success")
    return redirect(url_for("index.index"))
