from __future__ import annotations

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user

from ..services.simulation_config import (
    get_effective_simulation_settings,
    reset_simulation_settings,
    save_simulation_settings,
    simulation_settings_choices,
)

configuration_bp = Blueprint("configuration", __name__)


@configuration_bp.route("/simulation", methods=["GET", "POST"])
def simulation_configuration():
    if request.method == "POST":
        action = request.form.get("action", "save")
        if action == "reset":
            settings = reset_simulation_settings(current_user)
            flash("Simulation configuration reset to DerivaPro defaults.", "success")
        else:
            settings = save_simulation_settings(current_user, request.form)
            flash("Simulation configuration saved.", "success")
        return redirect(url_for("configuration.simulation_configuration"))

    settings = get_effective_simulation_settings(current_user)
    return render_template(
        "configuration/simulation.html",
        settings=settings,
        choices=simulation_settings_choices(),
    )
