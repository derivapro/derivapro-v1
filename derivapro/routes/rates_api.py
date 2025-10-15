# routes/rates_api.py
from flask import Blueprint, request, jsonify
import datetime as dt
import io

from derivapro.models.curve import Curve
from derivapro.models.swaps import price_plain_swap, _read_curve_csv, _read_curve_file
from derivapro.models.vol_surface import VolSurface
from derivapro.models.swaptions import (
    price_european as price_swaption_eu,
    price_hw1f_mc
)

rates_api_bp = Blueprint("rates_api", __name__, url_prefix="/api/rates")

# -------- helpers ----------
def _jget(p, key, default=None):
    v = p.get(key) if isinstance(p, dict) else None
    return default if (v is None or v == "") else v

def _ffloat(p, key, default=None):
    v = _jget(p, key, None)
    return (float(v) if v is not None else default)

def _fint(p, key, default=None):
    v = _jget(p, key, None)
    return (int(v) if v is not None else default)

# -------- SWAP -------------
@rates_api_bp.post("/price_swap")
def price_swap():
    try:
        ctype = (request.content_type or "").lower()
        as_json = "application/json" in ctype
        p = (request.get_json(silent=True) or {}) if as_json else request.form.to_dict()

        # dates
        start = dt.date.fromisoformat(_jget(p, "start_date", dt.date.today().isoformat()))
        tenor_years = float(_jget(p, "tenor_years", 5.0))
        end = dt.date(start.year + int(tenor_years), start.month, min(start.day, 28))

        # inputs
        notional    = _ffloat(p, "notional", 1_000_000.0)
        fixed_rate  = _ffloat(p, "fixed_rate", 0.04)
        side        = _jget(p, "side", "pay_fixed")
        pay_freq    = _fint(p, "pay_freq_per_year", 2)
        dc_fixed    = _jget(p, "dc_fixed", "30/360")
        dc_float    = _jget(p, "dc_float", "ACT/360")

        col_t = _jget(p, "col_t", "tenor_years")
        col_r = _jget(p, "col_r", "zero_rate")

        # --- Discount curve (upload preferred; else path; else error) ---
        disc_file = request.files.get("disc_file")
        disc_path = _jget(p, "disc_path")

        if disc_file and disc_file.filename:
            disc = _read_curve_file(io.BytesIO(disc_file.read()), col_t, col_r)
        elif disc_path:
            try:
                disc = _read_curve_csv(disc_path, col_t, col_r)
            except Exception as ex:
                return jsonify({"ok": False, "error": f"Could not read discount curve at '{disc_path}'. Please upload the CSV. ({ex})"}), 400
        else:
            return jsonify({"ok": False, "error": "Provide a discount curve: upload 'disc_file' or set 'disc_path'."}), 400

        # --- Forward curve (upload preferred; else path; else single-curve) ---
        fwd_file = request.files.get("fwd_file")
        fwd_path = _jget(p, "fwd_path")
        if fwd_file and fwd_file.filename:
            fwd = _read_curve_file(io.BytesIO(fwd_file.read()), col_t, col_r)
        elif fwd_path:
            try:
                fwd = _read_curve_csv(fwd_path, col_t, col_r)
            except Exception as ex:
                return jsonify({"ok": False, "error": f"Could not read forward curve at '{fwd_path}'. Please upload the CSV or leave blank to use single-curve. ({ex})"}), 400
        else:
            fwd = disc  # single-curve by default

        res = price_plain_swap(
            start=start, end=end,
            notional=notional, fixed_rate=fixed_rate, side=side,
            pay_freq_per_year=pay_freq, dc_fixed=dc_fixed, dc_float=dc_float,
            disc_curve=disc, fwd_curve=fwd
        )
        res["legs"] = res["legs"].to_dict(orient="records")
        return jsonify({"ok": True, "result": res}), 200

    except Exception as e:
        # Generic safety net
        msg = str(e)
        if "not found" in msg.lower() or "missing" in msg.lower():
            msg += " | Tip: upload the CSV if the fallback path is not available."
        return jsonify({"ok": False, "error": msg}), 400

# -------- SWAPTION ---------
@rates_api_bp.post("/price_swaption")
def price_swaption():
    try:
        ctype = (request.content_type or "").lower()
        is_json = "application/json" in ctype
        p = (request.get_json(silent=True) or {}) if is_json else request.form.to_dict()

        method    = _jget(p, "method", "black").lower()
        is_payer  = (_jget(p, "is_payer", True) in [True, "true", "True", "1", 1])
        strike    = _ffloat(p, "strike", 0.04)
        pay_freq  = _fint(p, "pay_freq_per_year", 2)
        dc_fixed  = _jget(p, "dc_fixed", "30/360")
        dc_float  = _jget(p, "dc_float", "ACT/360")
        notional  = _ffloat(p, "notional", 1.0)

        # valuation context
        today      = dt.date.today()
        val_date   = dt.date.fromisoformat(_jget(p, "valuation_date", today.isoformat()))

        # option & swap dates
        option_expiry = dt.date.fromisoformat(_jget(p, "option_expiry", dt.date(today.year+1, today.month, min(today.day,28)).isoformat()))
        swap_start    = dt.date.fromisoformat(_jget(p, "swap_start", val_date.isoformat()))
        if _jget(p, "swap_end"):
            swap_end = dt.date.fromisoformat(_jget(p, "swap_end"))
        else:
            years = _ffloat(p, "tenor_years", 5.0)
            swap_end = dt.date(swap_start.year + int(years), swap_start.month, min(swap_start.day, 28))

        # sanity: expiry must be after valuation date
        if option_expiry <= val_date:
            return jsonify({"ok": False, "error": "Option expiry must be after valuation_date."}), 400

        # column names
        col_t = _jget(p, "col_t", "tenor_years")
        col_r = _jget(p, "col_r", "zero_rate")
        vol_e = _jget(p, "vol_e", "expiry")
        vol_t = _jget(p, "vol_t", "tenor")
        vol_v = _jget(p, "vol_v", "vol")

        # --- discount curve (upload OR path; fallback flat if neither) ---
        disc_file = request.files.get("disc_file")
        disc_path = _jget(p, "disc_path")
        if disc_file and disc_file.filename:
            disc = _read_curve_file(io.BytesIO(disc_file.read()), col_t, col_r)
        elif disc_path:
            try:
                disc = _read_curve_csv(disc_path, col_t, col_r)
            except Exception as ex:
                return jsonify({"ok": False, "error": f"Could not read discount curve at '{disc_path}'. Please upload the CSV or omit to use a flat 4% proxy. ({ex})"}), 400
        else:
            disc = Curve([0.5, 1, 2, 5, 10, 30], [0.04]*6)

        # --- VOL SURFACE (REQUIRED): upload OR path; helpful error if path fails ---
        vol_file = request.files.get("vol_file")
        vol_path = _jget(p, "vol_path")
        if vol_file and vol_file.filename:
            try:
                vs = VolSurface.from_file(io.BytesIO(vol_file.read()), vol_e, vol_t, vol_v, model=method)
            except Exception as ex:
                return jsonify({"ok": False, "error": f"Could not parse uploaded vol surface file. Ensure columns {vol_e}, {vol_t}, {vol_v}. ({ex})"}), 400
        elif vol_path:
            try:
                vs = VolSurface.from_csv(vol_path, vol_e, vol_t, vol_v, model=method)
            except Exception as ex:
                return jsonify({"ok": False, "error": f"Could not read vol surface at '{vol_path}'. Please upload the CSV. ({ex})"}), 400
        else:
            return jsonify({"ok": False, "error": "Vol surface is required. Upload 'vol_file' or provide 'vol_path'."}), 400

        if method in ("black", "bachelier"):
            res = price_swaption_eu(
                method=method, is_payer=is_payer, strike=strike,
                option_expiry=option_expiry, swap_start=swap_start, swap_end=swap_end,
                pay_freq_per_year=pay_freq, dc_fixed=dc_fixed,
                disc_curve=disc, vol_surf=vs, notional=notional,
                valuation_date=val_date
            )
            return jsonify({"ok": True, "result": res}), 200

        elif method == "hw_mc":
            a      = _ffloat(p, "a", 0.03)
            sigma  = _ffloat(p, "sigma", 0.01)
            paths  = _fint(p, "paths", 10000)
            stepsy = _fint(p, "steps_per_year", 24)
            res = price_hw1f_mc(
                disc_curve=disc,
                start=swap_start, end=swap_end, expiry=option_expiry,
                strike=strike, is_payer=is_payer,
                pay_freq_per_year=pay_freq, dc_fixed=dc_fixed, dc_float=dc_float,
                a=a, sigma=sigma, n_paths=paths, steps_per_year=stepsy,
                notional=notional,
                valuation_date=val_date
            )
            return jsonify({"ok": True, "result": res}), 200

        return jsonify({"ok": False, "error": "Unsupported method. Use 'black', 'bachelier', or 'hw_mc'."}), 400

    except Exception as e:
        msg = str(e)
        if "not found" in msg.lower() or "missing" in msg.lower():
            msg += " | Tip: upload the CSV if the fallback path is not available."
        return jsonify({"ok": False, "error": msg}), 400
