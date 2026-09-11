from ..models.mdls_bonds import NCFixedBonds, NCFloatingBonds
from flask import Blueprint, abort, render_template, request, json
from flask_login import current_user
import QuantLib as ql
import os
import markdown
from dotenv import load_dotenv
import logging
from copy import deepcopy
import datetime as dt

from ..extensions import db
from ..models.db_models import Instrument, PricingResult
from ..utils.lazy_imports import LazyAttribute
from ..services.validation import check_positive
from .result_state import get_latest_pricing_result_for_user
from ..models.rates_fixed_income import (
    AssetSwapTerms,
    BondForwardTreasuryLockTerms,
    BondSeriesTerms,
    CallableAmortizingBondTerms,
    CallableBondTerms,
    CapFloorTerms,
    FraTerms,
    GenericBondTerms,
    InflationLinkedBondTerms,
    LoanLeaseAnnuityTerms,
    parse_discount_factor_curve,
    parse_curve,
    parse_date,
    price_asset_swap,
    price_bond_forward_treasury_lock,
    price_bond_series,
    price_callable_amortizing_bond,
    price_callable_putable_bond,
    price_cap_floor,
    price_fra,
    price_generic_bond,
    price_inflation_linked_bond,
    price_loan_lease_annuity,
)

logger = logging.getLogger(__name__)

llm_client = LazyAttribute("derivapro.llm", "llm_client")

# Initialize Flask app
nc_bonds_bp = Blueprint("nc_bonds", __name__)

load_dotenv()

# Get the values from the environment variables
model = os.getenv("LLM_MODEL", os.getenv("Model"))


FIXED_INCOME_EXTENSION_CONFIGS = {
    "fra": {
        "title": "Forward Rate Agreement",
        "subtitle": "Price OTC forward-rate exposure using discount and forward curves.",
        "asset_class": "Fixed Income / Rates",
        "methodology_doc": "forward_rate_agreement",
        "description_title": "Single-period interest-rate forward contract with curve-based PV.",
        "description_body": (
            "A forward rate agreement locks a fixed rate for a future accrual period. "
            "The workflow projects the forward rate from the supplied forward curve, compares it with the contract rate, "
            "and discounts the payoff using the supplied discount curve."
        ),
        "chips": ["Forward curve", "Discount curve", "Day count", "Scenario shocks"],
        "fields": [
            {"name": "valuation_date", "label": "Valuation Date", "type": "date", "value": "2026-08-13"},
            {"name": "start_date", "label": "FRA Start Date", "type": "date", "value": "2027-02-13"},
            {"name": "end_date", "label": "FRA End Date", "type": "date", "value": "2027-08-13"},
            {"name": "notional", "label": "Notional", "type": "number", "step": "1000", "value": "1000000"},
            {"name": "strike_rate", "label": "Contract Rate", "type": "number", "step": "0.0001", "value": "0.0425"},
            {
                "name": "position",
                "label": "Position",
                "type": "select",
                "value": "pay_fixed",
                "options": [("pay_fixed", "Pay Fixed / Receive Floating"), ("receive_fixed", "Receive Fixed / Pay Floating")],
            },
            {
                "name": "day_count",
                "label": "Accrual Day Count",
                "type": "select",
                "value": "ACT/360",
                "options": [("ACT/360", "ACT/360"), ("ACT/365", "ACT/365"), ("30/360", "30/360")],
            },
            {"name": "scenario_shock_bp", "label": "Scenario Shock (bp)", "type": "number", "step": "1", "value": "25"},
        ],
    },
    "cap-floor": {
        "title": "Interest Rate Cap / Floor",
        "subtitle": "Value caplet or floorlet strips using Black-style rate optionality.",
        "asset_class": "Fixed Income / Rates",
        "methodology_doc": "cap_floor",
        "description_title": "A portfolio of rate options on forward reset periods.",
        "description_body": (
            "Caps and floors are strips of caplets or floorlets that reference forward rates over scheduled reset periods. "
            "This first-wave workflow values the strip with a Black rate-option approximation, supplied forward/discount curves, "
            "and user-entered volatility."
        ),
        "chips": ["Caps", "Floors", "Black caplets", "Rate/vol scenarios"],
        "fields": [
            {"name": "valuation_date", "label": "Valuation Date", "type": "date", "value": "2026-08-13"},
            {"name": "start_date", "label": "Start Date", "type": "date", "value": "2026-08-13"},
            {"name": "maturity_date", "label": "Maturity Date", "type": "date", "value": "2029-08-13"},
            {"name": "notional", "label": "Notional", "type": "number", "step": "1000", "value": "1000000"},
            {"name": "strike_rate", "label": "Strike Rate", "type": "number", "step": "0.0001", "value": "0.0450"},
            {"name": "volatility", "label": "Forward Rate Volatility", "type": "number", "step": "0.001", "value": "0.25"},
            {
                "name": "option_type",
                "label": "Product Type",
                "type": "select",
                "value": "cap",
                "options": [("cap", "Cap"), ("floor", "Floor")],
            },
            {
                "name": "payments_per_year",
                "label": "Payments / Year",
                "type": "select",
                "value": "4",
                "options": [("1", "Annual"), ("2", "Semiannual"), ("4", "Quarterly")],
            },
            {
                "name": "day_count",
                "label": "Accrual Day Count",
                "type": "select",
                "value": "ACT/360",
                "options": [("ACT/360", "ACT/360"), ("ACT/365", "ACT/365"), ("30/360", "30/360")],
            },
            {"name": "scenario_rate_shock_bp", "label": "Rate Shock (bp)", "type": "number", "step": "1", "value": "25"},
            {"name": "scenario_vol_shock", "label": "Vol Shock", "type": "number", "step": "0.001", "value": "0.05"},
        ],
    },
    "callable-amortizing-bond": {
        "title": "Callable Amortizing Bond",
        "subtitle": "Value callable or putable bonds with amortizing notionals, variable coupons, and scheduled exercise rights.",
        "asset_class": "Fixed Income",
        "methodology_doc": "callable_amortizing_bond",
        "description_title": "A schedule-driven callable bond workspace for American and Bermudan exercise structures.",
        "description_body": (
            "Callable amortizing bonds combine contractual coupon/principal schedules with issuer call or investor put rights. "
            "This workspace uses explicit payment and exercise schedules, then values the embedded optionality with a transparent "
            "short-rate lattice and reports option-adjusted PV, OAS, effective duration, yield diagnostics, and exercise probabilities."
        ),
        "chips": ["Amortizing notional", "Step-up coupons", "Bermudan exercise", "American approximation", "OAS"],
        "field_sections": [
            {
                "title": "Instrument Terms",
                "description": "Define the contractual bond schedule, clean-price reference, and accrual convention.",
                "icon": "T",
                "fields": [
                    {"name": "valuation_date", "label": "Settlement / Value Date", "type": "date", "value": "2019-09-24"},
                    {"name": "effective_date", "label": "Effective Date", "type": "date", "value": "2019-09-24", "hint": "Contract effective date from the benchmark setup."},
                    {"name": "dated_date", "label": "Dated Date / Accrual Start", "type": "date", "value": "2019-09-24", "hint": "The source leaves this optional field blank. Its contractual default is the effective date, shown explicitly here to avoid browser date placeholders."},
                    {"name": "first_coupon_date", "label": "First Coupon Date", "type": "hidden", "value": ""},
                    {"name": "last_coupon_date", "label": "Penultimate Coupon Date", "type": "hidden", "value": ""},
                    {"name": "maturity_date", "label": "Maturity / Terminating Date", "type": "date", "value": "2024-09-24", "hint": "Contractual final payment date. Coupon terms cutoff dates may extend beyond this date."},
                    {"name": "schedule_mode", "label": "Payment Schedule Source", "type": "select", "value": "period_terms", "options": [("period_terms", "Coupon Terms Table (Generate Payment Dates)"), ("explicit", "Explicit Payment Table"), ("generated_bullet", "Generated Bullet Schedule"), ("generated_straight_line", "Generated Straight-Line Amortization")], "hint": "The benchmark uses the coupon terms table. Each row applies through its cutoff while frequency and maturity generate the actual payment dates."},
                    {"name": "notional", "label": "Original Face Value", "type": "number", "step": "any", "value": "100"},
                    {"name": "coupon_rate", "label": "Base Coupon Rate", "type": "number", "step": "0.0001", "value": "0.0500"},
                    {"name": "market_clean_price_pct", "label": "OAS Target Clean Price (% of Par)", "type": "number", "step": "0.01", "value": "100.00", "hint": "Used only for OAS and yield diagnostics. It does not affect model fair value or embedded option value."},
                    {
                        "name": "payments_per_year",
                        "label": "Coupon Frequency",
                        "type": "select",
                        "value": "2",
                        "options": [("1", "Annual"), ("2", "Semiannual"), ("4", "Quarterly")],
                    },
                    {
                        "name": "day_count",
                        "label": "Accrual Method",
                        "type": "select",
                        "value": "30/360 ISDA",
                        "options": [("ACT/ACT ISMA", "Actual/Actual (ISMA)"), ("30/360 ISDA", "30/360 ISDA"), ("30/360", "30/360"), ("ACT/360", "ACT/360"), ("ACT/365", "ACT/365")],
                    },
                    {"name": "amortization_style", "label": "Amortization Style", "type": "hidden", "value": "sinking_schedule"},
                ],
            },
            {
                "title": "Calendar & Conventions",
                "description": "Capture date-adjustment and notification assumptions separately from discount-curve construction.",
                "icon": "D",
                "fields": [
                    {
                        "name": "business_day_convention",
                        "label": "Business Day Convention",
                        "type": "select",
                        "value": "none",
                        "options": [("none", "No Adjustment"), ("following", "Following"), ("modified_following", "Modified Following")],
                        "hint": "Adjusts payment dates using weekends and the holiday table. Accrual dates remain contractual.",
                    },
                    {"name": "notification_days", "label": "Notification Days (Calendar)", "type": "number", "step": "1", "value": "30", "hint": "Exercise is decided this many calendar days before redemption. Cashflows during the notice period are retained."},
                    {
                        "name": "holiday_dates",
                        "label": "Holiday Dates",
                        "type": "hidden",
                        "value": "2004-12-31;2008-02-13;2011-03-28;2014-05-10;2017-06-22;2020-08-04;2023-09-17;2026-10-30;2029-12-12",
                    },
                ],
            },
            {
                "title": "Exercise & Model Configuration",
                "description": "Choose exercise rights, exercise style, lattice granularity, and short-rate model assumptions.",
                "icon": "M",
                "fields": [
                    {
                        "name": "option_rights",
                        "label": "Embedded Option Rights",
                        "type": "select",
                        "value": "callable",
                        "options": [("callable", "Callable"), ("putable", "Putable"), ("callable_putable", "Callable + Putable")],
                    },
                    {
                        "name": "exercise_style",
                        "label": "Exercise Style",
                        "type": "select",
                        "value": "bermudan",
                        "options": [("bermudan", "Bermudan (End Dates)"), ("american_grid", "American (Window Grid)")],
                    },
                    {"name": "exercise_schedule_source", "label": "Exercise Schedule Source", "type": "select", "value": "custom", "options": [("custom", "Editable Exercise Table"), ("generated", "Generate from Coupon Dates")], "hint": "The benchmark uses its supplied exercise table. Generator fields appear only when generation is selected."},
                    {"name": "first_exercise_date", "label": "Generator First Exercise Date", "type": "date", "value": "2019-09-24"},
                    {"name": "exercise_price_pct", "label": "Generator Exercise Price (% Outstanding)", "type": "number", "step": "0.01", "value": "100.00"},
                    {
                        "name": "short_rate_model",
                        "label": "Short-Rate Model",
                        "type": "select",
                        "value": "hull_white",
                        "options": [
                            ("hull_white", "Hull-White 1F (Normal)"),
                            ("black_karasinski", "Black-Karasinski 1F (Lognormal)"),
                        ],
                        "hint": "Curve-fitted trinomial tree. Model parameters are supplied rather than calibrated here.",
                    },
                    {"name": "short_rate_volatility_pct", "label": "Short-Rate Volatility (%)", "type": "number", "step": "0.01", "value": "20.00", "hint": "Benchmark: 20.00% (0.20 internally). Hull-White uses absolute instantaneous rate volatility; production inputs should be calibrated."},
                    {"name": "short_rate_mean_reversion_pct", "label": "Mean Reversion (%)", "type": "number", "step": "0.01", "value": "0.50", "hint": "Benchmark input: 0.50%, equivalent to model parameter a = 0.005."},
                    {"name": "lattice_steps_per_period", "label": "Tree Refinement / Steps per Coupon Period", "type": "number", "step": "1", "value": "5"},
                    {
                        "name": "tree_generation",
                        "label": "Tree Generation",
                        "type": "select",
                        "value": "maturity",
                        "options": [
                            ("maturity", "To Instrument Maturity Date"),
                            ("last_callable_date", "Truncated at Last Callable Date"),
                        ],
                    },
                    {"name": "scenario_shock_bp", "label": "Scenario Shock (bp)", "type": "number", "step": "1", "value": "1"},
                    {"name": "coupon_schedule", "label": "Coupon Schedule", "type": "hidden", "value": ""},
                    {"name": "principal_schedule", "label": "Principal Schedule", "type": "hidden", "value": ""},
                    {"name": "fixed_payment_schedule", "label": "Fixed Payment Schedule", "type": "hidden", "value": ""},
                    {"name": "fixed_payment_treatment", "label": "Fixed Payment Treatment", "type": "hidden", "value": "principal_redemption"},
                    {
                        "name": "cashflow_schedule",
                        "label": "Payment Schedule",
                        "type": "hidden",
                        "value": "2020-12-20|100|0.0500|0|0;2022-06-20|100|0.0550|0|20;2022-12-20|80|0.0550|0|0;2025-06-20|80|0.0550|0|80",
                    },
                    {
                        "name": "exercise_schedule",
                        "label": "Exercise Schedule",
                        "type": "hidden",
                        "value": "2019-09-24|2020-09-24|100|0;2020-09-24|2021-09-24|100|0;2021-09-24|2022-09-24|102|0;2022-09-24|2024-09-24|105|0",
                    },
                ],
            },
        ],
        "curve_defaults": {
            "discount_curve_input_type": "discount_factors",
            "interpolation_method": "linear_discount",
        },
    },
    "callable-putable-bond": {
        "title": "Callable / Putable Bond",
        "subtitle": "Evaluate embedded exercise optionality plus yield-to-best and yield-to-worst cases.",
        "asset_class": "Fixed Income",
        "methodology_doc": "callable_putable_bond",
        "description_title": "Fixed-rate bond plus embedded interest-rate optionality and exercise-date yield diagnostics.",
        "description_body": (
            "Callable and putable bonds extend the existing fixed-rate bond workflow by adding exercise optionality. "
            "This page compares straight-bond PV with an option-adjusted value from a transparent short-rate lattice approximation, "
            "then calculates yield-to-best and yield-to-worst across maturity and eligible exercise-date cases."
        ),
        "chips": ["Callable bonds", "Putable bonds", "Short-rate lattice", "Yield to worst"],
        "fields": [
            {"name": "valuation_date", "label": "Valuation Date", "type": "date", "value": "2026-08-13"},
            {"name": "maturity_date", "label": "Maturity Date", "type": "date", "value": "2031-08-13"},
            {"name": "notional", "label": "Face Value", "type": "number", "step": "1000", "value": "1000000"},
            {"name": "coupon_rate", "label": "Coupon Rate", "type": "number", "step": "0.0001", "value": "0.0550"},
            {"name": "market_clean_price_pct", "label": "Market Clean Price (% of Par)", "type": "number", "step": "0.01", "value": "100.00"},
            {
                "name": "payments_per_year",
                "label": "Coupon Frequency",
                "type": "select",
                "value": "2",
                "options": [("1", "Annual"), ("2", "Semiannual"), ("4", "Quarterly")],
            },
            {
                "name": "option_type",
                "label": "Embedded Option",
                "type": "select",
                "value": "callable",
                "options": [("callable", "Callable"), ("putable", "Putable")],
            },
            {"name": "call_or_put_price", "label": "Exercise Price (% of Par)", "type": "number", "step": "0.01", "value": "100.00"},
            {"name": "first_exercise_year", "label": "First Exercise Year", "type": "number", "step": "0.25", "value": "2.0"},
            {"name": "short_rate_volatility", "label": "Short-Rate Volatility", "type": "number", "step": "0.001", "value": "0.015"},
            {
                "name": "day_count",
                "label": "Coupon Day Count",
                "type": "select",
                "value": "30/360",
                "options": [("30/360", "30/360"), ("ACT/360", "ACT/360"), ("ACT/365", "ACT/365")],
            },
            {"name": "scenario_shock_bp", "label": "Scenario Shock (bp)", "type": "number", "step": "1", "value": "25"},
        ],
    },
    "level-coupon-bond": {
        "title": "Level Coupon Bond",
        "subtitle": "Value standard bullet fixed-rate bonds with term-structure discounting.",
        "asset_class": "Fixed Income",
        "methodology_doc": "level_coupon_bond",
        "description_title": "Plain fixed-rate bond cash-flow valuation under a supplied zero-rate curve.",
        "description_body": (
            "This workspace standardizes the legacy fixed-rate bond workflow into the product-standard layout. "
            "It generates periodic coupons, terminal principal, clean-price yield, duration, convexity, DV01, and rate-shock scenarios."
        ),
        "chips": ["Level coupon", "Bullet principal", "YTM", "DV01"],
        "fields": [
            {"name": "valuation_date", "label": "Valuation Date", "type": "date", "value": "2026-08-13"},
            {"name": "maturity_date", "label": "Maturity Date", "type": "date", "value": "2031-08-13"},
            {"name": "notional", "label": "Face Value", "type": "number", "step": "1000", "value": "1000000"},
            {"name": "coupon_rate", "label": "Coupon Rate", "type": "number", "step": "0.0001", "value": "0.0500"},
            {"name": "market_clean_price_pct", "label": "Market Clean Price (% of Par)", "type": "number", "step": "0.01", "value": "100.00"},
            {
                "name": "payments_per_year",
                "label": "Coupon Frequency",
                "type": "select",
                "value": "2",
                "options": [("1", "Annual"), ("2", "Semiannual"), ("4", "Quarterly")],
            },
            {
                "name": "day_count",
                "label": "Coupon Day Count",
                "type": "select",
                "value": "30/360",
                "options": [("30/360", "30/360"), ("ACT/360", "ACT/360"), ("ACT/365", "ACT/365")],
            },
            {"name": "scenario_shock_bp", "label": "Scenario Shock (bp)", "type": "number", "step": "1", "value": "25"},
        ],
    },
    "amortizing-stepup-sinking-bond": {
        "title": "Structured Amortizing Bonds",
        "subtitle": "Configure amortizing, step-up, step-down, and sinking-fund bond structures.",
        "asset_class": "Fixed Income",
        "methodology_doc": "amortizing_stepup_sinking_bond",
        "description_title": "Generic bond cash flows with coupon schedules, principal schedules, and clean/dirty price diagnostics.",
        "description_body": (
            "This workspace covers bonds whose coupon and outstanding balance vary across contractual periods. "
            "It supports yield-based clean-price benchmarks, straight-line amortization, bullet principal, and explicit sinking schedules."
        ),
        "chips": ["Amortizing", "Step-up coupon", "Sinking fund", "Clean/dirty price"],
        "field_sections": [
            {
                "title": "Contract Dates",
                "description": "Define settlement, accrual anchors, coupon boundaries, and contractual maturity.",
                "icon": "D",
                "fields": [
                    {"name": "valuation_date", "label": "Settlement / Value Date", "type": "date", "value": "2026-08-30"},
                    {"name": "dated_date", "label": "Dated Date / Accrual Start", "type": "date", "value": "2026-06-20"},
                    {"name": "first_coupon_date", "label": "First Coupon Date", "type": "date", "value": "2026-12-20"},
                    {"name": "last_coupon_date", "label": "Penultimate Coupon Date", "type": "date", "value": "2040-12-20"},
                    {"name": "maturity_date", "label": "Maturity / Final Payment Date", "type": "date", "value": "2041-06-20"},
                ],
            },
            {
                "title": "Economics & Conventions",
                "description": "Set the pricing basis, principal profile, coupon convention, and parallel-rate scenario.",
                "icon": "T",
                "fields": [
                    {"name": "notional", "label": "Original Face Value", "type": "number", "step": "1000", "value": "1000000"},
                    {"name": "coupon_rate", "label": "Base Coupon Rate", "type": "number", "step": "0.0001", "value": "0.0500"},
                    {
                        "name": "pricing_basis",
                        "label": "Pricing Basis",
                        "type": "select",
                        "value": "yield",
                        "options": [("yield", "Price from Yield"), ("curve", "Price from Curve")],
                        "hint": "Use Price from Yield for clean-price benchmarking against external calculators.",
                    },
                    {"name": "yield_to_maturity", "label": "Yield to Maturity", "type": "number", "step": "0.0001", "value": "0.0600"},
                    {"name": "market_clean_price_pct", "label": "Market Clean Price (% of Par)", "type": "number", "step": "0.01", "value": "100.00"},
                    {
                        "name": "amortization_style",
                        "label": "Principal Schedule Type",
                        "type": "select",
                        "value": "straight_line",
                        "options": [("bullet", "Bullet"), ("straight_line", "Straight-Line Amortization"), ("sinking_schedule", "Sinking Schedule")],
                    },
                    {"name": "coupon_schedule", "label": "Coupon Schedule", "type": "hidden", "value": ""},
                    {"name": "principal_schedule", "label": "Sinking Schedule", "type": "hidden", "value": ""},
                    {"name": "cashflow_schedule", "label": "Payment Schedule", "type": "hidden", "value": ""},
                    {
                        "name": "payments_per_year",
                        "label": "Coupon Frequency",
                        "type": "select",
                        "value": "2",
                        "options": [("1", "Annual"), ("2", "Semiannual"), ("4", "Quarterly")],
                    },
                    {
                        "name": "day_count",
                        "label": "Accrual Method",
                        "type": "select",
                        "value": "ACT/ACT ISMA",
                        "options": [("ACT/ACT ISMA", "Actual/Actual (ISMA)"), ("30/360", "30/360"), ("ACT/360", "ACT/360"), ("ACT/365", "ACT/365")],
                    },
                    {"name": "scenario_shock_bp", "label": "Scenario Shock (bp)", "type": "number", "step": "1", "value": "25"},
                ],
            },
        ],
    },
    "custom-structured-bond": {
        "title": "Custom Structured Bond",
        "subtitle": "Build user-defined fixed-income cash flows from coupon, principal, and fixed-payment tables.",
        "asset_class": "Fixed Income",
        "methodology_doc": "custom_structured_bond",
        "description_title": "Configurable bond cash-flow workbench for non-standard contractual features.",
        "description_body": (
            "Custom structured bonds use the generic bond engine with user-defined coupon, principal, and fixed-payment schedules. "
            "This is the foundation for structured fixed-income instruments that do not fit a plain bullet or amortizing template."
        ),
        "chips": ["Coupon table", "Principal table", "Fixed payments", "Generic PV"],
        "fields": [
            {"name": "valuation_date", "label": "Valuation Date", "type": "date", "value": "2026-08-13"},
            {"name": "maturity_date", "label": "Maturity Date", "type": "date", "value": "2033-08-13"},
            {"name": "notional", "label": "Original Face Value", "type": "number", "step": "1000", "value": "1000000"},
            {"name": "coupon_rate", "label": "Fallback Coupon Rate", "type": "number", "step": "0.0001", "value": "0.0400"},
            {"name": "market_clean_price_pct", "label": "Market Clean Price (% of Par)", "type": "number", "step": "0.01", "value": "99.50"},
            {"name": "coupon_schedule", "label": "Coupon Schedule", "type": "text", "value": "2026-08-13:0.04,2028-08-13:0.05,2031-08-13:0.06", "hint": "Optional effective-date schedule: YYYY-MM-DD:rate, ..."},
            {"name": "principal_schedule", "label": "Principal / Sink Schedule", "type": "text", "value": "2030-08-13:0.25,2032-08-13:0.25", "hint": "Optional date:pct_original schedule."},
            {"name": "fixed_payment_schedule", "label": "Fixed Payment Schedule", "type": "text", "value": "2028-08-13:25000,2031-08-13:25000", "hint": "Optional date:amount schedule for additional fixed cash flows."},
            {
                "name": "payments_per_year",
                "label": "Payment Frequency",
                "type": "select",
                "value": "2",
                "options": [("1", "Annual"), ("2", "Semiannual"), ("4", "Quarterly")],
            },
            {
                "name": "day_count",
                "label": "Day Count",
                "type": "select",
                "value": "30/360",
                "options": [("30/360", "30/360"), ("ACT/360", "ACT/360"), ("ACT/365", "ACT/365")],
            },
            {"name": "scenario_shock_bp", "label": "Scenario Shock (bp)", "type": "number", "step": "1", "value": "25"},
        ],
    },
    "bond-series": {
        "title": "Bond Series",
        "subtitle": "Aggregate serial bond maturities into issuer-level PV, yield, and maturity analytics.",
        "asset_class": "Fixed Income",
        "methodology_doc": "bond_series",
        "description_title": "Multi-maturity bond program valuation and cash-flow aggregation.",
        "description_body": (
            "Bond series analytics value multiple related maturities together, then aggregate PV, yield, weighted-average maturity, "
            "and series-level scenario exposure. This is useful for municipal, issuer program, and tranche-style fixed-income books."
        ),
        "chips": ["Serial bonds", "Aggregate cash flows", "Series yield", "Program risk"],
        "fields": [
            {"name": "valuation_date", "label": "Valuation Date", "type": "date", "value": "2026-08-13"},
            {"name": "series_table", "label": "Bond Series Table", "type": "textarea", "value": "2028-08-13|1000000|0.040|100; 2030-08-13|1500000|0.045|100; 2032-08-13|2000000|0.050|100", "hint": "Rows use maturity|principal|coupon_rate|redemption_pct separated by semicolons."},
            {
                "name": "day_count",
                "label": "Day Count",
                "type": "select",
                "value": "30/360",
                "options": [("30/360", "30/360"), ("ACT/360", "ACT/360"), ("ACT/365", "ACT/365")],
            },
            {"name": "scenario_shock_bp", "label": "Scenario Shock (bp)", "type": "number", "step": "1", "value": "25"},
        ],
    },
    "loan-lease-annuity": {
        "title": "Loans / Leases / Annuities",
        "subtitle": "Value scheduled contractual payment streams with common amortization patterns.",
        "asset_class": "Fixed Income / Cash Flows",
        "methodology_doc": "loan_lease_annuity",
        "description_title": "Present-value cash-flow analytics for level-payment, equal-principal, and interest-only structures.",
        "description_body": (
            "This workspace covers loan-style, lease-style, and annuity-style cash-flow instruments using scheduled payments, "
            "contractual interest, principal amortization, discounting, and rate-shock scenarios."
        ),
        "chips": ["PVCF", "Level payment", "Equal principal", "Annuity"],
        "fields": [
            {"name": "valuation_date", "label": "Valuation Date", "type": "date", "value": "2026-08-13"},
            {"name": "start_date", "label": "Start Date", "type": "date", "value": "2026-08-13"},
            {"name": "maturity_date", "label": "Maturity Date", "type": "date", "value": "2031-08-13"},
            {"name": "principal", "label": "Principal / Financed Amount", "type": "number", "step": "1000", "value": "1000000"},
            {"name": "contract_rate", "label": "Contract Rate", "type": "number", "step": "0.0001", "value": "0.0550"},
            {
                "name": "structure_type",
                "label": "Payment Structure",
                "type": "select",
                "value": "level_payment",
                "options": [("level_payment", "Level Payment"), ("equal_principal", "Equal Principal"), ("interest_only", "Interest Only / Bullet")],
            },
            {
                "name": "payments_per_year",
                "label": "Payment Frequency",
                "type": "select",
                "value": "12",
                "options": [("1", "Annual"), ("2", "Semiannual"), ("4", "Quarterly"), ("12", "Monthly")],
            },
            {
                "name": "day_count",
                "label": "Day Count",
                "type": "select",
                "value": "ACT/365",
                "options": [("ACT/365", "ACT/365"), ("ACT/360", "ACT/360"), ("30/360", "30/360")],
            },
            {"name": "scenario_shock_bp", "label": "Scenario Shock (bp)", "type": "number", "step": "1", "value": "25"},
        ],
    },
    "asset-swap": {
        "title": "Asset Swap",
        "subtitle": "Analyze bond-versus-swap relative value and par asset-swap spread.",
        "asset_class": "Fixed Income",
        "methodology_doc": "asset_swap",
        "description_title": "A funded bond position transformed into floating-rate exposure.",
        "description_body": (
            "Asset swaps combine a cash bond with an interest-rate swap overlay. "
            "This workflow compares the bond coupon stream with the par swap rate and estimates the par asset-swap spread "
            "using user-supplied bond price and discount-curve assumptions."
        ),
        "chips": ["Bond relative value", "Par ASW spread", "Curve PV", "Spread scenarios"],
        "fields": [
            {"name": "valuation_date", "label": "Valuation Date", "type": "date", "value": "2026-08-13"},
            {"name": "maturity_date", "label": "Bond Maturity Date", "type": "date", "value": "2031-08-13"},
            {"name": "notional", "label": "Face Value", "type": "number", "step": "1000", "value": "1000000"},
            {"name": "coupon_rate", "label": "Bond Coupon Rate", "type": "number", "step": "0.0001", "value": "0.0525"},
            {"name": "market_clean_price_pct", "label": "Market Clean Price (% of Par)", "type": "number", "step": "0.01", "value": "99.25"},
            {"name": "quoted_asset_swap_spread_bp", "label": "Quoted ASW Spread (bp)", "type": "number", "step": "1", "value": "85"},
            {
                "name": "payments_per_year",
                "label": "Coupon Frequency",
                "type": "select",
                "value": "2",
                "options": [("1", "Annual"), ("2", "Semiannual"), ("4", "Quarterly")],
            },
            {
                "name": "day_count",
                "label": "Coupon Day Count",
                "type": "select",
                "value": "30/360",
                "options": [("30/360", "30/360"), ("ACT/360", "ACT/360"), ("ACT/365", "ACT/365")],
            },
            {"name": "scenario_rate_shock_bp", "label": "Rate Shock (bp)", "type": "number", "step": "1", "value": "25"},
            {"name": "scenario_spread_shock_bp", "label": "Spread Shock (bp)", "type": "number", "step": "1", "value": "25"},
        ],
    },
    "inflation-linked-bond": {
        "title": "Inflation-Linked Bond",
        "subtitle": "Project indexed coupons and principal using CPI and real-rate assumptions.",
        "asset_class": "Fixed Income / Inflation",
        "methodology_doc": "inflation_linked_bond",
        "description_title": "A bond whose coupon and redemption cash flows are linked to an inflation index.",
        "description_body": (
            "Inflation-linked bonds adjust coupons and principal by an index ratio. "
            "This first-pass workflow supports CPI inputs, indexation lag, principal-floor treatment, real discounting, "
            "and inflation/real-rate scenario analysis."
        ),
        "chips": ["CPI indexation", "Real discounting", "Index lag", "Inflation scenarios"],
        "fields": [
            {"name": "valuation_date", "label": "Valuation Date", "type": "date", "value": "2026-08-13"},
            {"name": "maturity_date", "label": "Maturity Date", "type": "date", "value": "2036-08-13"},
            {"name": "notional", "label": "Face Value", "type": "number", "step": "1000", "value": "1000000"},
            {"name": "real_coupon_rate", "label": "Real Coupon Rate", "type": "number", "step": "0.0001", "value": "0.0125"},
            {"name": "base_cpi", "label": "Base CPI", "type": "number", "step": "0.01", "value": "300.00"},
            {"name": "current_cpi", "label": "Current CPI", "type": "number", "step": "0.01", "value": "318.50"},
            {"name": "annual_inflation_rate", "label": "Projected Inflation Rate", "type": "number", "step": "0.0001", "value": "0.0225"},
            {"name": "indexation_lag_months", "label": "Indexation Lag (Months)", "type": "number", "step": "1", "value": "3"},
            {
                "name": "principal_floor",
                "label": "Principal Floor",
                "type": "select",
                "value": "yes",
                "options": [("yes", "Floor at Par"), ("no", "No Floor")],
            },
            {
                "name": "payments_per_year",
                "label": "Coupon Frequency",
                "type": "select",
                "value": "2",
                "options": [("1", "Annual"), ("2", "Semiannual"), ("4", "Quarterly")],
            },
            {
                "name": "day_count",
                "label": "Coupon Day Count",
                "type": "select",
                "value": "ACT/365",
                "options": [("ACT/365", "ACT/365"), ("ACT/360", "ACT/360"), ("30/360", "30/360")],
            },
            {"name": "scenario_real_rate_shock_bp", "label": "Real Rate Shock (bp)", "type": "number", "step": "1", "value": "25"},
            {"name": "scenario_inflation_shock_bp", "label": "Inflation Shock (bp)", "type": "number", "step": "1", "value": "50"},
        ],
    },
    "bond-forward-treasury-lock": {
        "title": "Bond Forward / Treasury Lock",
        "subtitle": "Estimate forward bond price and duration-based treasury-lock PV.",
        "asset_class": "Fixed Income / Rates",
        "methodology_doc": "bond_forward_treasury_lock",
        "description_title": "Forward delivery and rate-lock analytics for bond-linked exposure.",
        "description_body": (
            "Bond forwards and treasury locks are used to hedge or express future rate exposure. "
            "This workflow applies cost-of-carry pricing to a spot dirty bond price and estimates treasury-lock PV "
            "from the implied forward yield versus the locked yield."
        ),
        "chips": ["Bond forwards", "Treasury locks", "Cost of carry", "Duration PV"],
        "fields": [
            {"name": "valuation_date", "label": "Valuation Date", "type": "date", "value": "2026-08-13"},
            {"name": "delivery_date", "label": "Forward Delivery Date", "type": "date", "value": "2027-02-13"},
            {"name": "bond_maturity_date", "label": "Bond Maturity Date", "type": "date", "value": "2031-08-13"},
            {"name": "notional", "label": "Face Value", "type": "number", "step": "1000", "value": "1000000"},
            {"name": "coupon_rate", "label": "Bond Coupon Rate", "type": "number", "step": "0.0001", "value": "0.0475"},
            {"name": "spot_dirty_price_pct", "label": "Spot Dirty Price (% of Par)", "type": "number", "step": "0.01", "value": "101.25"},
            {"name": "financing_rate", "label": "Financing / Repo Rate", "type": "number", "step": "0.0001", "value": "0.0425"},
            {"name": "locked_forward_yield", "label": "Locked Forward Yield", "type": "number", "step": "0.0001", "value": "0.0450"},
            {"name": "modified_duration", "label": "Forward Bond Modified Duration", "type": "number", "step": "0.01", "value": "4.25"},
            {
                "name": "position",
                "label": "Treasury Lock Position",
                "type": "select",
                "value": "receive_fixed",
                "options": [("receive_fixed", "Receive Fixed / Long Duration"), ("pay_fixed", "Pay Fixed / Short Duration")],
            },
            {
                "name": "payments_per_year",
                "label": "Coupon Frequency",
                "type": "select",
                "value": "2",
                "options": [("1", "Annual"), ("2", "Semiannual"), ("4", "Quarterly")],
            },
            {
                "name": "day_count",
                "label": "Coupon Day Count",
                "type": "select",
                "value": "30/360",
                "options": [("30/360", "30/360"), ("ACT/360", "ACT/360"), ("ACT/365", "ACT/365")],
            },
            {"name": "scenario_rate_shock_bp", "label": "Rate Shock (bp)", "type": "number", "step": "1", "value": "25"},
        ],
    },
}


CURVE_FIELD_DEFAULTS = {
    "discount_curve_input_type": "zero_rates",
    "discount_curve_tenors": "0.25,0.5,1,2,3,5,7,10",
    "discount_curve_rates": "0.0400,0.0410,0.0420,0.0430,0.0440,0.0450,0.0460,0.0470",
    "discount_factor_curve": (
        "2019-09-24|1;2019-09-27|0.99959907;2019-10-01|0.99906474;2019-10-24|0.99599788;"
        "2019-11-24|0.99187918;2019-12-24|0.98790956;2020-03-16|0.97714014;"
        "2020-06-15|0.96532608;2020-09-21|0.95276295;2020-12-21|0.94124363;"
        "2021-03-15|0.93073406;2021-06-21|0.91862120;2021-09-20|0.90751459;"
        "2021-09-24|0.90702948;2022-09-24|0.86372214;2023-09-24|0.82259251;"
        "2024-09-24|0.78342144;2025-09-24|0.74611566;2026-09-24|0.71049136;"
        "2027-09-24|0.67665844;2028-09-24|0.64443661;2029-09-24|0.61374915;"
        "2030-09-24|0.58444487"
    ),
    "forward_curve_tenors": "0.25,0.5,1,2,3,5,7,10",
    "forward_curve_rates": "0.0410,0.0420,0.0430,0.0440,0.0450,0.0460,0.0470,0.0480",
}


def ask_gpt(question):
    """Send a request to the configured LLM provider and return text."""
    try:
        return llm_client.generate_response(prompt=question, model=model)
    except Exception as e:
        error_msg = str(e)
        logger.exception("Error occurred while calling LLM provider")

        if "403" in error_msg:
            logger.error(
                "Authentication failed or access is restricted for the LLM provider."
            )
            return "Error: Access to the AI service is currently restricted. Please verify the API configuration or contact support."
        elif "401" in error_msg:
            logger.error("Unauthorized access attempt. Please check API credentials.")
            return "Error: Authentication failed. Please verify the API configuration or contact support."
        elif "429" in error_msg:
            logger.error("Rate limit exceeded for LLM provider")
            return "Error: Too many requests. Please wait a moment and try again."
        else:
            logger.error("Unexpected error in LLM provider call")
            return f"An error occurred while generating the assessment. Please try again. Error details: {error_msg}"


def _config_fields(config):
    if "field_sections" in config:
        return [
            field
            for section in config["field_sections"]
            for field in section["fields"]
        ]
    return config["fields"]


def _default_form_data(config):
    data = {field["name"]: field["value"] for field in _config_fields(config)}
    data.update(CURVE_FIELD_DEFAULTS)
    data.update(config.get("curve_defaults", {}))
    return data


def _callable_amortizing_benchmark_presets(config):
    base = _default_form_data(config)
    presets = [
        {
            "id": "callable_benchmark",
            "label": "Callable benchmark",
            "meta": "20% volatility",
            "description": "External callable-bond benchmark with the original sinking schedule.",
            "values": base,
        },
        {
            "id": "putable_benchmark",
            "label": "Puttable benchmark",
            "meta": "10% volatility",
            "description": "Same bond and curve with investor put rights replacing issuer calls.",
            "values": {
                **base,
                "option_rights": "putable",
                "short_rate_volatility_pct": "10.00",
                "exercise_schedule": (
                    "2019-09-24|2020-09-24|0|100;"
                    "2020-09-24|2021-09-24|0|100;"
                    "2021-09-24|2022-09-24|0|102;"
                    "2022-09-24|2024-09-24|0|105"
                ),
            },
        },
        {
            "id": "faster_amortization",
            "label": "Faster amortization",
            "meta": "15% volatility",
            "description": "Callable benchmark with four 25-point principal redemptions.",
            "values": {
                **base,
                "short_rate_volatility_pct": "15.00",
                "cashflow_schedule": (
                    "2020-12-20|100|0.0500|0|25;"
                    "2022-06-20|75|0.0550|0|25;"
                    "2022-12-20|50|0.0550|0|25;"
                    "2025-06-20|25|0.0550|0|25"
                ),
            },
        },
    ]
    return presets


def _extension_form_data(config):
    data = _default_form_data(config)
    if request.method == "POST":
        for key in data:
            data[key] = request.form.get(key, data[key])
    return data


def _float_value(data, key):
    return float(data[key])


def _int_value(data, key):
    return int(float(data[key]))


def _discount_curve_from_form(form_data):
    interpolation_method = form_data.get("interpolation_method", "linear_zero")
    if form_data.get("discount_curve_input_type") == "discount_factors":
        return parse_discount_factor_curve(
            parse_date(form_data["valuation_date"]),
            form_data.get("discount_factor_curve", ""),
            interpolation_method,
        )
    return parse_curve(
        form_data["discount_curve_tenors"],
        form_data["discount_curve_rates"],
        interpolation_method,
    )


def _add_months_preserve_day(d: dt.date, months: int) -> dt.date:
    year = d.year + (d.month - 1 + months) // 12
    month = (d.month - 1 + months) % 12 + 1
    if month == 12:
        next_month = dt.date(year + 1, 1, 1)
    else:
        next_month = dt.date(year, month + 1, 1)
    last_day = (next_month - dt.timedelta(days=1)).day
    return dt.date(year, month, min(d.day, last_day))


def _format_schedule_number(value):
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _parse_payment_schedule_for_display(raw):
    rows = []
    if not raw:
        return rows
    for item in raw.replace("\n", ";").split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split("|")]
        if len(parts) not in {4, 5}:
            continue
        rows.append(
            {
                "payment_date": parts[0],
                "opening_notional": parts[1],
                "coupon_rate": parts[2],
                "principal": parts[3],
                "fixed_payment": parts[4] if len(parts) == 5 else "0",
            }
        )
    return rows


def _parse_exercise_schedule_for_display(raw):
    rows = []
    if not raw:
        return rows
    for item in raw.replace("\n", ";").split(";"):
        item = item.strip()
        if not item:
            continue
        parts = [part.strip() for part in item.split("|")]
        if len(parts) != 4:
            continue
        rows.append(
            {
                "start_date": parts[0],
                "end_date": parts[1],
                "call_price_pct": parts[2],
                "put_price_pct": parts[3],
            }
        )
    return rows


def _parse_discount_factor_rows_for_display(raw):
    rows = []
    if not raw:
        return rows
    for item in raw.replace("\n", ";").split(";"):
        item = item.strip()
        if not item:
            continue
        separator = "|" if "|" in item else ":"
        parts = [part.strip() for part in item.split(separator, 1)]
        if len(parts) != 2:
            continue
        rows.append({"curve_date": parts[0], "discount_factor": parts[1]})
    return rows


def _parse_zero_curve_rows_for_display(tenors_raw, rates_raw):
    tenors = [item.strip() for item in (tenors_raw or "").split(",") if item.strip()]
    rates = [item.strip() for item in (rates_raw or "").split(",") if item.strip()]
    row_count = max(len(tenors), len(rates))
    return [
        {
            "tenor": tenors[idx] if idx < len(tenors) else "",
            "zero_rate": rates[idx] if idx < len(rates) else "",
        }
        for idx in range(row_count)
    ]


def _parse_holiday_rows_for_display(raw):
    if not raw:
        return []
    return [
        {"holiday_date": item.strip()}
        for item in raw.replace("\n", ";").split(";")
        if item.strip()
    ]


def _structured_amortizing_schedule_rows(form_data):
    existing = _parse_payment_schedule_for_display(form_data.get("cashflow_schedule", ""))
    if existing:
        return existing

    try:
        first_coupon = parse_date(form_data["first_coupon_date"])
        maturity = parse_date(form_data["maturity_date"])
        notional = _float_value(form_data, "notional")
        coupon_rate = _float_value(form_data, "coupon_rate")
        payments_per_year = _int_value(form_data, "payments_per_year")
        amortization_style = form_data.get("amortization_style", "straight_line")
    except (KeyError, TypeError, ValueError):
        return []

    months = 12 // payments_per_year
    payment_dates = []
    current = first_coupon
    while current < maturity:
        payment_dates.append(current)
        current = _add_months_preserve_day(current, months)
    if not payment_dates or payment_dates[-1] != maturity:
        payment_dates.append(maturity)

    outstanding = notional
    rows = []
    for idx, payment_date in enumerate(payment_dates, start=1):
        if amortization_style == "bullet":
            principal = outstanding if idx == len(payment_dates) else 0.0
        elif amortization_style == "straight_line":
            principal = notional / len(payment_dates)
            if idx == len(payment_dates):
                principal = outstanding
        else:
            principal = 0.0
        principal = min(max(principal, 0.0), max(outstanding, 0.0))
        rows.append(
            {
                "payment_date": payment_date.isoformat(),
                "opening_notional": _format_schedule_number(outstanding),
                "coupon_rate": _format_schedule_number(coupon_rate),
                "principal": _format_schedule_number(principal),
                "fixed_payment": "0",
            }
        )
        outstanding -= principal
    return rows


def _callable_amortizing_exercise_rows(form_data, payment_rows):
    existing = _parse_exercise_schedule_for_display(form_data.get("exercise_schedule", ""))
    if existing:
        return existing

    try:
        valuation_date = parse_date(form_data["valuation_date"])
        first_exercise_date = parse_date(form_data["first_exercise_date"])
        maturity_date = parse_date(form_data["maturity_date"])
        exercise_price_pct = _float_value(form_data, "exercise_price_pct")
        option_rights = form_data.get("option_rights", "callable")
        exercise_style = form_data.get("exercise_style", "bermudan")
    except (KeyError, TypeError, ValueError):
        return []

    rows = []
    previous_date = valuation_date
    for payment_row in payment_rows:
        payment_date = parse_date(payment_row["payment_date"])
        if payment_date < first_exercise_date or payment_date >= maturity_date:
            previous_date = payment_date
            continue
        start_date = payment_date if exercise_style == "bermudan" else max(previous_date, first_exercise_date)
        rows.append(
            {
                "start_date": start_date.isoformat(),
                "end_date": payment_date.isoformat(),
                "call_price_pct": _format_schedule_number(exercise_price_pct if option_rights in {"callable", "callable_putable"} else 0.0),
                "put_price_pct": _format_schedule_number(exercise_price_pct if option_rights in {"putable", "callable_putable"} else 0.0),
            }
        )
        previous_date = payment_date
    return rows


def _price_fixed_income_extension(product_slug, form_data):
    discount_curve = _discount_curve_from_form(form_data)
    forward_curve = parse_curve(
        form_data["forward_curve_tenors"],
        form_data["forward_curve_rates"],
    )

    if product_slug == "fra":
        return price_fra(
            FraTerms(
                valuation_date=parse_date(form_data["valuation_date"]),
                start_date=parse_date(form_data["start_date"]),
                end_date=parse_date(form_data["end_date"]),
                notional=_float_value(form_data, "notional"),
                strike_rate=_float_value(form_data, "strike_rate"),
                position=form_data["position"],
                day_count=form_data["day_count"],
                discount_curve=discount_curve,
                forward_curve=forward_curve,
                scenario_shock_bp=_float_value(form_data, "scenario_shock_bp"),
            )
        )

    if product_slug == "cap-floor":
        return price_cap_floor(
            CapFloorTerms(
                valuation_date=parse_date(form_data["valuation_date"]),
                start_date=parse_date(form_data["start_date"]),
                maturity_date=parse_date(form_data["maturity_date"]),
                notional=_float_value(form_data, "notional"),
                strike_rate=_float_value(form_data, "strike_rate"),
                option_type=form_data["option_type"],
                volatility=_float_value(form_data, "volatility"),
                payments_per_year=_int_value(form_data, "payments_per_year"),
                day_count=form_data["day_count"],
                discount_curve=discount_curve,
                forward_curve=forward_curve,
                scenario_rate_shock_bp=_float_value(form_data, "scenario_rate_shock_bp"),
                scenario_vol_shock=_float_value(form_data, "scenario_vol_shock"),
            )
        )

    if product_slug == "callable-putable-bond":
        return price_callable_putable_bond(
            CallableBondTerms(
                valuation_date=parse_date(form_data["valuation_date"]),
                maturity_date=parse_date(form_data["maturity_date"]),
                notional=_float_value(form_data, "notional"),
                coupon_rate=_float_value(form_data, "coupon_rate"),
                market_clean_price_pct=_float_value(form_data, "market_clean_price_pct"),
                payments_per_year=_int_value(form_data, "payments_per_year"),
                day_count=form_data["day_count"],
                discount_curve=discount_curve,
                option_type=form_data["option_type"],
                call_or_put_price=_float_value(form_data, "call_or_put_price"),
                first_exercise_year=_float_value(form_data, "first_exercise_year"),
                short_rate_volatility=_float_value(form_data, "short_rate_volatility"),
                scenario_shock_bp=_float_value(form_data, "scenario_shock_bp"),
            )
        )

    if product_slug == "callable-amortizing-bond":
        schedule_source = form_data.get("schedule_mode", "period_terms")
        generated_schedule = schedule_source in {"generated_bullet", "generated_straight_line"}
        amortization_style = {
            "generated_bullet": "bullet",
            "generated_straight_line": "straight_line",
        }.get(schedule_source, "sinking_schedule")
        return price_callable_amortizing_bond(
            CallableAmortizingBondTerms(
                schedule_mode=schedule_source if not generated_schedule else "explicit",
                effective_date=parse_date(form_data["effective_date"]),
                valuation_date=parse_date(form_data["valuation_date"]),
                dated_date=parse_date(form_data["dated_date"]) if form_data.get("dated_date") else None,
                first_coupon_date=parse_date(form_data["first_coupon_date"]) if form_data.get("first_coupon_date") else None,
                last_coupon_date=parse_date(form_data["last_coupon_date"]) if form_data.get("last_coupon_date") else None,
                maturity_date=parse_date(form_data["maturity_date"]),
                notional=_float_value(form_data, "notional"),
                coupon_rate=_float_value(form_data, "coupon_rate"),
                market_clean_price_pct=_float_value(form_data, "market_clean_price_pct"),
                payments_per_year=_int_value(form_data, "payments_per_year"),
                day_count=form_data["day_count"],
                discount_curve=discount_curve,
                option_rights=form_data["option_rights"],
                exercise_style=form_data["exercise_style"],
                first_exercise_date=parse_date(form_data["first_exercise_date"]),
                exercise_price_pct=_float_value(form_data, "exercise_price_pct"),
                short_rate_model=form_data.get("short_rate_model", "hull_white"),
                short_rate_volatility=_float_value(form_data, "short_rate_volatility_pct") / 100.0,
                short_rate_mean_reversion=_float_value(form_data, "short_rate_mean_reversion_pct") / 100.0,
                lattice_steps_per_period=_int_value(form_data, "lattice_steps_per_period"),
                scenario_shock_bp=_float_value(form_data, "scenario_shock_bp"),
                coupon_schedule=form_data.get("coupon_schedule", ""),
                principal_schedule=form_data.get("principal_schedule", ""),
                cashflow_schedule="" if generated_schedule else form_data.get("cashflow_schedule", ""),
                exercise_schedule=form_data.get("exercise_schedule", "") if form_data.get("exercise_schedule_source", "custom") == "custom" else "",
                amortization_style=amortization_style,
                fixed_payment_schedule=form_data.get("fixed_payment_schedule", ""),
                fixed_payment_treatment=form_data.get("fixed_payment_treatment", "additional_cashflow"),
                business_day_convention=form_data.get("business_day_convention", "none"),
                notification_days=_int_value(form_data, "notification_days"),
                interpolation_method=form_data.get("interpolation_method", "linear"),
                tree_generation=form_data.get("tree_generation", "maturity"),
                holiday_dates=form_data.get("holiday_dates", ""),
            )
        )

    if product_slug == "level-coupon-bond":
        return price_generic_bond(
            GenericBondTerms(
                valuation_date=parse_date(form_data["valuation_date"]),
                maturity_date=parse_date(form_data["maturity_date"]),
                notional=_float_value(form_data, "notional"),
                coupon_rate=_float_value(form_data, "coupon_rate"),
                payments_per_year=_int_value(form_data, "payments_per_year"),
                day_count=form_data["day_count"],
                discount_curve=discount_curve,
                market_clean_price_pct=_float_value(form_data, "market_clean_price_pct"),
                scenario_shock_bp=_float_value(form_data, "scenario_shock_bp"),
                amortization_style="bullet",
            )
        )

    if product_slug == "amortizing-stepup-sinking-bond":
        return price_generic_bond(
            GenericBondTerms(
                valuation_date=parse_date(form_data["valuation_date"]),
                maturity_date=parse_date(form_data["maturity_date"]),
                notional=_float_value(form_data, "notional"),
                coupon_rate=_float_value(form_data, "coupon_rate"),
                payments_per_year=_int_value(form_data, "payments_per_year"),
                day_count=form_data["day_count"],
                discount_curve=discount_curve,
                market_clean_price_pct=_float_value(form_data, "market_clean_price_pct"),
                scenario_shock_bp=_float_value(form_data, "scenario_shock_bp"),
                coupon_schedule=form_data.get("coupon_schedule", ""),
                principal_schedule=form_data.get("principal_schedule", ""),
                cashflow_schedule=form_data.get("cashflow_schedule", ""),
                amortization_style=form_data["amortization_style"],
                pricing_basis=form_data.get("pricing_basis", "curve"),
                yield_to_maturity=_float_value(form_data, "yield_to_maturity") if form_data.get("yield_to_maturity") else None,
                dated_date=parse_date(form_data["dated_date"]) if form_data.get("dated_date") else None,
                first_coupon_date=parse_date(form_data["first_coupon_date"]) if form_data.get("first_coupon_date") else None,
                last_coupon_date=parse_date(form_data["last_coupon_date"]) if form_data.get("last_coupon_date") else None,
            )
        )

    if product_slug == "custom-structured-bond":
        return price_generic_bond(
            GenericBondTerms(
                valuation_date=parse_date(form_data["valuation_date"]),
                maturity_date=parse_date(form_data["maturity_date"]),
                notional=_float_value(form_data, "notional"),
                coupon_rate=_float_value(form_data, "coupon_rate"),
                payments_per_year=_int_value(form_data, "payments_per_year"),
                day_count=form_data["day_count"],
                discount_curve=discount_curve,
                market_clean_price_pct=_float_value(form_data, "market_clean_price_pct"),
                scenario_shock_bp=_float_value(form_data, "scenario_shock_bp"),
                coupon_schedule=form_data.get("coupon_schedule", ""),
                principal_schedule=form_data.get("principal_schedule", ""),
                fixed_payment_schedule=form_data.get("fixed_payment_schedule", ""),
                amortization_style="sinking_schedule",
            )
        )

    if product_slug == "bond-series":
        return price_bond_series(
            BondSeriesTerms(
                valuation_date=parse_date(form_data["valuation_date"]),
                series_table=form_data["series_table"],
                day_count=form_data["day_count"],
                discount_curve=discount_curve,
                scenario_shock_bp=_float_value(form_data, "scenario_shock_bp"),
            )
        )

    if product_slug == "loan-lease-annuity":
        return price_loan_lease_annuity(
            LoanLeaseAnnuityTerms(
                valuation_date=parse_date(form_data["valuation_date"]),
                start_date=parse_date(form_data["start_date"]),
                maturity_date=parse_date(form_data["maturity_date"]),
                principal=_float_value(form_data, "principal"),
                contract_rate=_float_value(form_data, "contract_rate"),
                payments_per_year=_int_value(form_data, "payments_per_year"),
                day_count=form_data["day_count"],
                structure_type=form_data["structure_type"],
                discount_curve=discount_curve,
                scenario_shock_bp=_float_value(form_data, "scenario_shock_bp"),
            )
        )

    if product_slug == "asset-swap":
        return price_asset_swap(
            AssetSwapTerms(
                valuation_date=parse_date(form_data["valuation_date"]),
                maturity_date=parse_date(form_data["maturity_date"]),
                notional=_float_value(form_data, "notional"),
                coupon_rate=_float_value(form_data, "coupon_rate"),
                market_clean_price_pct=_float_value(form_data, "market_clean_price_pct"),
                quoted_asset_swap_spread_bp=_float_value(form_data, "quoted_asset_swap_spread_bp"),
                payments_per_year=_int_value(form_data, "payments_per_year"),
                day_count=form_data["day_count"],
                discount_curve=discount_curve,
                scenario_rate_shock_bp=_float_value(form_data, "scenario_rate_shock_bp"),
                scenario_spread_shock_bp=_float_value(form_data, "scenario_spread_shock_bp"),
            )
        )

    if product_slug == "inflation-linked-bond":
        return price_inflation_linked_bond(
            InflationLinkedBondTerms(
                valuation_date=parse_date(form_data["valuation_date"]),
                maturity_date=parse_date(form_data["maturity_date"]),
                notional=_float_value(form_data, "notional"),
                real_coupon_rate=_float_value(form_data, "real_coupon_rate"),
                base_cpi=_float_value(form_data, "base_cpi"),
                current_cpi=_float_value(form_data, "current_cpi"),
                annual_inflation_rate=_float_value(form_data, "annual_inflation_rate"),
                indexation_lag_months=_int_value(form_data, "indexation_lag_months"),
                principal_floor=form_data["principal_floor"],
                payments_per_year=_int_value(form_data, "payments_per_year"),
                day_count=form_data["day_count"],
                real_discount_curve=discount_curve,
                nominal_curve=forward_curve,
                scenario_real_rate_shock_bp=_float_value(form_data, "scenario_real_rate_shock_bp"),
                scenario_inflation_shock_bp=_float_value(form_data, "scenario_inflation_shock_bp"),
            )
        )

    if product_slug == "bond-forward-treasury-lock":
        return price_bond_forward_treasury_lock(
            BondForwardTreasuryLockTerms(
                valuation_date=parse_date(form_data["valuation_date"]),
                delivery_date=parse_date(form_data["delivery_date"]),
                bond_maturity_date=parse_date(form_data["bond_maturity_date"]),
                notional=_float_value(form_data, "notional"),
                coupon_rate=_float_value(form_data, "coupon_rate"),
                spot_dirty_price_pct=_float_value(form_data, "spot_dirty_price_pct"),
                financing_rate=_float_value(form_data, "financing_rate"),
                locked_forward_yield=_float_value(form_data, "locked_forward_yield"),
                modified_duration=_float_value(form_data, "modified_duration"),
                position=form_data["position"],
                payments_per_year=_int_value(form_data, "payments_per_year"),
                day_count=form_data["day_count"],
                discount_curve=discount_curve,
                scenario_rate_shock_bp=_float_value(form_data, "scenario_rate_shock_bp"),
            )
        )

    raise ValueError("Unsupported fixed-income extension product.")


# Route to initialize bond classes with common parameters
@nc_bonds_bp.route("/", methods=["GET", "POST"])
def nc_bonds():
    return render_template("nc_bonds.html")


@nc_bonds_bp.route("/fixed_bonds", methods=["GET", "POST"])
def nc_fixed_bonds():

    readme_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "nc_fixed_bonds.md"
    )
    with open(readme_path, "r") as readme_file:
        content = readme_file.read()
    md_content = markdown.markdown(content)

    gpt_assessment = None
    fr_bond_results = None
    form_data = {}
    validation_errors = []

    if request.method == "POST":
        action = request.form.get("analysis_type")
        # Retrieve form data from URL parameters
        form_data = {
            "value_date": request.form["value_date"],
            "spot_dates": request.form["spot_dates"],
            "spot_rates": request.form["spot_rates"],
            "shocks": request.form["shocks"],
            "day_count_val": request.form["day_count"],
            "calendar_val": request.form["calendar"],
            "interpolation_val": request.form["interpolation"],
            "compounding_val": request.form["compounding"],
            "compounding_frequency_val": request.form["compounding_frequency"],
            "issue_date": request.form["issue_date"],
            "maturity_date": request.form["maturity_date"],
            "tenor_val": request.form["tenor"],
            "coupon_rate": request.form["coupon_rate"],
            "notional": request.form["notional"],
        }

        value_date = form_data["value_date"]
        spot_dates = form_data["spot_dates"]
        spot_rates = form_data["spot_rates"]
        shocks = form_data["shocks"]
        issue_date = form_data["issue_date"]
        maturity_date = form_data["maturity_date"]
        coupon_rate = form_data["coupon_rate"]
        notional = form_data["notional"]

        # A negative or zero notional silently produces meaningless prices, so
        # tell the user to correct it rather than handing it to QuantLib.
        validation_errors = check_positive(
            {"notional": notional, "coupon_rate": coupon_rate},
            {"notional": "Notional", "coupon_rate": "Coupon rate"},
            allow_zero=["coupon_rate"],
        )
        if validation_errors:
            return render_template(
                "ncfixedbonds.html",
                form_data=form_data,
                fr_bond_results=None,
                md_content=md_content,
                gpt_assessment=None,
                validation_errors=validation_errors,
            )

        calendar_val = form_data["calendar_val"]
        interpolation_val = form_data["interpolation_val"]
        compounding_val = form_data["compounding_val"]
        compounding_frequency_val = form_data["compounding_frequency_val"]
        tenor_val = form_data["tenor_val"]
        day_count_val = form_data["day_count_val"]

        # Map the calendar value to a QuantLib Calendar
        if calendar_val == "UnitedStates":
            calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
        elif calendar_val == "TARGET":
            calendar = ql.TARGET()
        elif calendar_val == "UnitedKingdom":
            calendar = ql.UnitedKingdom()
        elif calendar_val == "China":
            calendar = ql.China()

        # Map interpolation value to a QuantLib interpolation type
        if interpolation_val == "Linear":
            interpolation = ql.Linear()
        elif interpolation_val == "LogLinear":
            interpolation = ql.LogLinear()
        elif interpolation_val == "Cubic":
            interpolation = ql.Cubic()

        # Map compounding value to a QuantLib Compounding type
        if compounding_val == "Compounded":
            compounding = ql.Compounded
        elif compounding_val == "Simple":
            compounding = ql.Simple
        elif compounding_val == "Continuous":
            compounding = ql.Continuous

        # Map compounding frequency value to QuantLib Frequency
        if compounding_frequency_val == "Annual":
            compounding_frequency = ql.Annual
        elif compounding_frequency_val == "Semiannual":
            compounding_frequency = ql.Semiannual
        elif compounding_frequency_val == "Quarterly":
            compounding_frequency = ql.Quarterly
        elif compounding_frequency_val == "Monthly":
            compounding_frequency = ql.Monthly
        elif compounding_frequency_val == "Daily":
            compounding_frequency = ql.Daily

        # Map tenor value to QuantLib Tenor
        if tenor_val == "Annual":
            tenor = ql.Period(ql.Annual)
        elif tenor_val == "Semiannual":
            tenor = ql.Period(ql.Semiannual)
        elif tenor_val == "Quarterly":
            tenor = ql.Period(ql.Quarterly)
        elif tenor_val == "Monthly":
            tenor = ql.Period(ql.Monthly)

        # Map the day count value to a QuantLib DayCount
        if day_count_val == "ActualActual":
            day_count = ql.ActualActual(ql.ActualActual.Bond)
        elif day_count_val == "Thirty360":
            day_count = ql.Thirty360(ql.Thirty360.BondBasis)
        elif day_count_val == "Actual360":
            day_count = ql.Actual360()
        elif day_count_val == "Actual365Fixed":
            day_count = ql.Actual365Fixed()

        fixed_bond = NCFixedBonds(
            value_date,
            spot_dates,
            spot_rates,
            shocks,
            day_count,
            calendar,
            interpolation,
            compounding,
            compounding_frequency,
        )

        fr_bond_results = fixed_bond.fixed_rate(
            issue_date, maturity_date, tenor, coupon_rate, notional
        )

        if current_user.is_authenticated:
            baseline = fr_bond_results.get(0, {})
            instrument = Instrument(
                user_id=current_user.id,
                product_type="fixed_rate_bond",
                ticker=None,
                model_name="NCFixedBonds",
                start_date=str(issue_date),
                end_date=str(maturity_date),
                params_json={
                    "coupon_rate": coupon_rate,
                    "notional": notional,
                    "tenor": tenor_val,
                    "day_count": day_count_val,
                    "compounding": compounding_val,
                    "calendar": calendar_val,
                },
            )
            db.session.add(instrument)
            db.session.flush()

            pricing_result = PricingResult(
                user_id=current_user.id,
                instrument_id=instrument.id,
                price=baseline.get("Price"),
                delta=None,
                gamma=None,
                vega=None,
                theta=None,
                rho=None,
                result_json={
                    "npv": baseline.get("NPV"),
                    "price": baseline.get("Price"),
                    "ytm": baseline.get("YTM"),
                    "duration": baseline.get("Duration"),
                    "convexity": baseline.get("Convexity"),
                },
            )
            db.session.add(pricing_result)
            db.session.commit()

        if action == "ai_assessment":
            # AI Assessment logic
            if fr_bond_results:
                logger.debug("Fixed bond results available for AI assessment.")
                # Prepare the input for AI using the actual bond results
                assessment_input = f"Please assess the bond pricing results based on the following outputs: {fr_bond_results}. Focus on the price changes across different shocks and any notable patterns."
                gpt_assessment = ask_gpt(assessment_input)
            else:
                gpt_assessment = "No bond pricing results available for assessment."
    return render_template(
        "ncfixedbonds.html",
        form_data=form_data,
        fr_bond_results=fr_bond_results,
        md_content=md_content,
        gpt_assessment=gpt_assessment,
        validation_errors=validation_errors,
    )


@nc_bonds_bp.route("/fixed_amort_bonds", methods=["GET", "POST"])
def nc_fixed_amort_bonds():
    readme_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "nc_fixed_amort_bonds.md"
    )
    with open(readme_path, "r") as readme_file:
        content = readme_file.read()
    md_content = markdown.markdown(content)

    fram_bond_results = None
    gpt_assessment = None
    form_data = {}

    if request.method == "POST":
        action = request.form.get("analysis_type")
        # Retrieve form data from URL parameters
        form_data = {
            "value_date": request.form["value_date"],
            "spot_dates": request.form["spot_dates"],
            "spot_rates": request.form["spot_rates"],
            "shocks": request.form["shocks"],
            "day_count_val": request.form["day_count"],
            "calendar_val": request.form["calendar"],
            "interpolation_val": request.form["interpolation"],
            "compounding_val": request.form["compounding"],
            "compounding_frequency_val": request.form["compounding_frequency"],
            "issue_date": request.form["issue_date"],
            "maturity_date": request.form["maturity_date"],
            "tenor_val": request.form["tenor"],
            "coupon_rate": request.form["coupon_rate"],
            "notional": request.form["notional"],
        }

        value_date = form_data["value_date"]
        spot_dates = form_data["spot_dates"]
        spot_rates = form_data["spot_rates"]
        shocks = form_data["shocks"]
        issue_date = form_data["issue_date"]
        maturity_date = form_data["maturity_date"]
        coupon_rate = form_data["coupon_rate"]
        notional = form_data["notional"]

        calendar_val = form_data["calendar_val"]
        interpolation_val = form_data["interpolation_val"]
        compounding_val = form_data["compounding_val"]
        compounding_frequency_val = form_data["compounding_frequency_val"]
        tenor_val = form_data["tenor_val"]
        day_count_val = form_data["day_count_val"]

        # Map the calendar value to a QuantLib Calendar
        if calendar_val == "UnitedStates":
            calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
        elif calendar_val == "TARGET":
            calendar = ql.TARGET()
        elif calendar_val == "UnitedKingdom":
            calendar = ql.UnitedKingdom()
        elif calendar_val == "China":
            calendar = ql.China()

        # Map interpolation value to a QuantLib interpolation type
        if interpolation_val == "Linear":
            interpolation = ql.Linear()
        elif interpolation_val == "LogLinear":
            interpolation = ql.LogLinear()
        elif interpolation_val == "Cubic":
            interpolation = ql.Cubic()

        # Map compounding value to a QuantLib Compounding type
        if compounding_val == "Compounded":
            compounding = ql.Compounded
        elif compounding_val == "Simple":
            compounding = ql.Simple
        elif compounding_val == "Continuous":
            compounding = ql.Continuous

        # Map compounding frequency value to QuantLib Frequency
        if compounding_frequency_val == "Annual":
            compounding_frequency = ql.Annual
        elif compounding_frequency_val == "Semiannual":
            compounding_frequency = ql.Semiannual
        elif compounding_frequency_val == "Quarterly":
            compounding_frequency = ql.Quarterly
        elif compounding_frequency_val == "Monthly":
            compounding_frequency = ql.Monthly
        elif compounding_frequency_val == "Daily":
            compounding_frequency = ql.Daily

        # Map tenor value to QuantLib Tenor
        if tenor_val == "Annual":
            tenor = ql.Period(ql.Annual)
        elif tenor_val == "Semiannual":
            tenor = ql.Period(ql.Semiannual)
        elif tenor_val == "Quarterly":
            tenor = ql.Period(ql.Quarterly)
        elif tenor_val == "Monthly":
            tenor = ql.Period(ql.Monthly)

        # Map the day count value to a QuantLib DayCount
        if day_count_val == "ActualActual":
            day_count = ql.ActualActual(ql.ActualActual.Bond)
        elif day_count_val == "Thirty360":
            day_count = ql.Thirty360(ql.Thirty360.BondBasis)
        elif day_count_val == "Actual360":
            day_count = ql.Actual360()
        elif day_count_val == "Actual365Fixed":
            day_count = ql.Actual365Fixed()

        fixed_bond = NCFixedBonds(
            value_date,
            spot_dates,
            spot_rates,
            shocks,
            day_count,
            calendar,
            interpolation,
            compounding,
            compounding_frequency,
        )

        fram_bond_results = fixed_bond.fixed_rate_amortizing(
            issue_date, maturity_date, tenor, coupon_rate, notional
        )
        if action == "ai_assessment":
            # AI Assessment logic
            if fram_bond_results:
                # Prepare the input for AI using the actual bond results
                assessment_input = f"Please assess the amortizing bond pricing results based on the following outputs: {fram_bond_results}. Focus on the price changes across different shocks and any notable patterns."
                gpt_assessment = ask_gpt(assessment_input)
            else:
                gpt_assessment = "No bond pricing results available for assessment."
    return render_template(
        "ncfixedamortbonds.html",
        form_data=form_data,
        fram_bond_results=fram_bond_results,
        md_content=md_content,
        gpt_assessment=gpt_assessment,
    )


@nc_bonds_bp.route("/floating_bonds", methods=["GET", "POST"])
def nc_floating_bonds():

    readme_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "nc_floating_bonds.md"
    )
    with open(readme_path, "r") as readme_file:
        content = readme_file.read()
    md_content = markdown.markdown(content)

    fl_bond_results = None
    gpt_assessment = None
    form_data = {}

    if request.method == "POST":
        action = request.form.get("analysis_type")
        # Retrieve form data from URL parameters
        form_data = {
            "value_date": request.form["value_date"],
            "spot_dates": request.form["spot_dates"],
            "spot_rates": request.form["spot_rates"],
            "index_dates": request.form["index_dates"],
            "index_rates": request.form["index_rates"],
            "calendar_val": request.form["calendar"],
            "currency_val": request.form["currency"],
            "interpolation_val": request.form["interpolation"],
            "compounding_val": request.form["compounding"],
            "compounding_frequency_val": request.form["compounding_frequency"],
            "shocks": request.form["shocks"],
            "issue_date": request.form["issue_date"],
            "maturity_date": request.form["maturity_date"],
            "tenor_val": request.form["tenor"],
            "spread": request.form["spread"],
            "notional": request.form["notional"],
            "day_count_val": request.form["day_count"],
        }

        value_date = form_data["value_date"]
        spot_dates = form_data["spot_dates"]
        spot_rates = form_data["spot_rates"]
        index_dates = form_data["index_dates"]
        index_rates = form_data["index_rates"]
        shocks = form_data["shocks"]
        issue_date = form_data["issue_date"]
        maturity_date = form_data["maturity_date"]
        spread = form_data["spread"]
        notional = form_data["notional"]

        calendar_val = form_data["calendar_val"]
        currency_val = form_data["currency_val"]
        interpolation_val = form_data["interpolation_val"]
        compounding_val = form_data["compounding_val"]
        compounding_frequency_val = form_data["compounding_frequency_val"]
        tenor_val = form_data["tenor_val"]
        day_count_val = form_data["day_count_val"]

        # Map the calendar value to a QuantLib Calendar
        if calendar_val == "UnitedStates":
            calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
        elif calendar_val == "TARGET":
            calendar = ql.TARGET()
        elif calendar_val == "China":
            calendar = ql.China()

        # Map the currency value to a QuantLib Currency
        if currency_val == "USD":
            currency = ql.USDCurrency()
        elif currency_val == "EUR":
            currency = ql.EURCurrency()
        elif currency_val == "CNY":
            currency = ql.CNYCurrency()
        elif currency_val == "GBP":
            currency = ql.GBPCurrency()
        elif currency_val == "JPY":
            currency = ql.JPYCurrency()

        # Map interpolation value to a QuantLib interpolation type
        if interpolation_val == "Linear":
            interpolation = ql.Linear()
        elif interpolation_val == "LogLinear":
            interpolation = ql.LogLinear()
        elif interpolation_val == "Cubic":
            interpolation = ql.Cubic()

        # Map tenor value to QuantLib Period
        if tenor_val == "Annual":
            tenor = ql.Period(ql.Annual)
        elif tenor_val == "Semiannual":
            tenor = ql.Period(ql.Semiannual)
        elif tenor_val == "Quarterly":
            tenor = ql.Period(ql.Quarterly)
        elif tenor_val == "Monthly":
            tenor = ql.Period(ql.Monthly)
        else:
            tenor = ql.Period(ql.Semiannual)  # Default to Semiannual

        # Map compounding value to QuantLib Compounding
        if compounding_val == "Compounded":
            compounding = ql.Compounded
        elif compounding_val == "Continuous":
            compounding = ql.Continuous
        elif compounding_val == "Simple":
            compounding = ql.Simple
        else:
            compounding = ql.Compounded  # Default to Compounded

        # Map compounding frequency value to QuantLib Frequency
        if compounding_frequency_val == "Annual":
            compounding_frequency = ql.Annual
        elif compounding_frequency_val == "Semiannual":
            compounding_frequency = ql.Semiannual
        elif compounding_frequency_val == "Quarterly":
            compounding_frequency = ql.Quarterly
        elif compounding_frequency_val == "Monthly":
            compounding_frequency = ql.Monthly

        # Map tenor value to QuantLib Tenor
        if tenor_val == "Annual":
            tenor = ql.Period(ql.Annual)
        elif tenor_val == "Semiannual":
            tenor = ql.Period(ql.Semiannual)
        elif tenor_val == "Quarterly":
            tenor = ql.Period(ql.Quarterly)
        elif tenor_val == "Monthly":
            tenor = ql.Period(ql.Monthly)

        # Map the day count value to a QuantLib DayCount
        if day_count_val == "ActualActual":
            day_count = ql.ActualActual(ql.ActualActual.Bond)
        elif day_count_val == "Thirty360":
            day_count = ql.Thirty360(ql.Thirty360.BondBasis)
        elif day_count_val == "Actual360":
            day_count = ql.Actual360()
        elif day_count_val == "Actual365Fixed":
            day_count = ql.Actual365Fixed()

        floating_bond = NCFloatingBonds(
            value_date,
            spot_dates,
            spot_rates,
            index_dates,
            index_rates,
            calendar,
            currency,
            interpolation,
            compounding,
            compounding_frequency,
            epsilon=0.001,
        )

        fl_bond_results = floating_bond.price_floating(
            shocks, issue_date, maturity_date, tenor, spread, notional, day_count
        )
        if action == "ai_assessment":
            # AI Assessment logic
            if fl_bond_results:
                # Prepare the input for AI using the actual bond results
                assessment_input = f"Please assess the floating rate bond pricing results based on the following outputs: {fl_bond_results}. Focus on the price changes across different shocks and any notable patterns."
                gpt_assessment = ask_gpt(assessment_input)
            else:
                gpt_assessment = "No bond pricing results available for assessment."

    return render_template(
        "ncfloatingbonds.html",
        form_data=form_data,
        fl_bond_results=fl_bond_results,
        md_content=md_content,
        gpt_assessment=gpt_assessment,
    )


@nc_bonds_bp.route("/floating_amortizing_bonds", methods=["GET", "POST"])
def nc_floating_amort_bonds():

    readme_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "nc_floating_amort_bonds.md"
    )
    with open(readme_path, "r") as readme_file:
        content = readme_file.read()
    md_content = markdown.markdown(content)

    flam_bond_results = None
    gpt_assessment = None
    form_data = {}
    validation_errors = []

    if request.method == "POST":
        action = request.form.get("analysis_type")
        # Retrieve form data from URL parameters
        form_data = {
            "value_date": request.form["value_date"],
            "spot_dates": request.form["spot_dates"],
            "spot_rates": request.form["spot_rates"],
            "index_dates": request.form["index_dates"],
            "index_rates": request.form["index_rates"],
            "calendar_val": request.form["calendar"],
            "currency_val": request.form["currency"],
            "interpolation_val": request.form["interpolation"],
            "compounding_val": request.form["compounding"],
            "compounding_frequency_val": request.form["compounding_frequency"],
            "shocks": request.form["shocks"],
            "issue_date": request.form["issue_date"],
            "maturity_date": request.form["maturity_date"],
            "tenor_val": request.form["tenor"],
            "spread": request.form["spread"],
            "notional": request.form["notional"],
            "notional_dates": request.form["notional_dates"],
            "day_count_val": request.form["day_count"],
        }

        # The amortization schedule is free text, so check it before QuantLib
        # turns a malformed entry into an opaque overload error.
        try:
            NCFloatingBonds._parse_amortization_schedule(
                form_data["notional"], form_data["notional_dates"]
            )
        except ValueError as exc:
            validation_errors.append(str(exc))

        if validation_errors:
            return render_template(
                "ncfloatingamortbonds.html",
                form_data=form_data,
                flam_bond_results=None,
                md_content=md_content,
                gpt_assessment=None,
                validation_errors=validation_errors,
            )

        value_date = form_data["value_date"]
        spot_dates = form_data["spot_dates"]
        spot_rates = form_data["spot_rates"]
        index_dates = form_data["index_dates"]
        index_rates = form_data["index_rates"]
        shocks = form_data["shocks"]
        issue_date = form_data["issue_date"]
        maturity_date = form_data["maturity_date"]
        spread = form_data["spread"]
        notional = form_data["notional"]
        notional_dates = form_data["notional_dates"]

        calendar_val = form_data["calendar_val"]
        currency_val = form_data["currency_val"]
        interpolation_val = form_data["interpolation_val"]
        compounding_val = form_data["compounding_val"]
        compounding_frequency_val = form_data["compounding_frequency_val"]
        tenor_val = form_data["tenor_val"]
        day_count_val = form_data["day_count_val"]

        # Map the calendar value to a QuantLib Calendar
        if calendar_val == "UnitedStates":
            calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
        elif calendar_val == "TARGET":
            calendar = ql.TARGET()
        elif calendar_val == "China":
            calendar = ql.China()

        # Map the currency value to a QuantLib Currency
        if currency_val == "USD":
            currency = ql.USDCurrency()
        elif currency_val == "EUR":
            currency = ql.EURCurrency()
        elif currency_val == "CNY":
            currency = ql.CNYCurrency()
        elif currency_val == "GBP":
            currency = ql.GBPCurrency()
        elif currency_val == "JPY":
            currency = ql.JPYCurrency()

        # Map interpolation value to QuantLib Interpolation
        if interpolation_val == "Linear":
            interpolation = ql.Linear()
        elif interpolation_val == "LogLinear":
            interpolation = ql.LogLinear()
        elif interpolation_val == "Cubic":
            interpolation = ql.Cubic()
        else:
            interpolation = ql.Linear()  # Default to Linear

        # Map compounding value to QuantLib Compounding
        if compounding_val == "Compounded":
            compounding = ql.Compounded
        elif compounding_val == "Continuous":
            compounding = ql.Continuous
        elif compounding_val == "Simple":
            compounding = ql.Simple
        else:
            compounding = ql.Compounded  # Default to Compounded

        # Map compounding frequency value to QuantLib Frequency
        if compounding_frequency_val == "Annual":
            compounding_frequency = ql.Annual
        elif compounding_frequency_val == "Semiannual":
            compounding_frequency = ql.Semiannual
        elif compounding_frequency_val == "Quarterly":
            compounding_frequency = ql.Quarterly
        elif compounding_frequency_val == "Monthly":
            compounding_frequency = ql.Monthly
        else:
            compounding_frequency = ql.Annual  # Default to Annual

        # Map tenor value to QuantLib Period
        if tenor_val == "Annual":
            tenor = ql.Period(ql.Annual)
        elif tenor_val == "Semiannual":
            tenor = ql.Period(ql.Semiannual)
        elif tenor_val == "Quarterly":
            tenor = ql.Period(ql.Quarterly)
        elif tenor_val == "Monthly":
            tenor = ql.Period(ql.Monthly)
        else:
            tenor = ql.Period(ql.Semiannual)  # Default to Semiannual

        # Map the day count value to a QuantLib DayCount
        if day_count_val == "ActualActual":
            day_count = ql.ActualActual(ql.ActualActual.Bond)
        elif day_count_val == "Thirty360":
            day_count = ql.Thirty360(ql.Thirty360.BondBasis)
        elif day_count_val == "Actual360":
            day_count = ql.Actual360()
        elif day_count_val == "Actual365Fixed":
            day_count = ql.Actual365Fixed()
        else:
            day_count = ql.Actual360()  # Default to Actual/360

        # Initialize floating bond object
        floating_bond = NCFloatingBonds(
            value_date,
            spot_dates,
            spot_rates,
            index_dates,
            index_rates,
            calendar,
            currency,
            interpolation,
            compounding,
            compounding_frequency,
        )

        try:
            flam_bond_results = floating_bond.price_amortizing_floating(
                shocks,
                issue_date,
                maturity_date,
                tenor,
                spread,
                notional,
                notional_dates,
                day_count,
            )
        except Exception as exc:
            logger.exception("Floating amortizing bond pricing failed")
            return render_template(
                "ncfloatingamortbonds.html",
                form_data=form_data,
                flam_bond_results=None,
                md_content=md_content,
                gpt_assessment=None,
                validation_errors=[f"Pricing could not be completed: {exc}"],
            )

        if action == "ai_assessment":
            # AI Assessment logic
            if flam_bond_results:
                # Prepare the input for AI using the actual bond results
                assessment_input = f"Please assess the floating rate amortizing bond pricing results based on the following outputs: {flam_bond_results}. Focus on the price changes across different shocks and any notable patterns."
                gpt_assessment = ask_gpt(assessment_input)
            else:
                gpt_assessment = "No bond pricing results available for assessment."

    return render_template(
        "ncfloatingamortbonds.html",
        form_data=form_data,
        flam_bond_results=flam_bond_results,
        md_content=md_content,
        gpt_assessment=gpt_assessment,
        validation_errors=validation_errors,
    )


@nc_bonds_bp.route("/rates-fixed-income", methods=["GET"])
def rates_fixed_income_home():
    return render_template(
        "fixed_income_extensions_home.html",
        products=FIXED_INCOME_EXTENSION_CONFIGS,
    )


@nc_bonds_bp.route("/rates-fixed-income/<product_slug>", methods=["GET", "POST"])
def rates_fixed_income_product(product_slug):
    config = FIXED_INCOME_EXTENSION_CONFIGS.get(product_slug)
    if not config:
        abort(404)

    form_data = _extension_form_data(config)
    results = None
    pricing_error = None
    restored_run = False
    if (
        product_slug in {"amortizing-stepup-sinking-bond", "callable-amortizing-bond"}
        and request.method == "GET"
        and request.args.get("restore") == "latest"
        and current_user.is_authenticated
    ):
        latest_result = get_latest_pricing_result_for_user(
            f"fixed_income_{product_slug}",
            current_user.id,
        )
        if latest_result and latest_result.instrument:
            form_data.update(latest_result.instrument.params_json or {})
            results = latest_result.result_json or None
            restored_run = results is not None

    benchmark_presets = []
    selected_preset = ""
    selected_preset_label = "Custom scenario"
    if product_slug == "callable-amortizing-bond":
        benchmark_presets = _callable_amortizing_benchmark_presets(config)
        preset_ids = {preset["id"] for preset in benchmark_presets}
        requested_preset = request.form.get(
            "benchmark_preset",
            form_data.get("benchmark_preset", "callable_benchmark"),
        )
        selected_preset = requested_preset if requested_preset in preset_ids else "custom"
        if selected_preset != "custom":
            selected_preset_label = next(
                preset["label"] for preset in benchmark_presets if preset["id"] == selected_preset
            )
        form_data["benchmark_preset"] = selected_preset
    if request.method == "POST":
        try:
            results = _price_fixed_income_extension(product_slug, form_data)

            if current_user.is_authenticated:
                raw_price = None
                scenarios = results.get("scenarios") if isinstance(results, dict) else None
                if scenarios and isinstance(scenarios[0], dict):
                    raw_price = scenarios[0].get("pv") or scenarios[0].get("price")

                omitted_params = {
                    "discount_curve_tenors",
                    "discount_curve_rates",
                    "forward_curve_tenors",
                    "forward_curve_rates",
                }
                if product_slug in {"amortizing-stepup-sinking-bond", "callable-amortizing-bond"}:
                    omitted_params -= {"discount_curve_tenors", "discount_curve_rates"}

                instrument = Instrument(
                    user_id=current_user.id,
                    product_type=f"fixed_income_{product_slug}",
                    ticker=None,
                    model_name=config["title"],
                    start_date=form_data.get("valuation_date") or form_data.get("start_date"),
                    end_date=form_data.get("maturity_date") or form_data.get("end_date"),
                    params_json={
                        key: value
                        for key, value in form_data.items()
                        if key not in omitted_params
                    },
                )
                db.session.add(instrument)
                db.session.flush()

                pricing_result = PricingResult(
                    user_id=current_user.id,
                    instrument_id=instrument.id,
                    price=raw_price,
                    delta=None,
                    gamma=None,
                    vega=None,
                    theta=None,
                    rho=None,
                    result_json=results,
                )
                db.session.add(pricing_result)
                db.session.commit()
        except Exception as exc:
            logger.exception("Fixed income extension pricing failed for %s", product_slug)
            pricing_error = str(exc)

    if "field_sections" in config:
        field_sections = deepcopy(config["field_sections"])
    else:
        field_sections = [{"title": "Trade Terms", "fields": deepcopy(config["fields"])}]

    if product_slug == "callable-amortizing-bond":
        curve_fields = [
            {
                "name": "discount_curve_input_type",
                "label": "Discount Curve Input Type",
                "type": "select",
                "value": form_data["discount_curve_input_type"],
                "options": [("zero_rates", "Zero Rates"), ("discount_factors", "Dated Discount Factors")],
            },
            {
                "name": "interpolation_method",
                "label": "Curve Interpolation",
                "type": "select",
                "value": form_data["interpolation_method"],
                "options": [
                    ("linear_zero", "Linear Zero / Spot Rate"),
                    ("exponential", "Exponential / Log Discount Factor"),
                    ("linear_discount", "Linear Discount Factor"),
                    ("cubic_spline", "Natural Cubic Spline"),
                ],
                "hint": "Benchmark input: linear interpolation applied to dated discount factors.",
            },
            {
                "name": "discount_factor_curve",
                "label": "Discount Factor Curve",
                "type": "hidden",
                "value": form_data["discount_factor_curve"],
                "hint": "Rows use YYYY-MM-DD|discount_factor separated by semicolons or new lines.",
            },
            {
                "name": "discount_curve_tenors",
                "label": "Zero Curve Tenors (Years)",
                "type": "hidden",
                "value": form_data["discount_curve_tenors"],
                "hint": "Used when input type is Zero Rates.",
            },
            {
                "name": "discount_curve_rates",
                "label": "Zero Curve Rates",
                "type": "hidden",
                "value": form_data["discount_curve_rates"],
                "hint": "Used when input type is Zero Rates.",
            },
        ]
    elif product_slug == "amortizing-stepup-sinking-bond":
        curve_fields = [
            {
                "name": "discount_curve_tenors",
                "label": "Discount Curve Tenors (Years)",
                "type": "hidden",
                "value": form_data["discount_curve_tenors"],
                "hint": "Enter tenor and continuously compounded zero-rate pairs in the curve table.",
            },
            {
                "name": "discount_curve_rates",
                "label": "Discount Curve Zero Rates",
                "type": "hidden",
                "value": form_data["discount_curve_rates"],
                "hint": "Used when Pricing Basis is Price from Curve.",
            },
        ]
    else:
        curve_fields = [
            {
                "name": "discount_curve_tenors",
                "label": "Discount Curve Tenors (Years)",
                "type": "text",
                "value": form_data["discount_curve_tenors"],
                "hint": "Comma-separated year tenors.",
            },
            {
                "name": "discount_curve_rates",
                "label": "Discount Curve Zero Rates",
                "type": "text",
                "value": form_data["discount_curve_rates"],
                "hint": "Comma-separated continuously compounded zero rates.",
            },
        ]
    if product_slug in {"fra", "cap-floor", "inflation-linked-bond"}:
        curve_fields.extend(
            [
                {
                    "name": "forward_curve_tenors",
                    "label": "Forward Curve Tenors (Years)",
                    "type": "text",
                    "value": form_data["forward_curve_tenors"],
                    "hint": "Used for FRA and cap/floor projected rates.",
                },
                {
                    "name": "forward_curve_rates",
                    "label": "Forward Curve Zero Rates",
                    "type": "text",
                    "value": form_data["forward_curve_rates"],
                    "hint": "Used for FRA and cap/floor projected rates.",
                },
            ]
        )

    field_sections.append(
        {
            "title": "Curve Assumptions",
            "description": "Review zero-rate or discount-factor curve assumptions used for projection, discounting, and scenario analysis.",
            "icon": "C",
            "fields": curve_fields,
        }
    )

    for section in field_sections:
        for field in section["fields"]:
            field["value"] = form_data.get(field["name"], field.get("value", ""))
            field.setdefault("hint", "")
            field.setdefault("step", "any")

    schedule_rows = []
    exercise_rows = []
    discount_factor_rows = []
    zero_curve_rows = []
    holiday_rows = []
    if product_slug in {"amortizing-stepup-sinking-bond", "callable-amortizing-bond"}:
        schedule_rows = _structured_amortizing_schedule_rows(form_data)
    if product_slug == "callable-amortizing-bond":
        exercise_rows = _callable_amortizing_exercise_rows(form_data, schedule_rows)
        discount_factor_rows = _parse_discount_factor_rows_for_display(
            form_data.get("discount_factor_curve", "")
        )
        zero_curve_rows = _parse_zero_curve_rows_for_display(
            form_data.get("discount_curve_tenors", ""),
            form_data.get("discount_curve_rates", ""),
        )
        holiday_rows = _parse_holiday_rows_for_display(form_data.get("holiday_dates", ""))
    elif product_slug == "amortizing-stepup-sinking-bond":
        zero_curve_rows = _parse_zero_curve_rows_for_display(
            form_data.get("discount_curve_tenors", ""),
            form_data.get("discount_curve_rates", ""),
        )

    return render_template(
        "fixed_income_extension_product.html",
        config=config,
        product_slug=product_slug,
        field_sections=field_sections,
        form_data=form_data,
        results=results,
        pricing_error=pricing_error,
        schedule_rows=schedule_rows,
        exercise_rows=exercise_rows,
        discount_factor_rows=discount_factor_rows,
        zero_curve_rows=zero_curve_rows,
        holiday_rows=holiday_rows,
        benchmark_presets=benchmark_presets,
        selected_preset=selected_preset,
        selected_preset_label=selected_preset_label,
        restored_run=restored_run,
    )
