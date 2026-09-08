"""Reproduce the callable amortizing bond benchmark defaults."""

from derivapro.routes.bonds import (
    FIXED_INCOME_EXTENSION_CONFIGS,
    _default_form_data,
    _price_fixed_income_extension,
)


EXTERNAL = {
    20: {"price": 70.8033979, "straight": 101.7476543, "option": 30.9442564,
         "call_probability": 0.33939017, "duration": 2.40067326, "convexity": 9.15839942},
    10: {"price": 84.61036104, "straight": 101.7476542, "option": 17.13729311,
         "call_probability": 0.51367674, "duration": 2.1381096, "convexity": 8.289697491},
}


def metric_map(result):
    return {item["label"]: item["value"] for item in result["benchmark_metrics"]}


def main():
    config = FIXED_INCOME_EXTENSION_CONFIGS["callable-amortizing-bond"]
    for volatility_pct, external in EXTERNAL.items():
        form = _default_form_data(config)
        form["short_rate_volatility_pct"] = str(volatility_pct)
        result = _price_fixed_income_extension("callable-amortizing-bond", form)
        metrics = metric_map(result)
        derived = {
            "price": metrics["Clean Price Including Option"],
            "straight": metrics["Straight Bond Clean Price"],
            "option": metrics["Embedded Option Value"],
            "call_probability": metrics["Probability of Call"],
            "duration": metrics["Effective Duration"],
            "convexity": metrics["Effective Convexity"],
        }
        print(f"\nVolatility: {volatility_pct:.2f}%")
        print(f"{'Metric':<20} {'DerivaPro':>14} {'External':>14} {'Difference %':>14}")
        for name, value in derived.items():
            reference = external[name]
            difference = (value - reference) / reference * 100 if reference else float("nan")
            print(f"{name:<20} {value:>14.8f} {reference:>14.8f} {difference:>13.4f}%")
        print(f"Curve fit error: {metrics['Maximum Discount Factor Fit Error']:.3e}")
        print(f"Tree/cashflow straight-PV error: {metrics['Straight Bond Tree vs Cashflow PV Error']:.3e}")


if __name__ == "__main__":
    main()
