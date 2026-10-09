from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping


SIMULATION_METHOD_CHOICES = {
    "gbm": "Geometric Brownian Motion",
}

RANDOM_SEQUENCE_CHOICES = {
    "sobol": "Sobol low-discrepancy",
    "pseudo": "Pseudo-random",
}

VARIANCE_REDUCTION_CHOICES = {
    "none": "None",
    "antithetic": "Antithetic variates",
}

CONVERGENCE_MODE_CHOICES = {
    "fixed": "Fixed path count",
    "standard_error": "Target standard error",
}

RUNTIME_PROFILE_CHOICES = {
    "fast": "Fast preview",
    "balanced": "Balanced",
    "high_accuracy": "High accuracy",
}


@dataclass(frozen=True)
class SimulationSettings:
    simulation_method: str = "gbm"
    random_sequence: str = "sobol"
    num_paths: int = 10_000
    num_steps: int = 252
    random_seed: int = 42
    variance_reduction: str = "none"
    convergence_mode: str = "fixed"
    target_standard_error: float = 0.0
    runtime_profile: str = "balanced"
    apply_globally: bool = True
    allow_product_override: bool = True

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "SimulationSettings":
        source = dict(data or {})
        defaults = asdict(cls())

        def normalized_choice(key: str, allowed: set[str]) -> str:
            raw_value = str(source.get(key, defaults[key])).strip()
            return raw_value if raw_value in allowed else defaults[key]

        def normalized_int(key: str, min_value: int, max_value: int) -> int:
            try:
                value = int(float(source.get(key, defaults[key])))
            except (TypeError, ValueError):
                value = defaults[key]
            return max(min_value, min(value, max_value))

        def normalized_float(key: str, min_value: float, max_value: float) -> float:
            try:
                value = float(source.get(key, defaults[key]))
            except (TypeError, ValueError):
                value = defaults[key]
            return max(min_value, min(value, max_value))

        return cls(
            simulation_method=normalized_choice(
                "simulation_method",
                set(SIMULATION_METHOD_CHOICES),
            ),
            random_sequence=normalized_choice(
                "random_sequence",
                set(RANDOM_SEQUENCE_CHOICES),
            ),
            num_paths=normalized_int("num_paths", 100, 500_000),
            num_steps=normalized_int("num_steps", 1, 5_000),
            random_seed=normalized_int("random_seed", 0, 2_147_483_647),
            variance_reduction=normalized_choice(
                "variance_reduction",
                set(VARIANCE_REDUCTION_CHOICES),
            ),
            convergence_mode=normalized_choice(
                "convergence_mode",
                set(CONVERGENCE_MODE_CHOICES),
            ),
            target_standard_error=normalized_float(
                "target_standard_error",
                0.0,
                1_000_000.0,
            ),
            runtime_profile=normalized_choice(
                "runtime_profile",
                set(RUNTIME_PROFILE_CHOICES),
            ),
            apply_globally=bool(source.get("apply_globally", defaults["apply_globally"])),
            allow_product_override=bool(
                source.get("allow_product_override", defaults["allow_product_override"])
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def display_rows(self) -> list[dict[str, str]]:
        return [
            {
                "label": "Simulation Method",
                "value": SIMULATION_METHOD_CHOICES[self.simulation_method],
            },
            {
                "label": "Random Sequence",
                "value": RANDOM_SEQUENCE_CHOICES[self.random_sequence],
            },
            {"label": "Paths", "value": f"{self.num_paths:,}"},
            {"label": "Time Steps", "value": f"{self.num_steps:,}"},
            {"label": "Random Seed", "value": str(self.random_seed)},
            {
                "label": "Variance Reduction",
                "value": VARIANCE_REDUCTION_CHOICES[self.variance_reduction],
            },
            {
                "label": "Convergence Mode",
                "value": CONVERGENCE_MODE_CHOICES[self.convergence_mode],
            },
            {
                "label": "Runtime Profile",
                "value": RUNTIME_PROFILE_CHOICES[self.runtime_profile],
            },
        ]


def default_simulation_settings() -> SimulationSettings:
    return SimulationSettings()
