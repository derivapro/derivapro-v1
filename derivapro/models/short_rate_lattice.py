"""Curve-fitted OU trinomial lattice for schedule-driven interest-rate claims."""

import math

import numpy as np
import QuantLib as ql
from scipy.optimize import brentq
from scipy.special import logsumexp


class ShortRateLattice:
    """Expose rollback and state probabilities absent from the Python bond engines.

    QuantLib supplies the event grid and exact OU transition moments. The custom
    rollback supports amortization, notice periods and exercise diagnostics.
    """

    def __init__(self, curve, times, steps_per_year, mean_reversion, volatility, model):
        if model not in {"hull_white", "black_karasinski"}:
            raise ValueError("Unsupported short-rate model.")
        if not all(math.isfinite(v) and v >= 0 for v in (mean_reversion, volatility)):
            raise ValueError("Volatility and mean reversion must be finite and nonnegative.")
        mandatory = sorted(set([0.0, *times]))
        self.times = np.array(list(ql.TimeGrid(mandatory, max(1, math.ceil(mandatory[-1] * steps_per_year)))))
        if len(self.times) > 2501:
            raise ValueError("Tree exceeds 2,500 time steps; reduce refinement or maturity.")
        self.states = [np.array([0.0])]
        self.branches = []
        self.probabilities = []
        self.discounts = []
        self.state_prices = [np.array([1.0])]
        process = ql.OrnsteinUhlenbeckProcess(mean_reversion, volatility)
        self.curve_fit_error = 0.0
        for i, delta in enumerate(np.diff(self.times)):
            x = self.states[i]
            if volatility == 0:
                next_x = np.array([0.0])
                branches = np.zeros((len(x), 1), dtype=int)
                probabilities = np.ones((len(x), 1))
            else:
                variance = process.variance(float(self.times[i]), 0.0, float(delta))
                spacing = math.sqrt(3 * variance)
                mean = x * process.expectation(0.0, 1.0, float(delta))
                centers = np.floor(mean / spacing + 0.5).astype(int)
                error = mean - centers * spacing
                indices = centers[:, None] + np.array([-1, 0, 1])
                next_x = np.arange(indices.min(), indices.max() + 1) * spacing
                if len(next_x) > 20000:
                    raise ValueError("Event grid creates too many tree states; review closely spaced dates.")
                branches = indices - indices.min()
                probabilities = np.column_stack(((1 + error**2 / variance - error * math.sqrt(3 / variance)) / 6,
                                                 (2 - error**2 / variance) / 3,
                                                 (1 + error**2 / variance + error * math.sqrt(3 / variance)) / 6))
                if probabilities.min() < -1e-12:
                    raise ValueError("Invalid lattice transition probability.")
            state_prices = self.state_prices[i]
            target = curve.df(float(self.times[i + 1]))
            if not math.isfinite(target) or target <= 0:
                raise ValueError("Discount curve must be finite and positive on the tree grid.")
            if model == "hull_white":
                positive = state_prices > 0
                alpha = (logsumexp(np.log(state_prices[positive]) - x[positive] * delta) - math.log(target)) / delta
                discount = np.exp(-(x + alpha) * delta)
            else:
                if target >= state_prices.sum():
                    raise ValueError("Black-Karasinski requires strictly positive forward rates on the tree grid.")
                def discounts(alpha):
                    return np.exp(-np.exp(np.minimum(x + alpha, 700)) * delta)
                def objective(alpha):
                    return float(state_prices @ discounts(alpha)) - target
                bound = float(np.max(np.abs(x))) + 50
                alpha = brentq(objective, -bound, bound, xtol=1e-13)
                discount = discounts(alpha)
            next_prices = np.bincount(branches.ravel(),
                                     weights=(state_prices[:, None] * discount[:, None] * probabilities).ravel(),
                                     minlength=len(next_x))
            self.curve_fit_error = max(self.curve_fit_error, abs(float(next_prices.sum()) - target))
            self.states.append(next_x)
            self.branches.append(branches)
            self.probabilities.append(probabilities)
            self.discounts.append(discount)
            self.state_prices.append(next_prices)

    def index(self, time):
        index = int(np.argmin(abs(self.times - time)))
        if abs(self.times[index] - time) > 1e-9:
            raise ValueError("Cashflow or exercise date is missing from the event grid.")
        return index

    def rollback(self, values, step, spread=0.0):
        expected = (values[self.branches[step]] * self.probabilities[step]).sum(axis=1)
        return expected * self.discounts[step] * math.exp(-spread * (self.times[step + 1] - self.times[step]))

    def propagate(self, probabilities, step):
        return np.bincount(self.branches[step].ravel(),
                           weights=(probabilities[:, None] * self.probabilities[step]).ravel(),
                           minlength=len(self.states[step + 1]))
