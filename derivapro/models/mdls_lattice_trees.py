# Last updated Sep 08
from __future__ import annotations

import math
import logging
from typing import Union

from ..utils.lazy_imports import LazyAttribute, LazyImport

logger = logging.getLogger(__name__)

StockData = LazyAttribute("derivapro.models.market_data", "StockData")
np = LazyImport("numpy")
plt = LazyImport("matplotlib.pyplot")
sns = LazyImport("seaborn")


class LatticeModel:
    def __init__(
        self,
        ticker: str,
        strike_price: Union[float, int],
        start_date: str,
        end_date: str,
        risk_free_rate: float,
        volatility: float,
        spot_price: Union[float, int, None] = None,
        time_to_expiry: Union[float, None] = None,
    ) -> None:
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.strike_price = strike_price

        # Pages that collect the spot and tenor from the user pass them in
        # directly; only fall back to a market-data lookup when they are absent,
        # so the model no longer requires a resolvable ticker.
        if spot_price is None or time_to_expiry is None:
            market_data = StockData(ticker, start_date, end_date)
            if time_to_expiry is None:
                time_to_expiry = market_data.get_years_difference()
            if spot_price is None:
                spot_price = float(market_data.get_closing_price())

        self.time_to_expiry = float(time_to_expiry)
        self.spot_price = float(spot_price)
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility

    def Cox_Ross_Rubinstein_Tree(
        self,
        option_type: str = "call",
        steps: int = 100,
        plot_vis: str = "no",
        greeks: bool = False,
    ) -> Union[float, dict[str, float]]:
        logger.debug(
            "Running Cox Ross Rubinstein Tree: steps=%s, option_type=%s",
            steps,
            option_type,
        )

        u = math.exp(self.volatility * math.sqrt(self.time_to_expiry / steps))
        d = math.exp(-self.volatility * math.sqrt(self.time_to_expiry / steps))
        pu = ((math.exp(self.risk_free_rate * self.time_to_expiry / steps)) - d) / (
            u - d
        )
        pd = 1 - pu
        disc = math.exp(-self.risk_free_rate * self.time_to_expiry / steps)

        St: list[float] = [0] * (steps + 1)
        C: list[float] = [0] * (steps + 1)

        St[0] = self.spot_price * d**steps

        for j in range(1, steps + 1):
            St[j] = St[j - 1] * u / d

        for j in range(1, steps + 1):
            if option_type.lower() == "put":
                C[j] = max(self.strike_price - St[j], 0)
            elif option_type.lower() == "call":
                C[j] = max(St[j] - self.strike_price, 0)

        for i in range(steps, 0, -1):
            for j in range(0, i):
                C[j] = disc * (pu * C[j + 1] + pd * C[j])

        # Calculate Greeks
        delta = (C[1] - C[0]) / (St[1] - St[0]) if steps > 0 else 0
        gamma = (
            ((C[2] - C[1]) / (St[2] - St[1]) - (C[1] - C[0]) / (St[1] - St[0]))
            / ((St[1] - St[0]) / 2)
            if steps > 1
            else 0
        )

        if plot_vis.lower() == "yes":
            stock_prices = [self.spot_price * (u**i) for i in range(steps + 1)]
            plt.plot(stock_prices, C, label="Option Price")
            plt.xlabel("Stock Price")
            plt.ylabel("Option Price")
            plt.title("Option Price vs. Stock Price")
            plt.legend()
            plt.grid(True)
            plt.show()

        if greeks:
            return {"Delta": delta, "Gamma": gamma, "option_price": C[0]}
        else:
            return C[0]

    def CRRGreeks(self, option_type: str, steps: int) -> dict[str, float]:
        delta = self.Cox_Ross_Rubinstein_Tree(option_type, steps, greeks=True)["Delta"]
        gamma = self.Cox_Ross_Rubinstein_Tree(option_type, steps, greeks=True)["Gamma"]
        option_price = self.Cox_Ross_Rubinstein_Tree(option_type, steps)

        # Calculate Theta (using a small change in time)
        small_change = 1e-4  # A small change in time
        price_up = self.Cox_Ross_Rubinstein_Tree(option_type, steps, greeks=False)
        self.time_to_expiry -= small_change
        price_down = self.Cox_Ross_Rubinstein_Tree(option_type, steps, greeks=False)
        theta = (price_down - price_up) / (2 * small_change)
        self.time_to_expiry += small_change

        # Calculate Vega (using finite difference for volatility)
        small_vol_change = 0.01  # A small change in volatility
        price_up_vol = self.Cox_Ross_Rubinstein_Tree(option_type, steps, greeks=False)
        self.volatility += small_vol_change
        price_down_vol = self.Cox_Ross_Rubinstein_Tree(option_type, steps, greeks=False)
        vega = (price_down_vol - price_up_vol) / (2 * small_vol_change)
        self.volatility -= small_vol_change

        # Calculate Rho (using finite difference for interest rate)
        small_rate_change = 0.01  # A small change in risk-free rate
        price_up_rate = self.Cox_Ross_Rubinstein_Tree(option_type, steps, greeks=False)
        self.risk_free_rate += small_rate_change
        price_down_rate = self.Cox_Ross_Rubinstein_Tree(
            option_type, steps, greeks=False
        )
        rho = (price_down_rate - price_up_rate) / (2 * small_rate_change)
        self.risk_free_rate -= small_rate_change

        return {
            "Delta": delta,
            "Gamma": gamma,
            "Theta": theta,
            "Vega": vega,
            "Rho": rho,
            "option_price": option_price,
        }

    def Jarrow_Rudd_Tree(
        self,
        option_type: str = "call",
        steps: int = 100,
        plot_vis: str = "no",
        greeks: bool = False,
    ) -> Union[float, dict[str, float]]:
        logger.debug(
            "Running Jarrow Rudd Tree: steps=%s, option_type=%s",
            steps,
            option_type,
        )

        u = math.exp(
            (self.risk_free_rate - (self.volatility**2 / 2))
            * self.time_to_expiry
            / steps
            + self.volatility * math.sqrt(self.time_to_expiry / steps)
        )
        d = math.exp(
            (self.risk_free_rate - (self.volatility**2 / 2))
            * self.time_to_expiry
            / steps
            - self.volatility * math.sqrt(self.time_to_expiry / steps)
        )
        pu = 0.5
        pd = 1 - pu
        disc = math.exp(-self.risk_free_rate * self.time_to_expiry / steps)

        St: list[float] = [0] * (steps + 1)
        C: list[float] = [0] * (steps + 1)

        St[0] = self.spot_price * d**steps

        for j in range(1, steps + 1):
            St[j] = St[j - 1] * u / d

        for j in range(1, steps + 1):
            if option_type.lower() == "put":
                C[j] = max(self.strike_price - St[j], 0)
            elif option_type.lower() == "call":
                C[j] = max(St[j] - self.strike_price, 0)

        for i in range(steps, 0, -1):
            for j in range(0, i):
                C[j] = disc * (pu * C[j + 1] + pd * C[j])

        # Calculate Greeks
        delta = (C[1] - C[0]) / (St[1] - St[0]) if steps > 0 else 0
        gamma = (
            ((C[2] - C[1]) / (St[2] - St[1]) - (C[1] - C[0]) / (St[1] - St[0]))
            / ((St[1] - St[0]) / 2)
            if steps > 1
            else 0
        )

        if plot_vis.lower() == "yes":
            stock_prices = [self.spot_price * (u**i) for i in range(steps + 1)]
            plt.plot(stock_prices, C, label="Option Price")
            plt.xlabel("Stock Price")
            plt.ylabel("Option Price")
            plt.title("Option Price vs. Stock Price")
            plt.legend()
            plt.grid(True)
            plt.show()
        if greeks:
            return {"Delta": delta, "Gamma": gamma, "option_price": C[0]}
        else:
            return C[0]

    def JRTGreeks(self, option_type: str, steps: int) -> dict[str, float]:
        result = self.Jarrow_Rudd_Tree(option_type, steps, greeks=True)
        delta = result["Delta"]
        gamma = result["Gamma"]

        # Calculate Theta (using a small change in time)
        small_change = 1e-4  # A small change in time
        price_up = self.Jarrow_Rudd_Tree(option_type, steps)
        self.time_to_expiry -= small_change
        price_down = self.Jarrow_Rudd_Tree(option_type, steps)
        theta = (price_down - price_up) / (2 * small_change)
        self.time_to_expiry += small_change
        # Calculate Vega (using finite difference for volatility)
        small_vol_change = 0.01  # A small change in volatility
        price_up_vol = self.Jarrow_Rudd_Tree(option_type, steps)
        self.volatility += small_vol_change
        price_down_vol = self.Jarrow_Rudd_Tree(option_type, steps)
        vega = (price_down_vol - price_up_vol) / (2 * small_vol_change)
        self.volatility -= small_vol_change
        # Calculate Rho (using finite difference for interest rate)
        small_rate_change = 0.01  # A small change in risk-free rate
        price_up_rate = self.Jarrow_Rudd_Tree(option_type, steps)
        self.risk_free_rate += small_rate_change
        price_down_rate = self.Jarrow_Rudd_Tree(option_type, steps)
        rho = (price_down_rate - price_up_rate) / (2 * small_rate_change)
        self.risk_free_rate -= small_rate_change

        return {
            "Delta": delta,
            "Gamma": gamma,
            "Theta": theta,
            "Vega": vega,
            "Rho": rho,
        }

    ## define a calculator to see optimal number of steps
    def step_optimization(
        self,
        option_type: str = "call",
        start: int = 10,
        step: int = 50,
        limit: int = 1000,
    ) -> None:
        runs1 = list(range(start, limit, step))
        CRR1 = []
        JR1 = []

        for i in runs1:
            CRR1.append(
                LatticeModel(
                    self.ticker,
                    self.strike_price,
                    self.start_date,
                    self.end_date,
                    self.risk_free_rate,
                    self.volatility,
                ).Cox_Ross_Rubinstein_Tree(option_type=option_type, steps=i)
            )
            JR1.append(
                LatticeModel(
                    self.ticker,
                    self.strike_price,
                    self.start_date,
                    self.end_date,
                    self.risk_free_rate,
                    self.volatility,
                ).Jarrow_Rudd_Tree(option_type=option_type, steps=i)
            )

        plt.plot(runs1, CRR1, label="Cox Ross Rubinstein")
        plt.plot(runs1, JR1, label="Jarrow Rudd")
        plt.legend(loc="upper right")
        plt.show()

    def Trinomial_Asset_Pricing(
        self,
        option_type: str = "call",
        steps: int = 100,
        plot_vis: str = "no",
        american: bool = False,
        dividend_yield: float = 0.0,
    ) -> float:
        """Boyle trinomial tree price for a European or American option.

        The tree recombines, so step ``i`` holds exactly ``2i + 1`` distinct
        nodes indexed by net log displacement ``j`` in ``[-i, i]``. The previous
        implementation allocated two ``(steps+1)^3`` arrays and walked them with
        three nested loops, which both wasted memory and did not describe a
        recombining lattice. This version rolls a single vector of length
        ``2*steps + 1`` backwards through time.

        Set ``american=True`` for early exercise. ``dividend_yield`` enters the
        drift only - discounting stays at the risk-free rate.
        """
        logger.debug(
            "Running Trinomial Asset Pricing: steps=%s, option_type=%s, american=%s",
            steps,
            option_type,
            american,
        )

        option_type = option_type.lower()
        if option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'.")
        if steps < 1:
            raise ValueError("steps must be at least 1.")

        deltaT = self.time_to_expiry / steps
        D = (self.risk_free_rate - dividend_yield) - (0.5 * self.volatility**2)

        deltaX = np.sqrt(
            deltaT * (self.volatility**2) + (D**2) * (deltaT**2)
        )
        # Convergence floor: below this spacing the trinomial probabilities go
        # negative and the tree becomes unstable.
        if deltaX < self.volatility * np.sqrt(3 * deltaT):
            deltaX = self.volatility * np.sqrt(3 * deltaT)

        variance_term = (self.volatility**2 * deltaT + D**2 * deltaT**2) / deltaX**2
        drift_term = deltaT * D / deltaX

        pu = 0.5 * (variance_term + drift_term)
        pm = 1 - variance_term
        pd = 0.5 * (variance_term - drift_term)

        disc = np.exp(-self.risk_free_rate * deltaT)

        # Node prices for every reachable displacement, indexed by j + steps.
        displacements = np.arange(-steps, steps + 1)
        spot_grid = self.spot_price * np.exp(displacements * deltaX)

        if option_type == "call":
            intrinsic = np.maximum(spot_grid - self.strike_price, 0.0)
        else:
            intrinsic = np.maximum(self.strike_price - spot_grid, 0.0)

        values = intrinsic.copy()

        for i in range(steps - 1, -1, -1):
            lo = steps - i
            hi = steps + i
            continuation = disc * (
                pu * values[lo + 1:hi + 2]
                + pm * values[lo:hi + 1]
                + pd * values[lo - 1:hi]
            )
            if american:
                continuation = np.maximum(continuation, intrinsic[lo:hi + 1])
            values[lo:hi + 1] = continuation

        return float(values[steps])

    def TAPGreeks(self, option_type: str, steps: int) -> dict[str, float]:
        option_price = self.Trinomial_Asset_Pricing(option_type, steps)

        original_spot_price = self.spot_price
        self.spot_price *= 1.01
        price_up = self.Trinomial_Asset_Pricing(option_type, steps)
        self.spot_price = original_spot_price * 0.99
        price_down = self.Trinomial_Asset_Pricing(option_type, steps)
        delta = (price_up - price_down) / (self.spot_price * 0.02)
        self.spot_price = original_spot_price

        # Gamma
        gamma = (price_up - 2 * option_price + price_down) / (self.spot_price * 0.01**2)
        self.spot_price = original_spot_price

        # Theta
        small_change = 1e-4
        price_now = self.Trinomial_Asset_Pricing(option_type, steps)
        self.time_to_expiry -= small_change
        price_down = self.Trinomial_Asset_Pricing(option_type, steps)
        theta = (price_down - price_now) / small_change
        self.time_to_expiry += small_change

        # Vega
        small_vol_change = 0.01
        price_up_vol = self.Trinomial_Asset_Pricing(option_type, steps)
        self.volatility += small_vol_change
        price_down_vol = self.Trinomial_Asset_Pricing(option_type, steps)
        vega = (price_down_vol - price_up_vol) / small_vol_change
        self.volatility -= small_vol_change

        # Rho
        small_rate_change = 0.01
        price_up_rate = self.Trinomial_Asset_Pricing(option_type, steps)
        self.risk_free_rate += small_rate_change
        price_down_rate = self.Trinomial_Asset_Pricing(option_type, steps)
        rho = (price_down_rate - price_up_rate) / small_rate_change
        self.risk_free_rate -= small_rate_change

        return {
            "option_price": option_price,
            "Delta": delta,
            "Gamma": gamma,
            "Theta": theta,
            "Vega": vega,
            "Rho": rho,
        }

    def risk_pl_analysis(
        self,
        option_type: str = "call",
        steps: int = 100,
        price_change: float = 0.01,
        vol_change: float = 0.01,
        model: str = "CRR",
    ) -> dict[str, float]:
        """
        Risk-Based P&L Analysis for American Options.

        :param option_type: 'call' or 'put'
        :param steps: number of steps in the binomial/trinomial tree
        :param price_change: percentage change in the spot price for the P&L analysis
        :param model: tree model to use ('CRR', 'JR', 'TAP')
        :return: P&L results based on model and sensitivity analysis
        """
        if model == "CRR":
            price_initial = self.Cox_Ross_Rubinstein_Tree(option_type, steps)
        elif model == "JRT":
            price_initial = self.Jarrow_Rudd_Tree(option_type, steps)
        elif model == "TAP":
            price_initial = self.Trinomial_Asset_Pricing(option_type, steps)
        else:
            raise ValueError("Invalid model selection. Choose 'CRR', 'JRT', or 'TAP'.")

        original_spot_price = self.spot_price
        original_volatility = self.volatility

        # Adjust the spot price up and down by the specified percentage change
        self.spot_price = original_spot_price * (1 + price_change)
        self.volatility = original_volatility * (1 + vol_change)
        if model == "CRR":
            price_bump = self.Cox_Ross_Rubinstein_Tree(option_type, steps)
            delta_pl = self.CRRGreeks(option_type, steps)["Delta"] * (1 + price_change)
            gamma_pl = (
                self.CRRGreeks(option_type, steps)["Gamma"]
                * 0.5
                * (1 + price_change) ** 2
            )
            vega_pl = self.CRRGreeks(option_type, steps)["Vega"] * (1 + vol_change)
        elif model == "JRT":
            price_bump = self.Jarrow_Rudd_Tree(option_type, steps)
            delta_pl = self.JRTGreeks(option_type, steps)["Delta"] * (1 + price_change)
            gamma_pl = (
                self.JRTGreeks(option_type, steps)["Gamma"]
                * 0.5
                * (1 + price_change) ** 2
            )
            vega_pl = self.JRTGreeks(option_type, steps)["Vega"] * (1 + vol_change)
        elif model == "TAP":
            price_bump = self.Trinomial_Asset_Pricing(option_type, steps)
            delta_pl = self.TAPGreeks(option_type, steps)["Delta"] * (1 + price_change)
            gamma_pl = (
                self.TAPGreeks(option_type, steps)["Gamma"]
                * 0.5
                * (1 + price_change) ** 2
            )
            vega_pl = self.TAPGreeks(option_type, steps)["Vega"] * (1 + vol_change)

        # Reset spot price to original
        self.spot_price = original_spot_price
        self.volatility = original_volatility

        # Return P&L results
        return {
            "Initial Price": price_initial,
            "Bumped Price": price_bump,
            "Actual P&L": price_bump - price_initial,
            "Delta P&L": delta_pl,
            "Vega P&L": vega_pl,
            "Gamma P&L": gamma_pl,
            "Greek P&L Sum": (delta_pl + vega_pl + gamma_pl),
            "Difference": (price_bump - price_initial)
            - (delta_pl + vega_pl + gamma_pl),
        }


class AmericanOptionSmoothnessTest:
    def __init__(
        self,
        ticker: str,
        strike_price: Union[float, int],
        start_date: str,
        end_date: str,
        risk_free_rate: float,
        volatility: float,
        model: str,
        option_type: str,
        num_steps: int,
    ) -> None:
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.strike_price = strike_price
        self.time_to_expiry = StockData(
            ticker, start_date, end_date
        ).get_years_difference()
        self.spot_price = float(
            StockData(ticker, start_date, end_date).get_closing_price()
        )
        self.risk_free_rate = risk_free_rate
        self.volatility = volatility
        self.model = model
        self.option_type = option_type
        self.num_steps = num_steps

    def generate_variable_range(
        self, variable: str, num_steps: int, range_span: float
    ) -> np.ndarray:
        if variable == "strike_price":
            base = self.strike_price
        elif variable == "risk_free_rate":
            base = self.risk_free_rate
        elif variable == "volatility":
            base = self.volatility
        else:
            raise ValueError(
                "Unsupported variable type. Choose from 'strike_price', 'risk_free_rate', 'volatility'."
            )

        return np.linspace(base - range_span, base + range_span, num_steps)

    def calculate_single_greek(
        self, option: LatticeModel, target_variable: str
    ) -> float:
        if self.model == "CRR":
            if target_variable == "option_price":
                return option.CRRGreeks(self.option_type, self.num_steps)[
                    target_variable
                ]
            else:
                return option.CRRGreeks(self.option_type, self.num_steps)[
                    target_variable[0].upper() + target_variable[1:]
                ]
        elif self.model == "JRT":
            if target_variable == "option_price":
                return option.JRTGreeks(self.option_type, self.num_steps)[
                    target_variable
                ]
            else:
                return option.JRTGreeks(self.option_type, self.num_steps)[
                    target_variable[0].upper() + target_variable[1:]
                ]
        else:
            if target_variable == "option_price":
                return option.TAPGreeks(self.option_type, self.num_steps)[
                    target_variable
                ]
            else:
                return option.TAPGreeks(self.option_type, self.num_steps)[
                    target_variable[0].upper() + target_variable[1:]
                ]

    def calculate_greeks_over_range(
        self, variable: str, num_steps: int, range_span: float, target_variable: str
    ) -> tuple[np.ndarray, list[float]]:
        variable_values = self.generate_variable_range(variable, num_steps, range_span)
        greek_values = []

        for value in variable_values:
            if variable == "strike_price":
                option = LatticeModel(
                    self.ticker,
                    value,
                    self.start_date,
                    self.end_date,
                    self.risk_free_rate,
                    self.volatility,
                )
            elif variable == "risk_free_rate":
                option = LatticeModel(
                    self.ticker,
                    self.strike_price,
                    self.start_date,
                    self.end_date,
                    value,
                    self.volatility,
                )
            elif variable == "volatility":
                option = LatticeModel(
                    self.ticker,
                    self.strike_price,
                    self.start_date,
                    self.end_date,
                    self.risk_free_rate,
                    value,
                )
            else:
                raise ValueError(
                    "Unsupported variable type. Choose from 'strike_price', 'risk_free_rate', 'volatility'."
                )

            greek_value = self.calculate_single_greek(option, target_variable)
            greek_values.append(greek_value)

        logger.debug("Variable values: %s", variable_values)
        logger.debug("Greek values: %s", greek_values)

        return variable_values, greek_values

    def plot_single_greek(
        self,
        variable_values: np.ndarray,
        greek_values: list[float],
        target_variable: str,
        variable_name: str,
    ) -> None:
        plt.figure(figsize=(10, 6))
        plt.plot(
            variable_values, greek_values, label=target_variable.capitalize(), color="b"
        )
        plt.title(f"{target_variable.capitalize()} vs {variable_name.capitalize()}")
        plt.xlabel(variable_name.capitalize())
        plt.ylabel(target_variable.capitalize())
        plt.tight_layout()


def lattice_convergence_test(
    max_steps: int,
    max_sims: int,
    obs: int,
    pricer_class: type,
    pricer_params: dict,
    model: str,
    option_type: str,
    mode: str = "steps",
) -> list[tuple[int, float]]:
    logger.debug("lattice_convergence_test called with model: %r", model)

    if mode == "steps":
        steps = list(np.linspace(0, max_steps, obs).round().astype(int))
        steps.pop(0)
        sims = max_sims
    elif mode == "simulations":
        steps = max_steps
        sims = list(np.linspace(0, max_sims, obs).round().astype(int))
        sims.pop(0)
    else:
        raise ValueError("Invalid mode. Choose 'steps' or simulations'.")

    results = []

    for param in steps if mode == "steps" else sims:
        N = param if mode == "steps" else steps
        M = sims if mode == "steps" else param

        current_params = pricer_params.copy()
        current_params["N"] = N
        current_params["M"] = M

        current_params = pricer_params.copy()
        current_params["risk_free_rate"] = current_params.pop("r")
        current_params["volatility"] = current_params.pop("sigma")

        del current_params["max_steps"]
        del current_params["option_type"]
        del current_params["num_steps"]
        del current_params["model"]
        del current_params["obs"]
        del current_params["mode"]

        for key in ["num_paths", "mc_steps"]:
            if key in current_params:
                del current_params[key]

        lattice = pricer_class(**current_params)

        # Generates paths using the provided path_generator function
        if model == "Cox Ross Rubinstein Tree":
            option_price = lattice.Cox_Ross_Rubinstein_Tree(option_type, steps=N)
        elif model == "Jarrow Rudd Tree":
            option_price = lattice.Jarrow_Rudd_Tree(option_type, steps=N)
        elif model == "Trinomial Asset Pricing":
            option_price = lattice.Trinomial_Asset_Pricing(option_type, steps=N)
        else:
            raise ValueError(
                "Invalid model. Choose 'Cox Ross Rubinstein Tree', 'Jarrow Rudd Tree', or 'Trinomial Asset Pricing'."
            )

        # results.append((param, option_price))
        results.append((int(param), float(option_price)))
    return results


def plot_convergence(results, mode):
    x, y = zip(*results)
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    ax = sns.lineplot(x=x, y=y, marker="o")

    if mode == "steps":
        plt.xlabel("Number of Time Steps")
        plt.title("Option Price Convergence with Increasing Time Steps")
    elif mode == "simulations":
        plt.xlabel("Number of Simulations")
        plt.title("Option Price Convergence with Increasing Simulations")

    plt.ylabel("Option Price")
    plt.tight_layout()
