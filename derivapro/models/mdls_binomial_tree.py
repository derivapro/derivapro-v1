import math
import datetime
import warnings
import numpy as np
from scipy.stats import norm
from derivapro.models.market_data import StockData


class BinomialTreeEngineCRR:
    """
    CRR binomial engine with support for discrete proportional dividends.
    It splits original intervals so that dividends fall exactly on internal step boundaries,
    then runs a variable-step CRR (u_i, d_i, p^u_i, discount_i).
    """

    def __init__(
        self,
        ticker,
        strike_price,
        start_date,
        end_date,
        risk_free_rate,
        volatility,
        num_steps=100,
        option_type="call",
        dividends=None,
        **kwargs,
    ):
        self.ticker = ticker
        self.start_date = self._to_date(start_date)
        self.end_date = self._to_date(end_date)

        stock_data = StockData(ticker, self.start_date, self.end_date)
        self.S0 = float(stock_data.get_closing_price())
        self.K = float(strike_price)

        self.T = float(stock_data.get_years_difference())
        self.r = float(risk_free_rate)
        self.sigma = float(volatility)
        self.N = int(num_steps)
        self.option_type = option_type.lower()
        self.epsilon = 1 if self.option_type == "call" else -1

        self.dividends = sorted(dividends or [], key=lambda x: self._to_date(x[0]))

        self._build_dt_schedule()
        self._rebuild_params_variable_steps()
        self._map_dividends_to_internal_steps()
        self._validate_parameters()

    def _to_date(self, d):
        if isinstance(d, datetime.date):
            return d
        if isinstance(d, str):
            return datetime.datetime.strptime(d, "%Y-%m-%d").date()
        raise ValueError("Dates must be datetime.date or 'YYYY-MM-DD' strings")

    def diagnostics(self):
        print("[CRR Diagnostics] Inputs")
        print(
            f"  S0={self.S0:.6f}, K={self.K:.6f}, T={self.T:.6f}, r={self.r:.6f}, sigma={self.sigma:.6f}"
        )
        print(f"  Requested N={self.N}, Internal N_steps={self.N_steps}")
        print("[CRR Diagnostics] Derived params (first/last step shown)")
        print(f"  dt_first={self.dt_list[0]:.8f}, dt_last={self.dt_list[-1]:.8f}")
        print(
            f"  u_first={self.u_list[0]:.8f}, d_first={self.d_list[0]:.8f}, pu_first={self.pu_list[0]:.8f}"
        )
        print(f"  discount_first={self.discount_list[0]:.8f}")
        F = self._setup_forward_tree()
        S = self._setup_spot_tree(F)
        print("[CRR Diagnostics] Spot tree checks")
        print(f"  S[0,0]={S[0, 0]:.6f} (should equal S0)")
        print(
            f"  S[N_steps,0]={S[self.N_steps, 0]:.6f}, S[N_steps,N_steps]={S[self.N_steps, self.N_steps]:.6f}"
        )
        if self._internal_divs:
            print("[CRR Diagnostics] Internal dividends at steps:")
            for d in self._internal_divs:
                if d.get("type") == "prop":
                    print(
                        f"   - step_index={d['step_index']}, type=prop, q={d['q']:.6f}, t_rel={d['t_rel']:.6f}"
                    )
                else:
                    print(
                        f"   - step_index={d['step_index']}, type=cash, D={d.get('D')}, CF={d.get('CF')}, pv={d.get('pv', 0.0):.6f}, t_rel={d['t_rel']:.6f}"
                    )
        else:
            print("[CRR Diagnostics] No dividends detected. This is pure CRR.")

    def _validate_parameters(self):
        if self.S0 <= 0:
            raise ValueError(f"Spot price must be positive, got {self.S0}")
        if self.K <= 0:
            raise ValueError(f"Strike price must be positive, got {self.K}")
        if self.T <= 0:
            raise ValueError(f"Time to expiry must be positive, got {self.T}")
        if self.sigma <= 0:
            raise ValueError(f"Volatility must be positive, got {self.sigma}")
        if self.N <= 0:
            raise ValueError(f"Number of steps must be positive, got {self.N}")
        if self.option_type not in ("call", "put"):
            raise ValueError(
                f"Option type must be 'call' or 'put', got {self.option_type}"
            )
        for i, pu in enumerate(self.pu_list):
            if not (0.0 <= pu <= 1.0):
                raise ValueError(
                    f"Risk-neutral probability pu at step {i} = {pu:.6f} not in [0,1]. Check parameters."
                )

    def _build_dt_schedule(self):
        dt0 = self.T / self.N
        div_times = []
        for entry in self.dividends or []:
            if not entry:
                continue
            ex_raw = entry[0]
            ex_date = self._to_date(ex_raw)
            days = (ex_date - self.start_date).days
            if days < 0:
                continue
            t_rel = days / 365.0
            if t_rel <= 0 or t_rel >= self.T:
                continue
            div_times.append(t_rel)

        if len(div_times) == 0:
            self.dt_list = [dt0] * self.N
            self.N_steps = self.N
            self.time_grid = [0.0]
            cum = 0.0
            for _ in range(self.N):
                cum += dt0
                self.time_grid.append(cum)
            return

        div_times.sort()
        tol = 1e-12

        dt_list = []
        time_grid = [0.0]
        for i in range(self.N):
            a = i * dt0
            b = (i + 1) * dt0
            inside = [t for t in div_times if (t > a + tol and t < b - tol)]
            if not inside:
                dt_list.append(dt0)
                time_grid.append(time_grid[-1] + dt0)
            else:
                boundaries = [a] + sorted(inside) + [b]
                for k in range(len(boundaries) - 1):
                    seg = boundaries[k + 1] - boundaries[k]
                    if seg <= tol:
                        continue
                    dt_list.append(seg)
                    time_grid.append(time_grid[-1] + seg)
        total = sum(dt_list)
        if abs(total - self.T) > 1e-8:
            dt_list[-1] += self.T - total
            time_grid[-1] = self.T

        self.dt_list = dt_list
        self.N_steps = len(dt_list)
        self.time_grid = [0.0]
        cum = 0.0
        for dt in dt_list:
            cum += dt
            self.time_grid.append(cum)

    def _rebuild_params_variable_steps(self):
        self.u_list = []
        self.d_list = []
        self.pu_list = []
        self.pd_list = []
        self.discount_list = []
        for dt in self.dt_list:
            u = math.exp(self.sigma * math.sqrt(dt))
            d = 1.0 / u
            pu = (math.exp(self.r * dt) - d) / (u - d)
            pu = max(0.0, min(1.0, pu))
            pd = 1.0 - pu
            disc = math.exp(-self.r * dt)
            self.u_list.append(u)
            self.d_list.append(d)
            self.pu_list.append(pu)
            self.pd_list.append(pd)
            self.discount_list.append(disc)

    def _map_dividends_to_internal_steps(self):
        self._internal_divs = []
        tol = 1e-9

        parsed = []
        for entry in self.dividends or []:
            ex_raw = entry[0]
            ex_date = self._to_date(ex_raw)
            days = (ex_date - self.start_date).days
            if days < 0:
                continue
            t_rel = days / 365.0
            if t_rel <= 0 or t_rel >= self.T:
                continue

            if len(entry) == 2:
                q = float(entry[1])
                parsed.append({"t_rel": t_rel, "type": "prop", "q": q})
            elif len(entry) >= 3:
                D = float(entry[1])
                CF = float(entry[2])
                pv_at_ex = D / CF
                parsed.append({
                    "t_rel": t_rel,
                    "type": "cash",
                    "D": D,
                    "CF": CF,
                    "pv": pv_at_ex,
                })
            else:
                raise ValueError(
                    "Dividend entries must be (ex_date, q) or (ex_date, D, CF)"
                )

        for d in parsed:
            t_rel = d["t_rel"]
            found = False
            for k in range(1, len(self.time_grid) - 1):
                if (
                    abs(self.time_grid[k] - t_rel) < tol
                    or abs(self.time_grid[k] - t_rel) < 1e-8
                ):
                    entry = d.copy()
                    entry["step_index"] = k
                    self._internal_divs.append(entry)
                    found = True
                    break
            if not found:
                nearest_k = min(
                    range(1, len(self.time_grid) - 1),
                    key=lambda k: abs(self.time_grid[k] - t_rel),
                )
                entry = d.copy()
                entry["step_index"] = nearest_k
                self._internal_divs.append(entry)
                if abs(self.time_grid[nearest_k] - t_rel) > 1e-6:
                    warnings.warn(
                        f"Dividend at t_rel={t_rel:.6f} mapped to nearest internal boundary (index {nearest_k}) "
                        f"with time {self.time_grid[nearest_k]:.6f} (mapping offset {abs(self.time_grid[nearest_k] - t_rel):.6e})."
                    )

    def _setup_forward_tree(self):
        Nn = self.N_steps
        F = np.zeros((Nn + 1, Nn + 1))

        if not getattr(self, "_internal_divs", []):
            F[0, 0] = self.S0
        else:
            prod_prop = 1.0
            cash_contrib = 0.0
            for d in self._internal_divs:
                if d.get("type") == "prop":
                    prod_prop *= 1.0 - d["q"]
                else:
                    pv = d.get("pv", d.get("D", 0.0) / max(d.get("CF", 1.0), 1e-16))
                    cash_contrib += pv * math.exp(self.r * (self.T - d["t_rel"]))
            F0 = self.S0 * math.exp(self.r * self.T) * prod_prop - cash_contrib
            if not math.isfinite(F0) or F0 <= 0.0:
                F0 = max(
                    1e-12,
                    self.S0 * math.exp(self.r * self.T) * prod_prop - cash_contrib,
                )
            F[0, 0] = F0

        for i in range(1, Nn + 1):
            d = self.d_list[i - 1]
            u = self.u_list[i - 1]
            F[i, 0] = F[i - 1, 0] * d
            for j in range(1, i + 1):
                F[i, j] = F[i - 1, j - 1] * u
        return F

    def _setup_spot_tree(self, F=None):
        Nn = self.N_steps
        S = np.zeros((Nn + 1, Nn + 1))
        S[0, 0] = self.S0

        for i in range(Nn):
            u = self.u_list[i]
            d = self.d_list[i]

            S[i + 1, 0] = S[i, 0] * d
            for j in range(1, i + 2):
                S[i + 1, j] = S[i, j - 1] * u

            for div in (dd for dd in self._internal_divs if dd["step_index"] == i + 1):
                if div["type"] == "prop":
                    q = div["q"]
                    S[i + 1, : i + 2] *= 1.0 - q
                elif div["type"] == "cash":
                    pv = div.get(
                        "pv", div.get("D", 0.0) / max(div.get("CF", 1.0), 1e-16)
                    )
                    S[i + 1, : i + 2] = np.maximum(0.0, S[i + 1, : i + 2] - pv)

            S[i + 1, : i + 2] = np.where(
                S[i + 1, : i + 2] <= 0.0, 1e-12, S[i + 1, : i + 2]
            )

        return S

    def _bs_step_forward(self, F_node, K, sigma, tau, r, option_type):
        F_node = max(F_node, 1e-16)
        K = max(K, 1e-16)
        vol_sqrt_tau = sigma * math.sqrt(max(tau, 1e-16))
        if vol_sqrt_tau <= 0:
            disc = math.exp(-r * tau)
            if option_type == "call":
                return disc * max(F_node - K, 0.0)
            else:
                return disc * max(K - F_node, 0.0)

        d1 = (math.log(F_node / K) + 0.5 * sigma * sigma * tau) / vol_sqrt_tau
        d2 = d1 - vol_sqrt_tau
        d1 = max(min(d1, 1e6), -1e6)
        d2 = max(min(d2, 1e6), -1e6)

        disc = math.exp(-r * tau)
        if option_type == "call":
            return disc * (norm.cdf(d1) * F_node - norm.cdf(d2) * K)
        else:
            return disc * (norm.cdf(-d2) * K - norm.cdf(-d1) * F_node)

    def _build_premium_tree_american(self, F, S):
        Nn = self.N_steps
        C = np.zeros_like(S)

        for j in range(Nn + 1):
            C[Nn, j] = max(self.epsilon * (S[Nn, j] - self.K), 0.0)

        tau_last = self.dt_list[-1]
        for j in range(Nn):
            F_node = F[Nn - 1, j]
            bs_price = self._bs_step_forward(
                F_node, self.K, self.sigma, tau_last, self.r, self.option_type
            )
            intrinsic = max(self.epsilon * (S[Nn - 1, j] - self.K), 0.0)
            C[Nn - 1, j] = max(bs_price, intrinsic)

        for i in range(Nn - 2, -1, -1):
            disc_i = self.discount_list[i]
            pu_i = self.pu_list[i]
            pd_i = self.pd_list[i]
            for j in range(i + 1):
                continuation = (pu_i * C[i + 1, j + 1] + pd_i * C[i + 1, j]) * disc_i
                intrinsic = max(self.epsilon * (S[i, j] - self.K), 0.0)

                A_candidate = 0.0
                for d in self._internal_divs:
                    if d["step_index"] == i + 1:
                        t_i = self.time_grid[i]
                        t_ex = d["t_rel"]
                        cap = math.exp(self.r * (t_ex - t_i))
                        S_tilde_before = S[i, j] * cap

                        if d["type"] == "prop":
                            S_tilde_after = S_tilde_before * (1.0 - d["q"])
                        else:
                            pv = d.get(
                                "pv", d.get("D", 0.0) / max(d.get("CF", 1.0), 1e-16)
                            )
                            S_tilde_after = S_tilde_before - pv
                            if S_tilde_after < 0.0:
                                S_tilde_after = 0.0

                        disc_back = math.exp(-self.r * (t_ex - t_i))
                        candidate_before = (
                            max(self.epsilon * (S_tilde_before - self.K), 0.0)
                            * disc_back
                        )
                        candidate_after = (
                            max(self.epsilon * (S_tilde_after - self.K), 0.0)
                            * disc_back
                        )
                        A_candidate = max(
                            A_candidate, candidate_before, candidate_after, 0.0
                        )

                C[i, j] = max(continuation, intrinsic, A_candidate, 0.0)

        return C

    def price_american_option(self):
        F = self._setup_forward_tree()
        S = self._setup_spot_tree(F)
        print("DEBUG: F0=", F[0, 0], "minF=", np.min(F), "maxF=", np.max(F))
        print("DEBUG: S0=", S[0, 0], "minS=", np.min(S), "maxS=", np.max(S))
        C = self._build_premium_tree_american(F, S)
        return float(C[0, 0])

    def price(self):
        return self.price_american_option()

    def get_greeks(self):
        base_S0 = self.S0
        base_T = self.T
        base_r = self.r
        base_sigma = self.sigma

        base_price = self.price_american_option()

        ds = 0.01 * max(base_S0, 1.0)
        self.S0 = base_S0 + ds
        price_up = self.price_american_option()
        self.S0 = base_S0 - ds
        price_down = self.price_american_option()
        delta = (price_up - price_down) / (2.0 * ds)
        self.S0 = base_S0

        gamma = (price_up - 2.0 * base_price + price_down) / (ds * ds)

        dt_theta = 1.0 / 365.0
        theta = 0.0
        if base_T - dt_theta > 0:
            self.T = base_T - dt_theta
            self._build_dt_schedule()
            self._rebuild_params_variable_steps()
            self._map_dividends_to_internal_steps()
            price_shorter = self.price_american_option()
            theta = base_price - price_shorter
        self.T = base_T
        self._build_dt_schedule()
        self._rebuild_params_variable_steps()
        self._map_dividends_to_internal_steps()

        dvol = 0.01
        self.sigma = base_sigma + dvol
        self._rebuild_params_variable_steps()
        price_vega_bumped = self.price_american_option()
        vega = price_vega_bumped - base_price
        self.sigma = base_sigma
        self._rebuild_params_variable_steps()

        dr = 0.01
        self.r = base_r + dr
        self._rebuild_params_variable_steps()
        price_rho_bumped = self.price_american_option()
        rho = price_rho_bumped - base_price
        self.r = base_r
        self._rebuild_params_variable_steps()

        return {
            "delta": float(delta),
            "gamma": float(gamma),
            "theta": float(theta),
            "vega": float(vega),
            "rho": float(rho),
        }

    def get_exercise_boundary(self):
        F = self._setup_forward_tree()
        S = self._setup_spot_tree(F)
        C = np.zeros_like(S)
        boundary = []

        Nn = self.N_steps
        for j in range(Nn + 1):
            C[Nn, j] = max(self.epsilon * (S[Nn, j] - self.K), 0.0)

        tau_last = self.dt_list[-1]
        for j in range(Nn):
            F_node = F[Nn - 1, j]
            bs_price = self._bs_step_forward(
                F_node, self.K, self.sigma, tau_last, self.r, self.option_type
            )
            intrinsic = max(self.epsilon * (S[Nn - 1, j] - self.K), 0.0)
            C[Nn - 1, j] = max(bs_price, intrinsic)

        for i in range(Nn - 2, -1, -1):
            disc_i = self.discount_list[i]
            pu_i = self.pu_list[i]
            pd_i = self.pd_list[i]
            critical = None
            for j in range(i + 1):
                continuation = (pu_i * C[i + 1, j + 1] + pd_i * C[i + 1, j]) * disc_i
                intrinsic = max(self.epsilon * (S[i, j] - self.K), 0.0)

                A_candidate = 0.0
                for d in self._internal_divs:
                    if d["step_index"] == i + 1:
                        t_i = self.time_grid[i]
                        t_ex = d["t_rel"]
                        cap = math.exp(self.r * (t_ex - t_i))
                        S_tilde_before = S[i, j] * cap

                        if d["type"] == "prop":
                            S_tilde_after = S_tilde_before * (1.0 - d["q"])
                        else:
                            pv = d.get(
                                "pv", d.get("D", 0.0) / max(d.get("CF", 1.0), 1e-16)
                            )
                            S_tilde_after = S_tilde_before - pv
                            if S_tilde_after < 0.0:
                                S_tilde_after = 0.0

                        disc_back = math.exp(-self.r * (t_ex - t_i))
                        candidate_before = (
                            max(self.epsilon * (S_tilde_before - self.K), 0.0)
                            * disc_back
                        )
                        candidate_after = (
                            max(self.epsilon * (S_tilde_after - self.K), 0.0)
                            * disc_back
                        )
                        A_candidate = max(
                            A_candidate, candidate_before, candidate_after, 0.0
                        )

                C[i, j] = max(continuation, intrinsic, A_candidate)
                tol = 1e-10
                if intrinsic > 0.0 and intrinsic >= continuation - tol:
                    if critical is None or S[i, j] > critical:
                        critical = S[i, j]

            boundary.append(critical)

        return list(reversed(boundary))
