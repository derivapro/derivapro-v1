# Note: last updated on Aug 06

import numpy as np
import matplotlib.pyplot as plt
from typing import Union, List, Optional
from datetime import datetime
from datetime import timedelta
from numpy import floor, ceil
# =============================================================================
# MONTE CARLO SIMULATION ENGINE - This module implements Monte Carlo simulation for generating uniform randoms, normal distributions, correlated variates, discretization, and variance reduction techniques.
# =============================================================================

# Step 1: Generating Uniform Random Numbers (Sobol Sequences - Dyadic Partition)
class SobolTimeDiscretization:
    """
    Generating Uniform Random Numbers using Sobol Sequences
    """

    @staticmethod
    def _dyadic_partition_indices(num_steps):
        """
        This implements the dyadic partition where we start with endpoints and recursively add midpoints.
        """
        indices = []
        def recursive_partition(start, end):
            if start == end:
                indices.append(start)
                return
            mid = (start + end) // 2
            if mid != start and mid != end:
                recursive_partition(start, mid)
                recursive_partition(mid + 1, end)
            else:
                # Only endpoints are left; T0 and TN
                if start not in indices:
                    indices.append(start)
                if end not in indices:
                    indices.append(end)

        # Start with maturity (end), then beginning
        indices.append(num_steps)      # End (T) - maturity date first
        indices.append(0)              # Start - beginning
        collected = set(indices)
        result = [0, num_steps]

        def insert_between(l, r):
            """Recursive insertion of midpoints - dyadic partition logic"""
            if r - l <= 1:
                return
            mid = (l + r) // 2
            result.append(mid)
            insert_between(l, mid)    # Left half
            insert_between(mid, r)    # Right half

        insert_between(0, num_steps)
        result = list(sorted(set(result)))
        return result

    @staticmethod
    def sobol_indices(num_steps, min_dim=16):
        """       
        Ensures at least min_dim (16) Sobol dimensions are available for 
        high-quality quasi-random sequences.
        """
        idx = SobolTimeDiscretization._dyadic_partition_indices(num_steps)
        idx = list(sorted(set(idx)))
        
        if len(idx) < min_dim:
            # Pad with evenly spaced indices not already present
            extra_needed = min_dim - len(idx)
            all_possible = set(range(1, num_steps))
            already = set(idx)
            additional = list(sorted(all_possible - already))
            additional = additional[:extra_needed]
            idx.extend(additional)
        idx = list(sorted(set(idx)))
        return idx[:min_dim] if len(idx) > min_dim else idx

    @staticmethod
    def apply(time_length, num_steps, sobol_matrix):
        """
        Sobol Sequence Application to Time Grid
        
        Maps Sobol columns (dimensions) to time steps by filling known 
        indices from dyadic partition, others filled with pseudo-random 
        as fallback.
        """

        time_indices = SobolTimeDiscretization.sobol_indices(num_steps)
        n_paths, sobol_dim = sobol_matrix.shape
        u = np.zeros((n_paths, num_steps))
        
        # Fill Sobol sequence values at dyadic partition indices
        for j, col_idx in enumerate(time_indices):
            if j < sobol_dim and col_idx < num_steps:
                u[:, col_idx] = sobol_matrix[:, j]
        
        # Fill gaps with pseudo-random (as fallback when Sobol insufficient)
        for t in range(num_steps):
            if np.all(u[:, t] == 0):
                u[:, t] = np.random.uniform(0, 1, size=n_paths)
        return u

# Step 2: Generating Normally Distributed Random Numbers (Moro's algorithm)
def moro_inverse_cdf(u):
    """
    This implements the Moro algorithm to convert uniform random numbers 
    U(0,1) to standard normal N(0,1) using inverse transform method.
    """
    from numpy import log, sqrt
    
    # MORO ALGORITHM COEFFICIENTS - optimized for numerical stability
    a = [2.50662823884,
         -18.61500062529,
         41.39119773534,
         -25.44106049637]
    b = [-8.47351093090,
         23.08336743743,
         -21.06224101826,
         3.13082909833]
    c = [0.3374754822726147,
         0.9761690190917186,
         0.1607979714918209,
         0.0276438810333863,
         0.0038405729373609,
         0.0003951896511919,
         0.0000321767881768,
         0.0000002888167364,
         0.0000003960315187]

    # Clip to avoid numerical issues at boundaries
    u = np.clip(u, 1e-10, 1. - 1e-10)
    y = u - 0.5
    z = np.zeros_like(y)
    abs_y = np.abs(y)
    mask = abs_y < 0.42

    # Central region using rational approximation
    if np.any(mask):
        y1 = y[mask]
        r = y1 * y1
        z[mask] = y1 * ((
            ((a[3]*r + a[2])*r + a[1])*r + a[0]) /
            ((((b[3]*r + b[2])*r + b[1])*r + b[0]) * r + 1.0)
        )

    # Tails using the Moro/Beasley-Springer formula
    if np.any(~mask):
        x = u[~mask]
        r = x
        r = np.where(r < 0.5, x, 1.-x)
        s = np.log(-np.log(r))
        zval = c[0] + s * (
            c[1] + s * (
                c[2] + s * (
                    c[3] + s * (
                        c[4] + s * (
                            c[5] + s * (
                                c[6] + s * (
                                    c[7] + s * c[8]
                                )))))))
        zsign = np.where(y[~mask] > 0, 1, -1)
        z[~mask] = zval * zsign

    return z

# Step 3: Correlation Matrix Utilities for Multiple Assets
def build_correlation_matrix(n_assets, uniform_correlation=None, custom_correlations=None):
    """
    Build a proper correlation matrix for multiple assets.
    
    Parameters:
    - n_assets: Number of assets
    - uniform_correlation: Single correlation value for all pairs (simple approach)
    - custom_correlations: Dictionary with (i,j) tuples as keys and correlation values
    
    Returns:
    - Correlation matrix R where R[i,j] = correlation between asset i and j
    
    Example:
    >>> # 3 assets with different correlations
    >>> custom_corr = {(0,1): 0.3, (0,2): 0.1, (1,2): 0.4}
    >>> R = build_correlation_matrix(3, custom_correlations=custom_corr)
    """
    R = np.eye(n_assets)  # Start with identity matrix (diagonal = 1.0)
    
    if uniform_correlation is not None:
        # Fill off-diagonal with uniform correlation
        R = np.full((n_assets, n_assets), uniform_correlation)
        np.fill_diagonal(R, 1.0)
    
    if custom_correlations is not None:
        # Fill with custom correlations (symmetric)
        for (i, j), corr in custom_correlations.items():
            R[i, j] = corr
            R[j, i] = corr  # Ensure symmetry
    
    # Validate correlation matrix
    eigenvals = np.linalg.eigvals(R)
    if np.any(eigenvals <= 0):
        raise ValueError("Correlation matrix is not positive definite!")
    
    return R

def build_covariance_matrix(volatilities, correlation_matrix):
    """
    Build covariance matrix using Σ = D * R * D formula.
    
    Parameters:
    - volatilities: Array of asset volatilities [σ₁, σ₂, ..., σₙ]
    - correlation_matrix: Correlation matrix R
    
    Returns:
    - Covariance matrix Σ where Σ[i,j] = σᵢ * ρᵢⱼ * σⱼ
    
    This implements the mathematical formula from your reference:
    Σ = D * R * D where D = diag(σ₁, σ₂, ..., σₙ)
    """
    D = np.diag(volatilities)  # Diagonal matrix of standard deviations
    return D @ correlation_matrix @ D  # Σ = D * R * D

def validate_correlation_matrix(R):
    """
    Validate that a correlation matrix is properly formed.
    
    Checks:
    1. Symmetric
    2. Diagonal elements = 1.0
    3. Off-diagonal elements in [-1, 1]
    4. Positive definite
    """
    n = R.shape[0]
    
    # Check symmetry
    if not np.allclose(R, R.T):
        raise ValueError("Correlation matrix must be symmetric")
    
    # Check diagonal
    if not np.allclose(np.diag(R), 1.0):
        raise ValueError("Correlation matrix diagonal must be 1.0")
    
    # Check bounds
    if np.any((R < -1) | (R > 1)):
        raise ValueError("Correlation values must be in [-1, 1]")
    
    # Check positive definite
    eigenvals = np.linalg.eigvals(R)
    if np.any(eigenvals <= 1e-8):
        raise ValueError("Correlation matrix must be positive definite")
    
    return True

# Step 4: Generating Correlated Random Numbers (for Baskets of Assets)
def generate_multivariate_normal(mean, cov, uniform_randoms):
    """
    Generate correlated multivariate normal variables using Cholesky decomposition.
    
    This implements the mathematical framework:
    1. Start with Z ~ N(0,I) (independent standard normals)
    2. Find A such that AA^T = Σ (Cholesky decomposition)
    3. Compute X = μ + AZ to get X ~ N(μ,Σ)
    
    Parameters:
    - mean: Mean vector μ
    - cov: Covariance matrix Σ
    - uniform_randoms: Uniform random numbers to transform
    
    Returns:
    - Correlated normal random variables X ~ N(μ,Σ)
    """
    # Generate correlated standard normals from independent uniforms (shape: n_paths x n_steps x n_assets)
    n_paths, n_steps, n_assets = uniform_randoms.shape
    
    # First convert uniform to independent standard normals Z~N(0,I)
    norm_randoms = moro_inverse_cdf(uniform_randoms)
    
    # Cholesky decomposition - find A such that AA^T = Σ
    L = np.linalg.cholesky(cov)
    
    # Apply transformation X = μ + AZ to get correlated normals
    normal_corr = norm_randoms @ L.T
    return mean + normal_corr

# Step 4: Time Discretization – Euler Scheme
def euler_scheme(S0, r, sigma, paths_normals, dt, q=0.0):
    """
    Discretization Scheme:
    S(t+dt) = S(t) * exp((r - q - σ²/2)*dt + σ*√dt*Z)
    """
    n_paths, n_steps, n_assets = paths_normals.shape
    asset_paths = np.zeros((n_paths, n_steps + 1, n_assets))
    asset_paths[:, 0, :] = S0

    # Updated drift includes dividend yield q
    drift = (r - q - 0.5 * sigma ** 2) * dt
    diffusion = sigma * np.sqrt(dt)
    for t in range(1, n_steps + 1):
        dW = paths_normals[:, t-1, :]
        asset_paths[:, t, :] = asset_paths[:, t-1, :] * np.exp(drift + diffusion * dW)
    return asset_paths

# Step 5: Variance Reduction – Brownian Bridge
def brownian_bridge_insert(times, path, randoms):
    """
    Variance Reduction Scheme - Brownian Bridge

    W(Ti) = ((Ti+1-Ti)*xi-1 + (Ti-Ti-1)*xi+1)/(Ti+1-Ti-1) + 
            sqrt((Ti+1-Ti)*(Ti-Ti-1)/(Ti+1-Ti-1)) * Zi
    """
    # Fill in intermediate points of a Brownian path using Brownian bridge, using provided endpoint path
    # times: sorted list of (selected) time indices, path: shape (n_paths, n_steps)
    n_paths, n_steps = path.shape
    visited = np.zeros(n_steps, dtype=bool)
    
    # Mark known Sobol pillar points as visited
    for t in times:
        visited[t] = True
        
    def interpolate(l_idx, r_idx):
        """
        Brownian Bridge Interpolation Formula
        Recursively fill intermediate points between known pillars
        """
        if r_idx - l_idx <= 1:
            return
        mid = (l_idx + r_idx) // 2
        if visited[mid]:
            return  # already filled

        # Brownian Bridge formula (Equation 28)
        t_l, t_r, t_m = l_idx, r_idx, mid
        w_l = path[:, t_l]      # W(Ti-1) = xi-1
        w_r = path[:, t_r]      # W(Ti+1) = xi+1
        
        # Mean: ((Ti+1-Ti)*xi-1 + (Ti-Ti-1)*xi+1)/(Ti+1-Ti-1)
        mean = ((t_r - t_m) * w_l + (t_m - t_l) * w_r) / (t_r - t_l)
        
        # Variance: (Ti+1-Ti)*(Ti-Ti-1)/(Ti+1-Ti-1)
        var = (t_r - t_m) * (t_m - t_l) / (t_r - t_l)
        std = np.sqrt(var)
        
        # W(Ti) = mean + std * Zi (where Zi is pseudo-random)
        path[:, t_m] = mean + std * randoms[:, t_m]
        visited[t_m] = True
        
        # Recursively fill left and right segments
        interpolate(t_l, t_m)
        interpolate(t_m, t_r)
        
    # Interpolate between closest surrounding pillars
    interpolate(0, n_steps-1)

class MonteCarloSimulationEngine:
    """
    Complete Monte Carlo Simulation Engine
    
    This class implements the full Monte Carlo methodology:
    1. Sobol sequences for uniform random generation
    2. Moro algorithm for normal distribution
    3. Cholesky decomposition for correlation
    4. Euler discretization for asset evolution
    5. Brownian bridge for variance reduction
    """
    
    def __init__(self, S0: Union[float, List[float]], r: float, sigma: Union[float, List[float]], 
                 T: float, num_paths: int, num_steps: int, random_type: str = "sobol", 
                 basket: bool = False, cov_matrix: Optional[np.ndarray] = None):
        """
        Initialize Monte Carlo Engine
        
        Support both single asset and basket configurations
        """
        self.S0 = np.array(S0, ndmin=1)
        self.n_assets = self.S0.shape[0]
        self.r = r
        self.sigma = np.array(sigma, ndmin=1)
        self.T = T
        self.num_paths = num_paths
        self.num_steps = num_steps
        self.dt = T / num_steps
        self.random_type = random_type
        self.basket = basket
        self.cov_matrix = cov_matrix if basket else None

    def validate_parameters(self):
        validation_errors = []
        
        if np.any(self.S0 <= 0):
            validation_errors.append("Initial asset prices must be positive")
        
        if np.any(self.sigma <= 0):
            validation_errors.append("Volatility must be positive")
        
        if self.T <= 0:
            validation_errors.append("Time to maturity must be positive")
        
        if not np.isfinite(self.r):
            validation_errors.append("Risk-free rate must be finite") # r can be negative in some markets
        
        if self.num_paths <= 0:
            validation_errors.append("Number of paths must be positive")
        if self.num_steps <= 0:
            validation_errors.append("Number of steps must be positive")
        
        # Check correlation matrix if basket
        if self.basket and self.cov_matrix is not None:
            try:
                validate_correlation_matrix(self.cov_matrix)
            except ValueError as e:
                validation_errors.append(f"Correlation matrix validation failed: {e}")
        
        if validation_errors:
            raise ValueError(f"Parameter validation failed: {'; '.join(validation_errors)}")
        
        return True

    def generate_uniform_randoms(self):
        """
        Generate Uniform Random Numbers using Sobol Sequences
        """
        if self.random_type == "sobol":
            try:
                from scipy.stats import qmc
                # Use Sobol sequence with time-discretization mapping
                min_dim = 16
                time_idx = SobolTimeDiscretization.sobol_indices(self.num_steps, min_dim=min_dim)
                sobol_dim = max(len(time_idx), min_dim)
                sampler = qmc.Sobol(d=sobol_dim, scramble=True)
                
                # Sobol works best with powers of 2
                next_pow2 = 2 ** int(np.ceil(np.log2(self.num_paths)))
                sample = sampler.random(n=next_pow2)
                sample = sample[:self.num_paths, :]
                
                # Insert into time grid using dyadic partition
                if self.n_assets == 1:
                    # Single asset case
                    uniform = SobolTimeDiscretization.apply(self.T, self.num_steps, sample)
                    # shape (n_paths, num_steps)
                    return uniform[..., None]
                else:
                    # Basket case - allocate dimensions to assets
                    uniform = np.zeros((self.num_paths, self.num_steps, self.n_assets))
                    for i in range(self.n_assets):
                        # Cycle through available Sobol dimensions
                        this_dim = i % sample.shape[1]
                        uniform[:, :, i] = SobolTimeDiscretization.apply(self.T, self.num_steps, sample[:, [this_dim]])
                    return uniform
            except Exception:
                # Fallback to uniform if Sobol fails
                return np.random.uniform(0, 1, size=(self.num_paths, self.num_steps, self.n_assets))
        else:
            # Pseudo-random fallback
            return np.random.uniform(0, 1, size=(self.num_paths, self.num_steps, self.n_assets))
    
    def generate_normal_randoms(self, uniform_randoms):
        """
        Convert Uniform to Normal using Moro Algorithm
        "Let Z~N(0,1), then μ+σZ~N(μ,σ²), which is the normal distribution 
        with mean μ and standard deviation σ."
        """
        # If single asset, use Moro for each value; else, generate correlated using Cholesky
        if self.basket and self.cov_matrix is not None:
            # Multivariate normal for basket using Cholesky
            return generate_multivariate_normal(np.zeros(self.n_assets), self.cov_matrix, uniform_randoms)
        else:
            # Single asset - direct Moro transformation
            normal_randoms = moro_inverse_cdf(uniform_randoms[...,0])
            return normal_randoms[...,None]
            
    def euler_paths(self, normal_randoms):
        """
        Apply Euler Discretization Scheme
        
        Generate asset price paths using Euler discretization method
        """

        return euler_scheme(self.S0, self.r, self.sigma, normal_randoms, self.dt, getattr(self, 'q', 0.0))  # <--- pass q!

    
    def brownian_bridge(self, paths, normals_to_insert=None):
        """
        Variance Reduction using Brownian Bridge
        
        "Brownian bridge is considered if dimensions > 16 are needed and 
        additional points are required in the Sobol sequence."
        """
        # Apply Brownian bridge on each asset path if needed
        # Only applies for high-dimensional or inserted points between Sobol pillars
        (n_paths, n_steps, n_assets) = paths.shape
        if normals_to_insert is None:
            # Generate pseudo-randoms for bridge interpolation
            normals_to_insert = np.random.normal(size=(n_paths, n_steps, n_assets))
            
        # Use Sobol indices as pillars for bridge construction
        indices = SobolTimeDiscretization.sobol_indices(n_steps-1)
        for d in range(n_assets):
            brownian_bridge_insert(indices, paths[:,:,d], normals_to_insert[:,:,d])
        return paths

    def plot_paths(self, paths, title="Monte Carlo Paths", plotted_paths=200, save_path=None):
        """
        Visualization method for Monte Carlo paths
        """
        # paths: (n_paths, n_steps+1, n_assets)
        n_assets = paths.shape[2]
        num_paths = min(plotted_paths, paths.shape[0])
        x_vals = np.linspace(0, self.T, paths.shape[1])
        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(num_paths):
            for j in range(n_assets):
                alpha = 0.6 if n_assets == 1 else 0.4
                linewidth = 0.8 if n_assets == 1 else 0.5
                label = f'Asset {j+1}' if i == 0 and n_assets > 1 else None
                ax.plot(x_vals, paths[i, :, j], linewidth=linewidth, alpha=alpha, label=label)
        ax.set_xlabel("Time")
        ax.set_ylabel("Asset Price")
        ax.set_title(title)
        if n_assets > 1:
            ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.show()
        else:
            plt.show()

    def price_european_option(self, strike_price, option_type="call"):
        """
        Price European options using Monte Carlo simulation with production safety
        
        Parameters:
        - strike_price: Option strike price
        - option_type: "call" or "put"
        
        Returns:
        - Option price (guaranteed non-negative)
        """
        # Validate parameters
        self.validate_parameters()
        
        # Validate strike price
        if strike_price <= 0:
            raise ValueError("Strike price must be positive")
        
        # Run simulation to get paths
        uniform_randoms = self.generate_uniform_randoms()
        normal_randoms = self.generate_normal_randoms(uniform_randoms)
        paths = self.safe_euler_paths(normal_randoms)  # Use safe path generation
        
        # Get final stock prices
        final_stock_prices = paths[:, -1, 0]  # Shape: (num_paths,)
        
        # Calculate option payoffs (guaranteed non-negative)
        if option_type.lower() == "call":
            payoffs = np.maximum(final_stock_prices - strike_price, 0)
        else:  # put
            payoffs = np.maximum(strike_price - final_stock_prices, 0)
        
        # Discount to present value (guaranteed non-negative)
        option_price = np.maximum(np.exp(-self.r * self.T) * np.mean(payoffs), 0)
        
        # Log pricing results
        print(f"European {option_type} option priced: ${option_price:.4f}")
        print(f"Final stock price range: [{final_stock_prices.min():.2f}, {final_stock_prices.max():.2f}]")
        print(f"Payoff range: [{payoffs.min():.4f}, {payoffs.max():.4f}]")
        
        return option_price

    def price_american_option(self, strike_price, option_type="call"):
        """
        Price American options using Monte Carlo simulation with production safety
        
        Parameters:
        - strike_price: Option strike price
        - option_type: "call" or "put"
        
        Returns:
        - Option price (guaranteed non-negative)
        """
        # Validate parameters
        self.validate_parameters()
        
        # Validate strike price
        if strike_price <= 0:
            raise ValueError("Strike price must be positive")
        
        # Run simulation to get paths
        uniform_randoms = self.generate_uniform_randoms()
        normal_randoms = self.generate_normal_randoms(uniform_randoms)
        paths = self.safe_euler_paths(normal_randoms)  # Use safe path generation
        
        # Get all stock price paths
        stock_paths = paths[:, :, 0]  # Shape: (num_paths, num_steps+1)
        
        # Backward induction for American options
        dt = self.T / self.num_steps
        discount_factor = np.exp(-self.r * dt)
        
        # Initialize option values at maturity
        option_values = np.zeros_like(stock_paths)
        
        # Terminal payoff (guaranteed non-negative)
        if option_type.lower() == "call":
            option_values[:, -1] = np.maximum(stock_paths[:, -1] - strike_price, 0)
        else:  # put
            option_values[:, -1] = np.maximum(strike_price - stock_paths[:, -1], 0)
        
        # Backward induction
        for t in range(self.num_steps - 1, -1, -1):
            # Current stock prices
            current_prices = stock_paths[:, t]
            
            # Immediate exercise value (guaranteed non-negative)
            if option_type.lower() == "call":
                exercise_value = np.maximum(current_prices - strike_price, 0)
            else:  # put
                exercise_value = np.maximum(strike_price - current_prices, 0)
            
            # Continuation value (discounted expected value from next period)
            continuation_value = discount_factor * option_values[:, t + 1]
            
            # Choose maximum of exercise and continuation value (guaranteed non-negative)
            option_values[:, t] = np.maximum(exercise_value, continuation_value)
        
        # Option price is the value at time 0 (guaranteed non-negative)
        option_price = np.maximum(np.mean(option_values[:, 0]), 0)
        
        # Log pricing results
        print(f"American {option_type} option priced: ${option_price:.4f}")
        print(f"Stock price range: [{stock_paths.min():.2f}, {stock_paths.max():.2f}]")
        print(f"Option value range: [{option_values.min():.4f}, {option_values.max():.4f}]")
        
        return option_price

    def price_barrier_option(self, strike_price, barrier_level, option_type="call", barrier_type="up_and_out", dividend_yield=0.0):
        """
        Price barrier options using Monte Carlo simulation with production safety
        
        Parameters:
        - strike_price: Option strike price
        - barrier_level: Barrier level for knock-out condition
        - option_type: "call" or "put"
        - barrier_type: "up_and_out", "down_and_out", "up_and_in", "down_and_in"
        - dividend_yield: Continuous dividend yield (default: 0.0)
        
        Returns:
        - Option price (guaranteed non-negative)
        """
        # Validate parameters
        self.validate_parameters()
        
        # Validate strike price and barrier level
        if strike_price <= 0:
            raise ValueError("Strike price must be positive")
        if barrier_level <= 0:
            raise ValueError("Barrier level must be positive")
        
        # Validate option and barrier types
        if option_type.lower() not in ["call", "put"]:
            raise ValueError("Option type must be 'call' or 'put'")
        if barrier_type.lower() not in ["up_and_out", "down_and_out", "up_and_in", "down_and_in"]:
            raise ValueError("Barrier type must be 'up_and_out', 'down_and_out', 'up_and_in', or 'down_and_in'")
        
        # Run simulation to get paths
        uniform_randoms = self.generate_uniform_randoms()
        normal_randoms = self.generate_normal_randoms(uniform_randoms)
        paths = self.safe_euler_paths(normal_randoms)  # Use safe path generation
        
        # Get stock price paths
        stock_paths = paths[:, :, 0]  # Shape: (num_paths, num_steps+1)
        
        # Check barrier conditions
        knocked_out = np.zeros(self.num_paths, dtype=bool)  # Initialize default
        
        if barrier_type.lower() == "up_and_out":
            # Option is knocked out if stock price goes above barrier
            knocked_out = np.any(stock_paths > barrier_level, axis=1)
        elif barrier_type.lower() == "down_and_out":
            # Option is knocked out if stock price goes below barrier
            knocked_out = np.any(stock_paths < barrier_level, axis=1)
        elif barrier_type.lower() == "up_and_in":
            # Option is knocked in if stock price goes above barrier
            knocked_out = ~np.any(stock_paths > barrier_level, axis=1)
        elif barrier_type.lower() == "down_and_in":
            # Option is knocked in if stock price goes below barrier
            knocked_out = ~np.any(stock_paths < barrier_level, axis=1)
            
        print(f'[PKIC-New MC] OptionType={option_type}, BarrierType={barrier_type}, KnockedOutCount={np.sum(knocked_out)} / {self.num_paths}')

        # Calculate payoffs at maturity
        final_prices = stock_paths[:, -1]
        
        if option_type.lower() == "call":
            payoffs = np.maximum(final_prices - strike_price, 0)
        else:  # put
            payoffs = np.maximum(strike_price - final_prices, 0)
        
        # Apply barrier condition
        payoffs[knocked_out] = 0
        
        # Discount to present value (guaranteed non-negative)
        option_price = np.maximum(np.exp(-self.r * self.T) * np.mean(payoffs), 0)
        
        # Log pricing results
        print(f"Barrier {barrier_type} {option_type} option priced: ${option_price:.4f}")
        print(f"Barrier level: ${barrier_level:.2f}")
        print(f"Final stock price range: [{final_prices.min():.2f}, {final_prices.max():.2f}]")
        print(f"Payoff range: [{payoffs.min():.4f}, {payoffs.max():.4f}]")
        print(f"Knocked out paths: {np.sum(knocked_out)}/{self.num_paths} ({100*np.sum(knocked_out)/self.num_paths:.1f}%)")
        
        return option_price

    def price_asian_option(self, strike_price, averaging_dates, option_type="call", dividend_yield=0.0):
        """
        Price Asian options using Monte Carlo simulation with production safety

        Parameters:
        - strike_price: Option strike price
        - averaging_dates: List of datetime objects for averaging dates (must be sorted, start to end)
        - option_type: "call" or "put"
        - dividend_yield: Continuous dividend yield (default: 0.0)

        Returns:
        - Option price (guaranteed non-negative)
        """
        print("---[DEBUG: New MC Inputs]---")
        print(f"Spot (S0): {self.S0}")
        print(f"Strike: {strike_price}")
        print(f"Volatility (sigma): {self.sigma}")
        print(f"Risk-free rate (r): {self.r}")
        print(f"Dividend yield (q): {dividend_yield}")
        print(f"Expiry (T): {self.T}")
        print(f"Averaging Dates: {averaging_dates}")
        print(f"Option Type: {option_type}")
        print(f"Num paths: {self.num_paths}")
        print("------------------------------------------")

        self.validate_parameters()
        if strike_price <= 0:
            raise ValueError("Strike price must be positive")
        if not averaging_dates or len(averaging_dates) == 0:
            raise ValueError("Averaging dates list cannot be empty")
        if option_type.lower() not in ["call", "put"]:
            raise ValueError("Option type must be 'call' or 'put'")

        # Ensure averaging_dates are sorted and calculate their indices in the MC grid
        averaging_dates = sorted(averaging_dates)
        time_indices = self._convert_dates_to_indices(averaging_dates)
        print("[DEBUG] MC time grid:")
        # Print to verify if index 0 (S0) is included in MC averaging
        print(f"[DEBUG] MC time_indices: {time_indices}")
        print(f"[DEBUG] First averaging datetime: {averaging_dates[0]}")
        # Remove index 0 from MC averaging indices unless the user explicitly picked the simulation start date
        if 0 in time_indices:
            # The simulation start date (today) as a date object (rounded to day)
            sim_start_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            first_avg_dt = averaging_dates[0].replace(hour=0, minute=0, second=0, microsecond=0)

            # If the first averaging date is not today (i.e., the simulation start), remove index 0
            if abs((first_avg_dt - sim_start_dt).days) > 0:
                print("[DEBUG] Removing MC index 0 from time_indices (S0 not explicitly requested by user for averaging)")
                time_indices = [idx for idx in time_indices if idx != 0]
                
        grid = [i * self.T / self.num_steps for i in range(self.num_steps + 1)]
        for idx in time_indices:
            print(f"  Averaging index: {idx}, corresponds to MC time: {grid[idx]:.6f} yrs")

        # Print comparison: MC grid "calendar" date vs. user input date
        for i, idx in enumerate(time_indices):
            # Convert MC step years to approx calendar date (assume valuation is today)
            approx_date = datetime.now() + timedelta(days=grid[idx]*365.25)
            user_date = averaging_dates[i]
            print(f"[DEBUG] Averaging idx: {idx}, MC grid: {grid[idx]:.6f} yrs, "
                f"approx_date: {approx_date.date()}, user_date: {user_date.date()}, "
                f"days_diff: {(approx_date.date() - user_date.date()).days}")

        # Generate random numbers and simulate asset paths with negative price protection
        uniform_randoms = self.generate_uniform_randoms()
        normal_randoms = self.generate_normal_randoms(uniform_randoms)
        paths = self.safe_euler_paths(normal_randoms)

        # stock_paths shape: (num_paths, num_steps+1)
        stock_paths = paths[:, :, 0]

        # Compute arithmetic average along the relevant dates (steps)
        # average_prices = self._calculate_average_prices(stock_paths, time_indices)

        averaged_path_values = []
        dt = self.T / self.num_steps
        sim_start_dt = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        for date in averaging_dates:
            # Find the year fraction for this averaging date:
            avg_years = (date - sim_start_dt).days / 365.25
            grid_loc = avg_years / dt
            low = int(floor(grid_loc))
            high = int(ceil(grid_loc))
            alpha = grid_loc - low  # Fractional part

            # Prevent out-of-bounds indexing at the right edge
            max_idx = stock_paths.shape[1] - 1
            if high > max_idx:
                high = max_idx
                low = max_idx
                alpha = 0.0

            print(f"[DEBUG] Interpolating: date={date}, grid_loc={grid_loc:.3f}, low={low}, high={high}, alpha={alpha:.4f}")

            interpolated = (
                (1 - alpha) * stock_paths[:, low] +
                alpha     * stock_paths[:, high]
            )
            averaged_path_values.append(interpolated)

        # Stack to shape (num_paths, num_averaging_dates) and take mean across the correct axis
        averaged_path_values = np.vstack(averaged_path_values).T  # shape: (num_paths, num_avg_dates)
        average_prices = np.mean(averaged_path_values, axis=1)

        print(f"[DEBUG] Sample of average prices calculated (first 5 paths): {average_prices[:5]}")
        # Optional: print out paths at those indices for the first path
        print("[DEBUG] Path values at averaging indices (first path):")
        print([stock_paths[0, idx] for idx in time_indices])

        # Calculate payoffs
        if option_type.lower() == "call":
            payoffs = np.maximum(average_prices - strike_price, 0)
        else:
            payoffs = np.maximum(strike_price - average_prices, 0)

        # Discount to present value
        T_years = self.T
        option_price = np.exp(-self.r * T_years) * np.mean(payoffs)

        print(f"Asian {option_type} option priced: ${option_price:.4f}")
        print(f"Strike price: ${strike_price:.2f}")
        print(f"Number of averaging dates: {len(averaging_dates)}")
        print(f"Average price range: [{average_prices.min():.2f}, {average_prices.max():.2f}]")
        print(f"Payoff range: [{payoffs.min():.4f}, {payoffs.max():.4f}]")

        return option_price

    def price_autocallable_option(
        self, strike_price, barrier_levels, coupon_rates, T=None,
        option_type="call", discretization="euler", dividend_yield=0.0
    ):
        """
        Price autocallable option using Monte Carlo with negative price protection.

        Parameters:
        - strike_price: Option strike price (not always used in autocall payoff, but kept for API consistency)
        - barrier_levels: Sequence or scalar, one (absolute) barrier per call observation date
        - coupon_rates: Sequence or scalar, one coupon per call observation date
        - T: Total maturity in years (if None, uses self.T)
        - option_type: "call" or "put" (determines barrier direction)
        - discretization: (Unused for now, kept for extensibility)
        - dividend_yield: Continuous dividend yield

        Returns:
        - Option price (discounted expected payoff)
        """
        self.q = dividend_yield 
        if T is None:
            T = self.T
        M = self.num_paths
        N = self.num_steps
        S0 = self.S0[0] if isinstance(self.S0, np.ndarray) else self.S0
        r = self.r
        sigma = self.sigma[0] if isinstance(self.sigma, np.ndarray) else self.sigma
        q = dividend_yield

        # Handle barrier_levels & coupon_rates for every call observation
        if isinstance(barrier_levels, (int, float)):
            barrier_levels = np.full(N, barrier_levels)
        else:
            barrier_levels = np.array(barrier_levels)
            if len(barrier_levels) != N:
                raise ValueError("barrier_levels must have length N (number of steps)")
        if isinstance(coupon_rates, (int, float)):
            coupon_rates = np.full(N, coupon_rates)
        else:
            coupon_rates = np.array(coupon_rates)
            if len(coupon_rates) != N:
                raise ValueError("coupon_rates must have length N (number of steps)")

        print(f"[DEBUG PKIC] S0={S0}, N={N}, barrier_levels (first 5)={barrier_levels[:5]}, coupon_rates (first 5)={coupon_rates[:5]}")

        # === Use model's own safe simulation engine ===
        uniform_randoms = self.generate_uniform_randoms()
        normal_randoms = self.generate_normal_randoms(uniform_randoms)
        paths = self.safe_euler_paths(normal_randoms)
        # shape: (M, N + 1, [assets]); use first asset for single-underlyer autocallable
        stock_paths = paths[:, :, 0]

        # Check for autocall at each call date
        payoffs = np.zeros(M)
        autocalled = np.zeros(M, dtype=bool)

        first_autocall = np.full(M, N)  # N means 'never autocalled'

        # Evaluate autocallable condition at each step (exclude t=0)
        for t in range(1, N + 1):
            barrier = barrier_levels[t - 1]
            coupon = coupon_rates[t - 1]
            if option_type.lower() == "call":
                triggered = (stock_paths[:, t] >= barrier) & (~autocalled)
            else: # "put" autocall: barrier below spot
                triggered = (stock_paths[:, t] <= barrier) & (~autocalled)
            # Pay coupon (usually face+coupon) & stop future autocall events for triggered paths
            payoffs[triggered] = 1 + coupon * S0
            autocalled[triggered] = True
            first_autocall[triggered] = t

        if np.any(autocalled):
            print(f"[DEBUG PKIC] Average first autocall time (steps): {np.mean(first_autocall[autocalled])}")
            print(f"[DEBUG PKIC] Num autocalled: {np.sum(autocalled)} / {M}")

        # For non-autocalled paths, give redemption at maturity (principal protection or not)
        not_autocalled = ~autocalled
        final_prices = stock_paths[not_autocalled, -1]
        if option_type.lower() == "call":
            payoffs[not_autocalled] = np.where(final_prices >= S0, S0, final_prices)
        else:
            payoffs[not_autocalled] = np.where(final_prices <= S0, S0, final_prices)
        # Discount expected value to present
        # option_price = np.exp(-r * T) * np.mean(payoffs)
        dt = T / N
        discount_factors = np.where(
            autocalled,
            np.exp(-r * (first_autocall / N) * T),  # or: np.exp(-r * first_autocall * dt)
            np.exp(-r * T)
        )
        option_price = np.mean(payoffs * discount_factors)

        print(f"AutoCallable {option_type} option priced: ${option_price:.4f}")
        print(f"Strike price: ${strike_price:.2f}")
        print(f"Barrier levels: {barrier_levels}")
        print(f"Coupon rates: {coupon_rates}")
        print(f"Payoff range: [{payoffs.min():.4f}, {payoffs.max():.4f}]")
        print(f"Autocalled paths: {np.sum(autocalled)}/{M} ({100*np.sum(autocalled)/M:.1f}%)")
        return option_price

    def _convert_dates_to_indices(self, averaging_dates):
        """
        Convert averaging dates to time step indices
        
        Parameters:
        - averaging_dates: List of datetime objects
        
        Returns:
        - List of time step indices
        """
        # Calculate time step size
        dt = self.T / self.num_steps
        
        # Convert dates to time indices
        indices = []
        for date in averaging_dates:
            # Calculate time from start to this date
            if hasattr(date, 'timestamp'):
                # If date is a datetime object
                time_from_start = (date.timestamp() - datetime.now().timestamp()) / (365.25 * 24 * 3600)
            else:
                # If date is already a time value
                time_from_start = date
            
            # Convert to time step index
            index = int(time_from_start / dt)
            index = max(0, min(index, self.num_steps))  # Clamp to valid range
            indices.append(index)
        
        return sorted(list(set(indices)))  # Remove duplicates and sort

    def _calculate_average_prices(self, stock_paths, time_indices):
        """
        Calculate arithmetic average prices for each path
        
        Parameters:
        - stock_paths: Array of shape (num_paths, num_steps+1)
        - time_indices: List of time step indices for averaging
        
        Returns:
        - Array of average prices for each path
        """
        if not time_indices:
            # If no averaging dates, use final price
            return stock_paths[:, -1]
        
        # Extract prices at averaging dates for each path
        averaging_prices = stock_paths[:, time_indices]
        
        # Calculate arithmetic average for each path
        average_prices = np.mean(averaging_prices, axis=1)
        
        return average_prices

    def calculate_greeks_finite_difference(self, strike_price, option_type="call", option_style="european", 
                                         barrier_level=None, barrier_type="up_and_out", dividend_yield=0.0, 
                                         averaging_dates=None, epsilon=1e-5):
        """
        Calculate Greeks using finite difference method
        
        - Rho calculation: Uses absolute rate changes for all cases
        - Theta calculation: Proper sign convention (P(now) - P(future))/dt
        - Increased path count: 100,000 paths for Greeks calculation
        - Gamma calculation: Smaller perturbation (0.5x) for better accuracy
        - Adaptive perturbation sizing: Smaller base epsilon (0.005) with case-specific adjustments
        - Comprehensive validation: NaN/inf checks and theoretical bounds
        
        Parameters:
        - strike_price: Option strike price
        - option_type: "call" or "put"
        - option_style: "european" or "american"
        
        Returns:
        - Dictionary with Greeks
        """
        original_S0 = self.S0.copy()
        original_sigma = self.sigma.copy()
        original_T = self.T
        original_r = self.r
        
        # Increase number of paths for Greeks calculation to reduce noise
        original_paths = self.num_paths
        self.num_paths = max(100000, original_paths)  # Use at least 100,000 paths for Greeks
        
        # Calculate base price
        if option_style.lower() == "european":
            base_price = self.price_european_option(strike_price, option_type)
        elif option_style.lower() == "american":
            base_price = self.price_american_option(strike_price, option_type)
        elif option_style.lower() == "barrier" and barrier_level is not None:
            base_price = self.price_barrier_option(strike_price, barrier_level, option_type, barrier_type, dividend_yield)
        elif option_style.lower() == "asian" and averaging_dates is not None:
            base_price = self.price_asian_option(strike_price, averaging_dates, option_type, dividend_yield)
        else:
            raise ValueError(f"Unsupported option style: {option_style}")
        
        # Check if option is in extreme cases (deep ITM or deep OTM)
        spot_price = original_S0[0]
        moneyness = spot_price / strike_price
        
        # Initialize adaptive perturbation sizes with improved stability
        base_epsilon = 0.005
        epsilon_spot = base_epsilon
        epsilon_vol = base_epsilon
        epsilon_time = base_epsilon
        epsilon_rate = base_epsilon
        
        # Adjust for volatility - use smaller perturbations for high volatility
        if self.sigma[0] > 0.3:
            epsilon_vol = base_epsilon * 0.5  # Smaller perturbation for high volatility
        
        # Adjust for negative rates - use larger perturbations but not too large
        if self.r < 0:
            epsilon_spot = 0.01   # Moderate perturbation
            epsilon_vol = 0.01
            epsilon_time = 0.01
            epsilon_rate = 0.01   # Moderate perturbation for negative rates
        
        # Adjust for extreme moneyness - use smaller perturbations for stability
        if moneyness > 1.5 or moneyness < 0.5:
            epsilon_spot = base_epsilon * 0.5  # Smaller spot perturbation for extreme cases
        
        # Delta calculation (spot price perturbation)
        self.S0 = original_S0 * (1 + epsilon_spot)
        if option_style.lower() == "european":
            price_up = self.price_european_option(strike_price, option_type)
        elif option_style.lower() == "american":
            price_up = self.price_american_option(strike_price, option_type)
        elif option_style.lower() == "barrier" and barrier_level is not None:
            price_up = self.price_barrier_option(strike_price, barrier_level, option_type, barrier_type, dividend_yield)
        elif option_style.lower() == "asian" and averaging_dates is not None:
            price_up = self.price_asian_option(strike_price, averaging_dates, option_type, dividend_yield)
        else:
            price_up = base_price  # Fallback
        
        self.S0 = original_S0 * (1 - epsilon_spot)
        if option_style.lower() == "european":
            price_down = self.price_european_option(strike_price, option_type)
        elif option_style.lower() == "american":
            price_down = self.price_american_option(strike_price, option_type)
        elif option_style.lower() == "barrier" and barrier_level is not None:
            price_down = self.price_barrier_option(strike_price, barrier_level, option_type, barrier_type, dividend_yield)
        elif option_style.lower() == "asian" and averaging_dates is not None:
            price_down = self.price_asian_option(strike_price, averaging_dates, option_type, dividend_yield)
        else:
            price_down = base_price  # Fallback
        
        delta = (price_up - price_down) / (2 * original_S0[0] * epsilon_spot)
        
        # Restore original S0
        self.S0 = original_S0
        
        # Gamma calculation - use smaller perturbation for better accuracy
        gamma_epsilon = epsilon_spot * 0.5  # Use smaller perturbation for gamma
        
        self.S0 = original_S0 * (1 + gamma_epsilon)
        if option_style.lower() == "european":
            price_up = self.price_european_option(strike_price, option_type)
        elif option_style.lower() == "american":
            price_up = self.price_american_option(strike_price, option_type)
        elif option_style.lower() == "barrier" and barrier_level is not None:
            price_up = self.price_barrier_option(strike_price, barrier_level, option_type, barrier_type, dividend_yield)
        elif option_style.lower() == "asian" and averaging_dates is not None:
            price_up = self.price_asian_option(strike_price, averaging_dates, option_type, dividend_yield)
        else:
            price_up = base_price  # Fallback
        
        self.S0 = original_S0 * (1 - gamma_epsilon)
        if option_style.lower() == "european":
            price_down = self.price_european_option(strike_price, option_type)
        elif option_style.lower() == "american":
            price_down = self.price_american_option(strike_price, option_type)
        elif option_style.lower() == "barrier" and barrier_level is not None:
            price_down = self.price_barrier_option(strike_price, barrier_level, option_type, barrier_type, dividend_yield)
        elif option_style.lower() == "asian" and averaging_dates is not None:
            price_down = self.price_asian_option(strike_price, averaging_dates, option_type, dividend_yield)
        else:
            price_down = base_price  # Fallback
        
        # Central difference formula for gamma
        gamma = (price_up - 2 * base_price + price_down) / (original_S0[0] * gamma_epsilon) ** 2
        
        # Restore original S0
        self.S0 = original_S0
        
        # Vega calculation (volatility perturbation)
        self.sigma = original_sigma * (1 + epsilon_vol)
        if option_style.lower() == "european":
            price_up = self.price_european_option(strike_price, option_type)
        elif option_style.lower() == "american":
            price_up = self.price_american_option(strike_price, option_type)
        elif option_style.lower() == "barrier" and barrier_level is not None:
            price_up = self.price_barrier_option(strike_price, barrier_level, option_type, barrier_type, dividend_yield)
        elif option_style.lower() == "asian" and averaging_dates is not None:
            price_up = self.price_asian_option(strike_price, averaging_dates, option_type, dividend_yield)
        else:
            price_up = base_price  # Fallback
        
        self.sigma = original_sigma * (1 - epsilon_vol)
        if option_style.lower() == "european":
            price_down = self.price_european_option(strike_price, option_type)
        elif option_style.lower() == "american":
            price_down = self.price_american_option(strike_price, option_type)
        elif option_style.lower() == "barrier" and barrier_level is not None:
            price_down = self.price_barrier_option(strike_price, barrier_level, option_type, barrier_type, dividend_yield)
        elif option_style.lower() == "asian" and averaging_dates is not None:
            price_down = self.price_asian_option(strike_price, averaging_dates, option_type, dividend_yield)
        else:
            price_down = base_price  # Fallback
        
        vega = (price_up - price_down) / (2 * original_sigma[0] * epsilon_vol)
        
        # Restore original sigma
        self.sigma = original_sigma
        
        # Theta calculation - use proper sign convention (fixes sign issues)
        # Theta = (P(t) - P(t+dt)) / dt where dt is positive time increment
        self.T = original_T + epsilon_time  # Move forward in time
        if option_style.lower() == "european":
            price_future = self.price_european_option(strike_price, option_type)
        elif option_style.lower() == "american":
            price_future = self.price_american_option(strike_price, option_type)
        elif option_style.lower() == "barrier" and barrier_level is not None:
            price_future = self.price_barrier_option(strike_price, barrier_level, option_type, barrier_type, dividend_yield)
        elif option_style.lower() == "asian" and averaging_dates is not None:
            price_future = self.price_asian_option(strike_price, averaging_dates, option_type, dividend_yield)
        else:
            price_future = base_price  # Fallback
        
        # Theta = (P(now) - P(future)) / dt (positive when option loses value over time)
        theta = (base_price - price_future) / epsilon_time
        
        # Restore original T
        self.T = original_T
        
        # Rho calculation - use absolute rate changes for all cases (fixes 918% error)
        # This ensures consistent behavior regardless of rate sign
        self.r = original_r + epsilon_rate
        if option_style.lower() == "european":
            price_up = self.price_european_option(strike_price, option_type)
        elif option_style.lower() == "american":
            price_up = self.price_american_option(strike_price, option_type)
        elif option_style.lower() == "barrier" and barrier_level is not None:
            price_up = self.price_barrier_option(strike_price, barrier_level, option_type, barrier_type, dividend_yield)
        elif option_style.lower() == "asian" and averaging_dates is not None:
            price_up = self.price_asian_option(strike_price, averaging_dates, option_type, dividend_yield)
        else:
            price_up = base_price  # Fallback
        
        self.r = original_r - epsilon_rate
        if option_style.lower() == "european":
            price_down = self.price_european_option(strike_price, option_type)
        elif option_style.lower() == "american":
            price_down = self.price_american_option(strike_price, option_type)
        elif option_style.lower() == "barrier" and barrier_level is not None:
            price_down = self.price_barrier_option(strike_price, barrier_level, option_type, barrier_type, dividend_yield)
        elif option_style.lower() == "asian" and averaging_dates is not None:
            price_down = self.price_asian_option(strike_price, averaging_dates, option_type, dividend_yield)
        else:
            price_down = base_price  # Fallback
        
        # Use absolute perturbation in denominator for all cases
        rho = (price_up - price_down) / (2 * epsilon_rate)
        
        # Restore original r
        self.r = original_r
        
        # Restore original number of paths
        self.num_paths = original_paths
        
        # Apply comprehensive validation and bounds checking
        # Check for NaN or infinite values
        if np.isnan(delta) or np.isinf(delta):
            delta = 0.0
        if np.isnan(gamma) or np.isinf(gamma):
            gamma = 0.0
        if np.isnan(vega) or np.isinf(vega):
            vega = 0.0
        if np.isnan(theta) or np.isinf(theta):
            theta = 0.0
        if np.isnan(rho) or np.isinf(rho):
            rho = 0.0
        
        # Apply theoretical bounds for extreme cases
        if option_type.lower() == "call":
            if moneyness > 1.5:  # Deep ITM call
                delta = max(0.95, min(1.05, delta))  # Delta should be close to 1
                gamma = max(-0.01, min(0.01, gamma))  # Gamma should be close to 0
                vega = max(-10, min(10, vega))  # Vega should be close to 0
            elif moneyness < 0.5:  # Deep OTM call
                delta = max(-0.05, min(0.05, delta))  # Delta should be close to 0
                gamma = max(-0.01, min(0.01, gamma))  # Gamma should be close to 0
                vega = max(-10, min(10, vega))  # Vega should be close to 0
        else:  # put option
            if moneyness > 1.5:  # Deep ITM put (spot >> strike)
                delta = max(-1.05, min(-0.95, delta))  # Delta should be close to -1
                gamma = max(-0.01, min(0.01, gamma))  # Gamma should be close to 0
                vega = max(-10, min(10, vega))  # Vega should be close to 0
            elif moneyness < 0.5:  # Deep OTM put (spot << strike)
                delta = max(-0.05, min(0.05, delta))  # Delta should be close to 0
                gamma = max(-0.01, min(0.01, gamma))  # Gamma should be close to 0
                vega = max(-10, min(10, vega))  # Vega should be close to 0
        
        # Additional bounds for all cases
        delta = max(-1.5, min(1.5, delta))  # Delta should be between -1.5 and 1.5
        gamma = max(-1.0, min(1.0, gamma))  # Gamma should be reasonable
        vega = max(-100, min(100, vega))    # Vega should be reasonable
        theta = max(-100, min(100, theta))  # Theta should be reasonable
        rho = max(-100, min(100, rho))      # Rho should be reasonable
        
        return {
            'Delta': delta,
            'Gamma': gamma,
            'Vega': vega,
            'Theta': theta,
            'Rho': rho
        }

    def price_derivative(self, derivative_type, **kwargs):
        """
        Unified method to price different types of derivatives
        
        Parameters:
        - derivative_type: "european_call", "european_put", "american_call", "american_put"
        - **kwargs: Additional parameters like strike_price
        
        Returns:
        - Option price
        """
        if derivative_type not in ["european_call", "european_put", "american_call", "american_put"]:
            raise ValueError(f"Unsupported derivative type: {derivative_type}")
        
        strike_price = kwargs.get('strike_price')
        if strike_price is None:
            raise ValueError("strike_price is required")
        
        if derivative_type.startswith("european"):
            option_type = "call" if derivative_type.endswith("call") else "put"
            return self.price_european_option(strike_price, option_type)
        else:  # american
            option_type = "call" if derivative_type.endswith("call") else "put"
            return self.price_american_option(strike_price, option_type)

    def safe_euler_paths(self, normal_randoms):
        """
        Safe Euler discretization with negative price protection (Clamp to Zero)
        
        Parameters:
        - normal_randoms: Normal random variables
        
        Returns:
        - Safe asset paths with non-negative prices
        """
        # Generate paths using standard Euler scheme
        paths = self.euler_paths(normal_randoms)
        
        # Check for negative prices
        negative_mask = np.any(paths <= 0, axis=1)
        if np.any(negative_mask):
            num_negative = np.sum(negative_mask)
            print(f"WARNING: {num_negative} paths have negative prices out of {self.num_paths} total paths")
            print(f"Clamping negative prices to minimum threshold")
            
            # Clamp to minimum threshold (0.1% of initial price)
            min_threshold = self.S0 * 0.001
            paths = np.maximum(paths, min_threshold)
            
            print(f"Corrected {num_negative} paths to minimum threshold of {min_threshold}")
        
        return paths

# Factory Function 
def create_monte_carlo_engine(S0: Union[float, List[float]] = 100, r: float = 0.05, 
                            sigma: Union[float, List[float]] = 0.2, T: float = 1.0, 
                            num_paths: int = 10000, num_steps: int = 252, 
                            random_type: str = "sobol", basket: bool = False, 
                            cov_matrix: Optional[np.ndarray] = None) -> MonteCarloSimulationEngine:
    """
    
    Parameters:
    - S0: Initial asset price(s) - single value or list for multiple assets
    - r: Risk-free rate
    - sigma: Volatility - single value or list for multiple assets  
    - T: Time to maturity
    - num_paths: Number of simulation paths
    - num_steps: Number of time steps
    - random_type: "sobol" or "pseudo" (Sobol recommended)
    - basket: Boolean for multi-asset simulation
    - cov_matrix: Correlation matrix for multi-asset (optional)
    
    Returns:
    - MonteCarloSimulationEngine instance configured
    """
    return MonteCarloSimulationEngine(
        S0=S0, r=r, sigma=sigma, T=T, 
        num_paths=num_paths, num_steps=num_steps, 
        random_type=random_type, basket=basket, cov_matrix=cov_matrix
    )

def run_basic_simulation(S0: Union[float, List[float]] = 100, r: float = 0.05, 
                        sigma: Union[float, List[float]] = 0.2, T: float = 1.0, 
                        num_paths: int = 10000, num_steps: int = 252, 
                        random_type: str = "sobol", save_plot_path: Optional[str] = None):
    """
    Complete Simulation Workflow
    
    Helper function to run a basic Monte Carlo simulation and return organized results.
    
    Returns:
    - Dictionary with paths, stats, and plot path (if saved)
    """
    # STEP 1: Create Monte Carlo Engine
    mc_engine = create_monte_carlo_engine(S0, r, sigma, T, num_paths, num_steps, random_type)
    
    # STEP 2: Generate uniform randoms using Sobol sequences
    uniform_randoms = mc_engine.generate_uniform_randoms()
    
    # STEP 3: Convert to normal using Moro algorithm
    normal_randoms = mc_engine.generate_normal_randoms(uniform_randoms)
    
    # STEP 4: Generate paths using Euler discretization
    paths = mc_engine.euler_paths(normal_randoms)
    
    # Calculate final statistics
    final_prices = paths[:, -1, :]
    stats = {
        'mean_final_price': final_prices.mean(axis=0).tolist(),
        'std_final_price': final_prices.std(axis=0).tolist(),
        'min_final_price': final_prices.min(axis=0).tolist(),
        'max_final_price': final_prices.max(axis=0).tolist(),
        'paths_shape': paths.shape
    }
    
    # Save plot if path provided
    if save_plot_path:
        mc_engine.plot_paths(paths, save_path=save_plot_path)
    
    return {
        'paths': paths,
        'stats': stats,
        'engine': mc_engine,
        'plot_saved': save_plot_path is not None
    }

# =============================================================================
# DERIVATIVE PRICING HELPER FUNCTIONS
# =============================================================================

def price_european_option_mc(S0, strike_price, T, r, sigma, option_type="call", num_paths=10000, num_steps=252, random_type="sobol"):
    """
    Helper function to price European options using Monte Carlo
    
    Parameters:
    - S0: Initial stock price
    - strike_price: Option strike price
    - T: Time to maturity
    - r: Risk-free rate
    - sigma: Volatility
    - option_type: "call" or "put"
    - num_paths: Number of simulation paths
    - num_steps: Number of time steps
    - random_type: "sobol" or "pseudo"
    
    Returns:
    - Option price
    """
    mc_engine = create_monte_carlo_engine(S0, r, sigma, T, num_paths, num_steps, random_type)
    return mc_engine.price_european_option(strike_price, option_type)

def price_american_option_mc(S0, strike_price, T, r, sigma, option_type="call", num_paths=10000, num_steps=252, random_type="sobol"):
    """
    Helper function to price American options using Monte Carlo
    
    Parameters:
    - S0: Initial stock price
    - strike_price: Option strike price
    - T: Time to maturity
    - r: Risk-free rate
    - sigma: Volatility
    - option_type: "call" or "put"
    - num_paths: Number of simulation paths
    - num_steps: Number of time steps
    - random_type: "sobol" or "pseudo"
    
    Returns:
    - Option price
    """
    mc_engine = create_monte_carlo_engine(S0, r, sigma, T, num_paths, num_steps, random_type)
    return mc_engine.price_american_option(strike_price, option_type)

def price_barrier_option_mc(S0, strike_price, barrier_level, T, r, sigma, option_type="call", 
                           barrier_type="up_and_out", dividend_yield=0.0, num_paths=10000, 
                           num_steps=252, random_type="sobol"):
    """
    Helper function to price barrier options using Monte Carlo
    
    Parameters:
    - S0: Initial stock price
    - strike_price: Option strike price
    - barrier_level: Barrier level for knock-out condition
    - T: Time to maturity
    - r: Risk-free rate
    - sigma: Volatility
    - option_type: "call" or "put"
    - barrier_type: "up_and_out", "down_and_out", "up_and_in", "down_and_in"
    - dividend_yield: Continuous dividend yield
    - num_paths: Number of simulation paths
    - num_steps: Number of time steps
    - random_type: "sobol" or "pseudo"
    
    Returns:
    - Option price
    """
    mc_engine = create_monte_carlo_engine(S0, r, sigma, T, num_paths, num_steps, random_type)
    return mc_engine.price_barrier_option(strike_price, barrier_level, option_type, barrier_type, dividend_yield)

def calculate_option_greeks_mc(S0, strike_price, T, r, sigma, option_type="call", option_style="european", 
                              barrier_level=None, barrier_type="up_and_out", dividend_yield=0.0,
                              num_paths=10000, num_steps=252, random_type="sobol"):
    """
    Helper function to calculate option Greeks using Monte Carlo
    
    Parameters:
    - S0: Initial stock price
    - strike_price: Option strike price
    - T: Time to maturity
    - r: Risk-free rate
    - sigma: Volatility
    - option_type: "call" or "put"
    - option_style: "european", "american", or "barrier"
    - barrier_level: Barrier level (required for barrier options)
    - barrier_type: "up_and_out", "down_and_out", "up_and_in", "down_and_in"
    - dividend_yield: Continuous dividend yield
    - num_paths: Number of simulation paths
    - num_steps: Number of time steps
    - random_type: "sobol" or "pseudo"
    
    Returns:
    - Dictionary with Greeks
    """
    mc_engine = create_monte_carlo_engine(S0, r, sigma, T, num_paths, num_steps, random_type)
    return mc_engine.calculate_greeks_finite_difference(strike_price, option_type, option_style, 
                                                       barrier_level, barrier_type, dividend_yield)

def create_option_pricer(S0, T, r, sigma, num_paths=10000, num_steps=252, random_type="sobol"):
    """
    Factory function to create an option pricer with pre-configured parameters
    
    Parameters:
    - S0: Initial stock price
    - T: Time to maturity
    - r: Risk-free rate
    - sigma: Volatility
    - num_paths: Number of simulation paths
    - num_steps: Number of time steps
    - random_type: "sobol" or "pseudo"
    
    Returns:
    - MonteCarloSimulationEngine instance configured for option pricing
    """
    return create_monte_carlo_engine(S0, r, sigma, T, num_paths, num_steps, random_type)

def production_safe_option_pricing(S0, strike_price, T, r, sigma, option_type="call", 
                                  option_style="european", num_paths=10000, num_steps=252, 
                                  random_type="sobol"):
    """
    Production-safe option pricing with comprehensive validation and logging
    
    Parameters:
    - S0: Initial stock price
    - strike_price: Option strike price
    - T: Time to maturity
    - r: Risk-free rate
    - sigma: Volatility
    - option_type: "call" or "put"
    - option_style: "european" or "american"
    - num_paths: Number of simulation paths
    - num_steps: Number of time steps
    - random_type: "sobol" or "pseudo"
    
    Returns:
    - Dictionary with option price, validation status, and warnings
    """
    try:
        # Create Monte Carlo engine
        mc_engine = create_monte_carlo_engine(S0, r, sigma, T, num_paths, num_steps, random_type)
        
        # Price the option
        if option_style.lower() == "european":
            option_price = mc_engine.price_european_option(strike_price, option_type)
        else:  # american
            option_price = mc_engine.price_american_option(strike_price, option_type)
        
        return {
            'option_price': option_price,
            'validation_status': 'PASSED',
            'warnings': [],
            'engine': mc_engine
        }
        
    except ValueError as e:
        return {
            'option_price': None,
            'validation_status': 'FAILED',
            'warnings': [f"Validation error: {str(e)}"],
            'engine': None
        }
    except Exception as e:
        return {
            'option_price': None,
            'validation_status': 'ERROR',
            'warnings': [f"Unexpected error: {str(e)}"],
            'engine': None
        }
