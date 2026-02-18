"""
Equity Process Models for CVA Simulation
Implements geometric Brownian motion for multiple assets with correlation
"""

import numpy as np
from scipy.linalg import cholesky


class EquityProcessSimulator:
    """
    Simulates correlated equity processes using geometric Brownian motion.
    
    Implements:
    S(t + Δ) = S(t) * exp[(r - q - 0.5*σ²)*Δ + σ*Z*√Δ]
    
    Where Z_SX5E and Z_AEX are correlated standard normals with correlation ρ
    """
    
    def __init__(self, config, portfolio):
        """
        Initialize the simulator with parameters from config and portfolio.
        Args:
            config (dict): Configuration dictionary with all required parameters.
            portfolio (DataFrame): Portfolio data with initial spot prices (S0)
        """
        self.r = config['risk_free_rate']
        self.q = config['dividend_yield']
        self.dt = config['time_step_years']
        self.rho = config['correlation']
        self.sigma_sx5e = config['volatility_sx5e']
        self.sigma_aex = config['volatility_aex']
        self.T = config['counterparty_maturity']
        self.num_simulations = config.get('num_simulations', 1000)
        self.seed = config.get('random_seed', None)
        # Initial spot prices from portfolio
        self.S0_sx5e = portfolio[portfolio['Underlying'] == 'SX5E']['S0'].iloc[0]
        self.S0_aex = portfolio[portfolio['Underlying'] == 'AEX']['S0'].iloc[0]
        # Number of timesteps (robust rounding)
        self.num_steps = int(round(self.T / self.dt))
        # Precompute drift and sqrt(dt)
        self.drift_sx5e = (self.r - self.q - 0.5 * self.sigma_sx5e**2) * self.dt
        self.drift_aex = (self.r - self.q - 0.5 * self.sigma_aex**2) * self.dt
        self.sqrt_dt = np.sqrt(self.dt)
        # Precompute Cholesky decomposition
        corr_matrix = np.array([
            [1.0, self.rho],
            [self.rho, 1.0]
        ])
        self.cholesky = cholesky(corr_matrix, lower=True)
        
    def simulate_paths(self):
        """
        Simulate correlated equity paths.
        Returns:
            dict: Dictionary containing:
                - 'times': Time grid (num_steps,)
                - 'sx5e': SX5E paths (num_simulations, num_steps + 1)
                - 'aex': AEX paths (num_simulations, num_steps + 1)
        """
        if self.seed is not None:
            np.random.seed(self.seed)
        num_simulations = self.num_simulations
        times = np.linspace(0, self.T, self.num_steps + 1)
        # Initialize paths
        sx5e_paths = np.zeros((num_simulations, self.num_steps + 1))
        aex_paths = np.zeros((num_simulations, self.num_steps + 1))
        sx5e_paths[:, 0] = self.S0_sx5e
        aex_paths[:, 0] = self.S0_aex
        # Simulation loop
        for t in range(1, self.num_steps + 1):
            # Generate two independent standard normals
            Z = np.random.standard_normal((num_simulations, 2))
            # Apply Cholesky decomposition to get correlated normals
            Z_corr = Z @ self.cholesky.T
            Z_sx5e = Z_corr[:, 0]
            Z_aex = Z_corr[:, 1]
            # Diffusion term
            diffusion_sx5e = self.sigma_sx5e * Z_sx5e * self.sqrt_dt
            diffusion_aex = self.sigma_aex * Z_aex * self.sqrt_dt
            # Update paths
            sx5e_paths[:, t] = sx5e_paths[:, t-1] * np.exp(self.drift_sx5e + diffusion_sx5e)
            aex_paths[:, t] = aex_paths[:, t-1] * np.exp(self.drift_aex + diffusion_aex)
        return {
            'times': times,
            'sx5e': sx5e_paths,
            'aex': aex_paths,
        }
