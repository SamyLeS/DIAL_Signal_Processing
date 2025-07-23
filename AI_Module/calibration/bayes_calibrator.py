import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel

class BayesianCalibrator:
    def __init__(self, param_bounds, noise_level=1e-6):
        """
        param_bounds: dict of {param_name: (low, high)}
        """
        self.param_names = list(param_bounds.keys())
        self.bounds = np.array([param_bounds[n] for n in self.param_names])
        # kernel: constant × Matern + white noise
        kernel = ConstantKernel(1.0) * Matern(length_scale=np.ones(self.bounds.shape[0]), nu=2.5) \
                 + WhiteKernel(noise_level=noise_level)
        self.gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True)
        self.X_obs = np.zeros((0, len(self.param_names)))
        self.y_obs = np.zeros((0, ))
    
    def register_observation(self, params, error):
        """
        params: dict of {param_name: float}
        error: float (e.g. mean squared calibration error)
        """
        x = np.array([[params[n] for n in self.param_names]])
        self.X_obs = np.vstack([self.X_obs, x])
        self.y_obs = np.append(self.y_obs, error)
        self.gp.fit(self.X_obs, self.y_obs)
    
    def suggest_next(self, n_restarts=10, exploration=0.1):
        """
        Returns the next best parameter dict to try, by maximizing
        Expected Improvement.
        """
        from scipy.stats import norm
        def expected_improvement(x):
            mu, sigma = self.gp.predict(x[None, :], return_std=True)
            mu_sample_opt = np.min(self.y_obs)
            with np.errstate(divide='warn'):
                Z = (mu_sample_opt - mu) / sigma
                ei = (mu_sample_opt - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)
                return ei

        # random restarts
        best_x, best_ei = None, -np.inf
        for _ in range(n_restarts):
            x0 = np.random.uniform(self.bounds[:,0], self.bounds[:,1])
            # simple local improvement via random perturbations
            ei0 = expected_improvement(x0)
            if ei0 > best_ei:
                best_ei, best_x = ei0, x0
        return dict(zip(self.param_names, best_x))
