import deepxde as dde
import numpy as np
import torch

class GroundwaterPINN:
    def __init__(self, geom, time_domain, storage_coef, recharge_rate, pumping_rate):
        """
        Initialize the PINN for Groundwater flow.
        
        Args:
            geom: dde.geometry object (e.g., Rectangle or Polygon)
            time_domain: dde.geometry.TimeDomain object
            storage_coef (float): Storage coefficient (S)
            recharge_rate (float or func): Recharge rate (R)
            pumping_rate (float or func): Pumping rate (P)
        """
        self.geom = geom
        self.time_domain = time_domain
        self.geomtime = dde.geometry.GeometryXTime(geom, time_domain)
        self.S = storage_coef
        self.R = recharge_rate
        self.P = pumping_rate

    def pde(self, x, y):
        """
        The Boussinesq equation residual.
        x: Input tensor (x, y, t)
        y: Output tensor (h, T) -> We predict Head (h) and Transmissivity (T)
        """
        # Unpack inputs
        h = y[:, 0:1] # Head
        T = y[:, 1:2] # Transmissivity (spatially varying, assumed constant in time for this formulation)

        # Gradients
        h_t = dde.grad.jacobian(y, x, i=0, j=2)
        h_x = dde.grad.jacobian(y, x, i=0, j=0)
        h_y = dde.grad.jacobian(y, x, i=0, j=1)
        
        T_x = dde.grad.jacobian(y, x, i=1, j=0)
        T_y = dde.grad.jacobian(y, x, i=1, j=1)

        # Second derivatives for the diffusion term: div(T * grad(h))
        # Expansion: T * (h_xx + h_yy) + T_x * h_x + T_y * h_y
        h_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
        h_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)

        # Source/Sink terms
        # Note: self.R and self.P could be functions of x, y, t
        R_val = self.R if isinstance(self.R, (int, float)) else self.R(x)
        P_val = self.P if isinstance(self.P, (int, float)) else self.P(x)

        # PDE Residual: S * h_t - (T * Laplacian(h) + Grad(T) * Grad(h)) - R + P
        # We want this to be zero
        lhs = self.S * h_t
        rhs = (T * (h_xx + h_yy) + T_x * h_x + T_y * h_y) + R_val - P_val
        
        return lhs - rhs

    def build_model(self, obs_X, obs_h, iterations=10000, learning_rate=1e-3):
        """
        Constructs and compiles the DeepXDE model.
        
        Args:
            obs_X: Coordinates of observation points (N, 3) -> (x, y, t)
            obs_h: Observed head values (N, 1)
        """
        # Observation data boundary condition (PointSetBC)
        # We enforce that the model output 'h' matches observed data at specific points
        observe_bc = dde.icbc.PointSetBC(obs_X, obs_h, component=0)

        # Define the data object
        data = dde.data.TimePDE(
            self.geomtime,
            self.pde,
            [observe_bc], # Add boundary conditions here
            num_domain=2000, # Collocation points for PDE
            num_boundary=400,
            num_test=100
        )

        # Network architecture
        # Input: 3 (x, y, t)
        # Output: 2 (h, T)
        net = dde.nn.FNN([3] + [40] * 4 + [2], "tanh", "Glorot normal")
        
        # Enforce Transmissivity positivity (T > 0)
        # T = exp(output_2) or softplus
        def output_transform(x, y):
            h = y[:, 0:1]
            T = torch.exp(y[:, 1:2]) # Force T to be positive
            return torch.cat([h, T], dim=1)
            
        net.apply_output_transform(output_transform)

        model = dde.Model(data, net)
        model.compile("adam", lr=learning_rate)
        
        return model

if __name__ == "__main__":
    # Example usage / Test run
    geom = dde.geometry.Rectangle([-1, -1], [1, 1])
    timedomain = dde.geometry.TimeDomain(0, 1)
    
    # Placeholder parameters
    pinn = GroundwaterPINN(geom, timedomain, storage_coef=0.1, recharge_rate=0.01, pumping_rate=0.0)
    
    # Dummy data for scaffold testing
    obs_X = np.random.rand(10, 3) # 10 random points (x,y,t)
    obs_h = np.random.rand(10, 1) # Random head values
    
    print("Building model...")
    model = pinn.build_model(obs_X, obs_h)
    print("Model built successfully.")
    # model.train(iterations=100)

