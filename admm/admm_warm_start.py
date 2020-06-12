import numpy as np
from admm_shifted_order import ADMMShiftedOrder

class ADMMWarmStart(ADMMShiftedOrder):
    def __init__(self, ydims, Pis, Cs, D, lower_bounds, upper_bounds, vdims, T, yi_dims, rho=0.5):
        super().__init__(ydims, Pis, Cs, D, lower_bounds, upper_bounds, vdims, T, rho)
        self.yi_dims = yi_dims
    
    def advance_variables(self):
        start_idx = 0
        end_idx = 0
        for i in range(len(self.ydims)):
            end_idx += self.ydims[i]

            self.y[start_idx:end_idx-self.yi_dims[i]] = self.y[start_idx+self.yi_dims[i]:end_idx]
            self.zeta[start_idx:end_idx-self.yi_dims[i]] = self.zeta[start_idx+self.yi_dims[i]:end_idx]
            self.epsilon[start_idx:end_idx-self.yi_dims[i]] = self.epsilon[start_idx+self.yi_dims[i]:end_idx]
            self.lambda_zeta[start_idx:end_idx-self.yi_dims[i]] = self.lambda_zeta[start_idx+self.yi_dims[i]:end_idx]
            self.lambda_epsilon[start_idx:end_idx-self.yi_dims[i]] = self.lambda_epsilon[start_idx+self.yi_dims[i]:end_idx]

            start_idx += self.ydims[i]

    def solve(self, problem):
        self.create_variants(problem)

        self.advance_variables()

        tol = 1e-3

        for iteration in range(100):
            start_idx = 0
            end_idx = 0
            for i in range(len(self.ydims)):
                end_idx += self.ydims[i]

                self.y[start_idx:end_idx] = \
                    self.Mis[i].dot( 
                        self.zeta[start_idx:end_idx] + \
                        self.epsilon[start_idx:end_idx] - \
                        self.lambda_zeta[start_idx:end_idx] - \
                        self.lambda_epsilon[start_idx:end_idx]
                    ) + self.Nis[i].dot(self.cs[i])

                self.zeta[start_idx:end_idx] = np.maximum(
                    np.minimum(
                        self.y[start_idx:end_idx] + self.lambda_zeta[start_idx:end_idx],
                        self.upper_bounds[i]
                    ),
                    self.lower_bounds[i]
                )

                start_idx += self.ydims[i]

            self.epsilon = self.D_mat.dot(
                self.y + self.lambda_epsilon
            ) + self.E_mat.dot(self.d)

            start_idx = 0
            end_idx = 0
            for i in range(len(self.ydims)):
                end_idx += self.ydims[i]

                self.lambda_zeta[start_idx:end_idx] += \
                    self.y[start_idx:end_idx] - self.zeta[start_idx:end_idx]
                self.lambda_epsilon[start_idx:end_idx] += \
                    self.y[start_idx:end_idx] - self.epsilon[start_idx:end_idx]

                start_idx += self.ydims[i]

            if np.linalg.norm(self.y-self.zeta) < tol and np.linalg.norm(self.y-self.epsilon) < tol:
                break