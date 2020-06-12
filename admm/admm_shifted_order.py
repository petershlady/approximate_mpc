import numpy as np
from admm import ADMMSolver

class ADMMShiftedOrder(ADMMSolver):
    def __init__(self, ydims, Pis, Cs, D, lower_bounds, upper_bounds, vdims, T, rho=0.5):
        super().__init__(ydims, Pis, Cs, D, lower_bounds, upper_bounds, vdims, T, rho)

    def solve(self, problem):
        self.create_variants(problem)

        self.y = np.zeros(sum(self.ydims))
        self.zeta = np.zeros(sum(self.ydims))
        self.epsilon = np.zeros(sum(self.ydims))
        self.lambda_zeta = np.zeros(sum(self.ydims))
        self.lambda_epsilon = np.zeros(sum(self.ydims))

        tol = 1e-3

        for iteration in range(100):
            start_idx = 0
            end_idx = 0
            for i in range(len(self.ydims)):
                end_idx += self.ydims[i]

                self.lambda_zeta[start_idx:end_idx] += \
                    self.y[start_idx:end_idx] - self.zeta[start_idx:end_idx]
                self.lambda_epsilon[start_idx:end_idx] += \
                    self.y[start_idx:end_idx] - self.epsilon[start_idx:end_idx]

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

            if np.linalg.norm(self.y-self.zeta) < tol and np.linalg.norm(self.y-self.epsilon) < tol:
                break
