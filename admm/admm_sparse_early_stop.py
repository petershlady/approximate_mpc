from admm_warm_start import ADMMWarmStart
from scipy.sparse import csc_matrix
import numpy as np

class ADMMSparseEarly(ADMMWarmStart):
    def __init__(self, ydims, Pis, Cs, D, lower_bounds, upper_bounds, vdims, T, yi_dims, rho=0.5):
        super().__init__(ydims, Pis, Cs, D, lower_bounds, upper_bounds, vdims, T, yi_dims, rho)

        self.D_mat = csc_matrix(self.D_mat)
        self.E_mat = csc_matrix(self.E_mat)

    def solve(self, problem, timing=False):
        self.create_variants(problem)

        self.advance_variables()

        tol = 0.5

        if timing:
            total_time = 0
            import time

        for iteration in range(100):
            start_idx = 0
            end_idx = 0

            if timing:
                times = []

            for i in range(len(self.ydims)):
                end_idx += self.ydims[i]

                if timing:
                    st = time.time()

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

                if timing:
                    times.append( time.time() - st )

                start_idx += self.ydims[i]

            if timing:
                st = time.time()

            self.epsilon = self.D_mat.dot(
                self.y + self.lambda_epsilon
            ) + self.E_mat.dot(self.d)

            if timing:
                total_time += time.time() - st
                total_time += max(times)
            
            if np.linalg.norm(self.y-self.zeta) < tol and np.linalg.norm(self.y-self.epsilon) < tol:
                break

        if timing:
            return total_time
