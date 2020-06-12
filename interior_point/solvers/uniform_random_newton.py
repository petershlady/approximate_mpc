from .newton_ip import NewtonIP
import numpy as np
from .newton_utils import take_uniform_random_step, get_residual_norm

class UniformRandomNewton(NewtonIP):
    def __init__(self, Q, R, S, Qf, q, qf, r, A, B, w, Fx, Fu, f, x0, n, m, T, r_dim=None):
        super().__init__(Q, R, S, Qf, q, qf, r, A, B, w, Fx, Fu, f, x0, n, m, T)

        if r_dim is None:
            self.r_dim = 2 ** int(np.log(n))
        else:
            self.r_dim = r_dim

    def solve(self, kappa=10, max_iterations=5, warm_start=True):
        # use x as it was set, return next u
        self.create_variants()

        self.z = np.zeros(self.sz_z)
        self.nu = np.zeros(self.sz_nu)

        kappa = 10
        tol = 1e-3

        while kappa > 1e-4:
            obj = get_residual_norm(
                self.H,
                self.z,
                self.g,
                kappa,
                self.P,
                self.C,
                self.nu,
                self.b,
                self.h
            )
            iters = 0
            import pdb; pdb.set_trace()
            while obj > tol and iters < max_iterations:
                self.z, self.nu = take_uniform_random_step(
                    self.z,
                    self.nu,
                    kappa,
                    self.H,
                    self.g,
                    self.C,
                    self.b,
                    self.P,
                    self.h,
                    self.r_dim,
                )

                obj = get_residual_norm(
                    self.H,
                    self.z,
                    self.g,
                    kappa,
                    self.P,
                    self.C,
                    self.nu,
                    self.b,
                    self.h
                )

                max_iterations += 1

            kappa /= 10