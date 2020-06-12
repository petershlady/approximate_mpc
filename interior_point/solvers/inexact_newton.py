from .newton_ip import NewtonIP
import numpy as np
from .newton_utils import take_step, take_sparse_step, get_residual_norm
from scipy.sparse import csc_matrix

class InexactNewton(NewtonIP):
    def __init__(self, Q, R, S, Qf, q, qf, r, A, B, w, Fx, Fu, f, x0, n, m, T, use_sparse=True):
        super().__init__(Q, R, S, Qf, q, qf, r, A, B, w, Fx, Fu, f, x0, n, m, T)
        self.use_sparse = use_sparse

        if self.use_sparse:
            self.H = csc_matrix(self.H)
            self.C = csc_matrix(self.C)
            self.P = csc_matrix(self.P)

    def advance_z_nu(self):
        # advance z and nu forward by one time step. leave the end values the same
        # under box constraints, this will be able to solve just fine
        shift = self.m+self.n

        self.z[:-shift] = self.z[shift:]
        self.nu[:-shift] = self.nu[shift:]

    def solve(self, kappa=10, max_iterations=5, warm_start=True):
        # use x as it was set, return next u
        self.create_variants()

        if not warm_start:
            self.z = np.zeros(self.sz_z)
            self.nu = np.zeros(self.sz_nu)
        else:
            self.advance_z_nu()

        # in this solver, we use a fixed value of kappa instead of
        # trying to move along the central path
        tol = 1e-3
        iterations = 0

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
        while obj > tol and iterations < max_iterations:
            if not self.use_sparse:
                self.z, self.nu = take_step(
                    self.z,
                    self.nu,
                    kappa,
                    self.H,
                    self.g,
                    self.C,
                    self.b,
                    self.P,
                    self.h,
                    use_diagonal=True
                )
            else:
                self.z, self.nu = take_sparse_step(
                    self.z,
                    self.nu,
                    kappa,
                    self.H,
                    self.g,
                    self.C,
                    self.b,
                    self.P,
                    self.h
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

            iterations += 1
