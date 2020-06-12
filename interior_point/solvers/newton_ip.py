import numpy as np
from .construction_utils import make_H, make_g, make_C, make_b, make_P, make_h
from .newton_utils import take_step, get_residual_norm

class NewtonIP:
    def __init__(self, Q, R, S, Qf, q, qf, r, A, B, w, Fx, Fu, f, x0, n, m, T):
        self.Q = Q
        self.R = R
        self.S = S
        self.Qf = Qf
        self.q = q
        self.qf = qf
        self.r = r
        self.A = A
        self.B = B
        self.w = w
        self.Fx = Fx
        self.Fu = Fu
        self.f = f

        self.x = x0

        self.sz_z = T*(n+m)
        self.sz_nu = T*n

        # initialize z in case it gets requested
        self.z = np.zeros(self.sz_z)
        # initialize nu in case it gets requested
        self.nu = np.zeros(self.sz_nu)

        self.n = n
        self.m = m
        self.T = T

        self.create_invariants()

    def create_invariants(self):
        # non-changing matrices
        # for now, assume time invariant dynamics
        self.H = make_H(self.Q, self.Qf, self.R, self.S, self.n, self.m, self.T)
        self.C = make_C(self.A, self.B, self.n, self.m, self.T)
        self.P = make_P(self.Fx, self.Fu, self.n, self.m, self.T)

    def create_variants(self):
        # things that can change over time (functions of state)
        self.g = make_g(self.S, self.x, self.r, self.q, self.qf, self.n, self.m, self.T)
        self.b = make_b(self.A, self.x, self.w, self.n, self.T)
        self.h = make_h(self.f, self.Fx, self.x, self.n, self.m, self.T)

    def set_x(self, x):
        self.x = x

    def solve(self):
        # use x as it was set, return next u
        self.create_variants()

        self.z = np.zeros(self.sz_z)
        self.nu = np.zeros(self.sz_nu)

        kappa = 10
        tol = 1e-3
        max_outer_iters = 100
        max_inner_iters = 100
        num_outer_iters = 0
        num_inner_iters = 0

        while kappa > 1e-4 and num_outer_iters < max_outer_iters:
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

            num_inner_iters = 0
            while obj > tol and num_inner_iters < max_inner_iters:
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
                num_inner_iters += 1

            kappa /= 10
            num_outer_iters += 1

    def get_command(self):
        return self.z[:self.m]

    def get_solution(self):
        # return all of z as calculated
        return self.z

    def get_dual_variables(self):
        # return calculated dual variables
        return self.nu
