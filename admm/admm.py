import numpy as np

class ADMMSolver:
    def __init__(self, ydims, Pis, Cs, D, lower_bounds, upper_bounds, vdims, T, rho=0.5):
        self.ydims = ydims
        self.Pis = Pis
        self.Cs = Cs
        self.D = D
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.vdims = vdims
        self.T = T

        self.rho = rho

        self.y = np.zeros(sum(self.ydims))
        self.zeta = np.zeros(sum(self.ydims))
        self.epsilon = np.zeros(sum(self.ydims))
        self.lambda_zeta = np.zeros(sum(self.ydims))
        self.lambda_epsilon = np.zeros(sum(self.ydims))

        self.create_invariants()

    def create_invariants(self):
        self.Mis = []
        self.Nis = []

        for i in range(len(self.Cs)):
            Pi = self.Pis[i]
            Ci = self.Cs[i]
            CPC = Ci.dot(Pi).dot(Ci.T)
            self.Mis += [
                Pi - \
                Pi.dot( Ci.T ).dot( np.linalg.solve(CPC, Ci.dot(Pi)) )
            ]

            self.Nis += [
                Pi.dot( Ci.T ).dot( np.linalg.inv(CPC) )
            ]

        self.D_mat = np.eye(self.D.shape[1]) - \
            self.D.T.dot(np.linalg.solve(self.D.dot(self.D.T), self.D))

        self.E_mat = self.D.T.dot(np.linalg.inv(self.D.dot(self.D.T)))

    def create_variants(self, problem):
        self.cs = problem.get_cs(self.T)
        self.d = problem.get_d(self.T)

    def set_x(self, x0):
        self.x = x0

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


    def get_command(self):
        u = np.zeros(sum(self.vdims))

        start_idx_u = 0
        end_idx_u = 0
        start_idx_y = 0
        end_idx_y = 0
        for i in range(len(self.vdims)):
            end_idx_u += self.vdims[i]
            end_idx_y += self.ydims[i]

            u[start_idx_u:end_idx_u] = self.y[start_idx_y:end_idx_y][:self.vdims[i]]

            start_idx_u += self.vdims[i]
            start_idx_y += self.ydims[i]

        return u

    def get_solution(self):
        return self.y
    
    def get_dual_variables(self):
        return self.lambda_zeta, self.lambda_epsilon