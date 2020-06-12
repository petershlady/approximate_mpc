import numpy as np

class OscillatingMasses:
    def __init__(self):
        self.mass = 1
        self.k = 1
        self.w_max = 0.5
        self.w_min = -0.5
        self.F_max = 0.5
        self.F_min = -0.5
        self.x_max = 4
        self.x_min = -4
        self.xdot_max = 5
        self.xdot_min = -5
        self.dt = 0.05

        self.x = np.random.randn(12)

        self.x[0::2] = np.maximum(np.minimum(self.x[0::2], self.x_max), self.x_min)
        self.x[1::2] = np.maximum(np.minimum(self.x[1::2], self.xdot_max), self.xdot_min)

        self.n = 12
        self.m = 3

    def get_Q(self):
        return np.eye(self.n)

    def get_R(self):
        return np.eye(self.m)

    def get_Qf(self):
        return self.get_Q()

    def get_q(self):
        return np.zeros(self.n)

    def get_qf(self):
        return np.zeros(self.n)

    def get_r(self):
        return np.zeros(self.m)

    def get_S(self):
        return np.zeros((self.n, self.m))

    def get_AB(self):
        A = np.zeros((self.n, self.n))
        B = np.zeros((self.n, self.m))

        # for discrete time update, need to include state at time k
        for i in range(self.n):
            A[i,i] = 1

        for i in range(self.n // 2):
            # xdot = xdot
            A[2*i, 2*i+1] = self.dt

        # mass 1
        A[1,0] = -2*self.k * self.dt / self.mass
        A[1,2] = self.k * self.dt / self.mass
        B[1,0] = self.dt / self.mass

        # mass 2
        A[3,0] = self.k * self.dt / self.mass
        A[3,2] = -2*self.k * self.dt / self.mass
        A[3,4] = self.k * self.dt / self.mass
        B[3,0] = -self.dt / self.mass

        # mass 3
        A[5,2] = self.k * self.dt / self.mass
        A[5,4] = -2*self.k * self.dt / self.mass
        A[5,6] = self.k * self.dt / self.mass
        B[5,1] = self.dt / self.mass

        # mass 4
        A[7,4] = self.k * self.dt / self.mass
        A[7,6] = -2*self.k * self.dt / self.mass
        A[7,8] = self.k * self.dt / self.mass
        B[7,2] = self.dt / self.mass

        # mass 5
        A[9,6] = self.k * self.dt / self.mass
        A[9,8] = -2*self.k * self.dt / self.mass
        A[9,10] = self.k * self.dt / self.mass
        B[9,1] = -self.dt / self.mass

        # mass 6
        A[11,8] = self.k * self.dt / self.mass
        A[11,10] = -2*self.k * self.dt / self.mass
        B[11,2] = -self.dt / self.mass

        return A, B

    def get_f(self):
        f = np.zeros(2*(self.n+self.m))

        for i in range(self.n):
            if i % 2:
                f[2*i] = self.xdot_max
                f[2*i+1] = -self.xdot_min
            else:
                f[2*i] = self.x_max
                f[2*i+1] = -self.x_min

        for i in range(self.m):
            f[2*self.n + 2*i] = self.F_max
            f[2*self.n + 2*i + 1] = -self.F_min

        return f

    def get_w(self):
        return np.zeros(self.n)

    def get_Fx(self):
        Fx = np.zeros((2*(self.n + self.m), self.n))

        for i in range(self.n):
            Fx[2*i, i] = 1
            Fx[2*i+1, i] = -1

        return Fx

    def get_Fu(self):
        Fu = np.zeros((2*(self.n + self.m), self.m))

        for i in range(self.m):
            Fu[2*self.n + 2*i, i] = 1
            Fu[2*self.n + 2*i + 1, i] = -1

        return Fu

    def step(self, u):
        A, B = self.get_AB()

        # input noise is uniformly distributed random
        # force acting on the masses
        input_noise = np.random.rand(self.n // 2) - 0.5

        corrupting = np.zeros(self.n)
        corrupting[1::2] = input_noise * self.dt / self.mass

        self.x = A.dot(self.x) + B.dot(u) + corrupting

if __name__ == "__main__":
    om = OscillatingMasses()
    A, B = om.get_AB()
    x0 = om.x

    x1 = A.dot(x0)
    print('dt, m, k: (%f, %f, %f)' % (om.dt, om.mass, om.k))

    Fx = om.get_Fx()
    Fu = om.get_Fu()
    f = om.get_f()

    import pdb; pdb.set_trace()
