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

        self.z_partition = 2 * [6]
        self.v_partition = [1, 0, 1, 0, 0, 1]

    def get_Q(self):
        return np.eye(self.n)

    def get_R(self):
        return np.eye(self.m)

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

    def step(self, u):
        A, B = self.get_AB()

        # input noise is uniformly distributed random
        # force acting on the masses
        input_noise = np.random.rand(self.n // 2) - 0.5

        corrupting = np.zeros(self.n)
        corrupting[1::2] = input_noise * self.dt / self.mass

        self.x = A.dot(self.x) + B.dot(u) + corrupting

    def get_As(self):
        A_i = np.array([
            [1, self.dt],
            [-2*self.k*self.dt / self.mass, 1]
        ])

        return 6 * [A_i]

    def get_Bs(self):
        B_p = np.array([[0, self.dt / self.mass]]).T
        B_n = -B_p

        empty = np.zeros(( 2, 0 ))

        return [B_p, empty, B_p, empty, empty, B_n]

    def get_Ws(self):
        w_single = np.array([[0, self.k*self.dt / self.mass]]).T

        Ws = []

        Ws += [w_single]
        Ws += [np.hstack(( w_single, w_single, -w_single ))]
        Ws += [np.hstack(( w_single, w_single ))]
        Ws += [np.hstack(( w_single, w_single, w_single ))]
        Ws += [np.hstack(( w_single, w_single, -w_single ))]
        Ws += [np.hstack(( w_single, w_single ))]

        return Ws

    def get_zbounds(self):
        z_lower = np.array([[ self.x_min, self.xdot_min ]]).T
        z_upper = np.array([[ self.x_max, self.xdot_max ]]).T

        return 6*[z_lower], 6*[z_upper]

    def get_vbounds(self):
        return [self.F_min, [], self.F_min, [], [], self.F_min], [self.F_max, [], self.F_max, [], [], self.F_max]

    def get_bounds(self, T):
        lower_bounds = []
        upper_bounds = []

        Ws = self.get_Ws()

        zbounds = self.get_zbounds()
        vbounds = self.get_vbounds()

        for i in range(6):
            wdim = Ws[i].shape[1]

            single_lower_bound = np.hstack((vbounds[0][i], -1e6*np.ones(wdim), zbounds[0][i].squeeze()))
            single_upper_bound = np.hstack((vbounds[1][i], 1e6*np.ones(wdim), zbounds[1][i].squeeze()))
            
            lower_bounds.append(np.tile(single_lower_bound, T))
            upper_bounds.append(np.tile(single_upper_bound, T))

        return lower_bounds, upper_bounds


    def get_Q_tildes(self):
        return 6*[self.get_Q()[:2,:2]]

    def get_R_tildes(self):
        return [
            np.array([[1]]),
            np.array([[]]),
            np.array([[1]]),
            np.array([[]]),
            np.array([[]]),
            np.array([[1]])
        ]

    def get_zinits(self):
        zinits = []

        for i in range(6):
            zinits.append(self.x[2*i:2*(i+1)])

        return zinits

    def get_Cs(self, T):
        As = self.get_As()
        Bs = self.get_Bs()
        Ws = self.get_Ws()

        Cs = []

        for i in range(6):
            vdim = Bs[i].shape[1]
            wdim = Ws[i].shape[1]
            zdim = As[i].shape[1]

            C_i = np.zeros(( T*zdim, vdim + wdim + zdim + (T-1)*(zdim + wdim + vdim) ))

            for j in range(T):
                row_start = zdim*j
                row_end = zdim*(j+1)
                if j==0:
                    col_start = 0
                    col_end = wdim + vdim + zdim

                    data = np.hstack(( Bs[i], Ws[i], -np.eye(zdim) ))
                else:
                    col_start = wdim + vdim + (j-1)*(zdim + wdim + vdim)
                    col_end = col_start + (2*zdim + wdim + vdim)

                    data = np.hstack(( As[i], Bs[i], Ws[i], -np.eye(zdim) ))

                C_i[row_start:row_end, col_start:col_end] = data

            Cs.append(C_i)
        
        return Cs

    def get_cs(self, T):
        As = self.get_As()

        zinits = self.get_zinits()

        cs = []

        for i in range(6):
            zdim = As[i].shape[1]

            c_i = np.zeros( T*zdim )
            c_i[:zdim] -= As[i].dot(zinits[i])
            cs.append(c_i)

        return cs

    def get_D(self, T):
        full_A, full_B = self.get_AB()

        As = self.get_As()
        Bs = self.get_Bs()
        Ws = self.get_Ws()

        n_w = sum([W.shape[1] for W in Ws])
        y_sizes = np.array([Ws[i].shape[1] + Bs[i].shape[1] + As[i].shape[1] for i in range(6)])
        n_y = (T * y_sizes).sum()

        D = np.zeros(( T*n_w, n_y ))

        D_row_st = 0
        D_row_end = 0

        for i in range(6):
            zdim_i = As[i].shape[1]
            wdim_i = Ws[i].shape[1]
            vdim_i = Bs[i].shape[1]

            D_row_end += T*wdim_i

            D_col_st = 0
            D_col_end = 0
            
            B_col_st = 0
            B_col_end = 0
            for j in range(6):
                zdim_j = As[j].shape[1]
                wdim_j = Ws[j].shape[1]
                vdim_j = Bs[j].shape[1]

                D_Aij = np.zeros(( wdim_i, (wdim_j + zdim_j + vdim_j) ))
                D_Bij = np.zeros(( wdim_i, (wdim_j + zdim_j + vdim_j) ))

                D_ij = np.zeros(( T*wdim_i, T*(wdim_j+zdim_j+vdim_j) ))

                B_col_end += self.v_partition[j]

                # build D_Aij, D_Bij
                if i == j:
                    D_Bij[:, vdim_j:vdim_j+wdim_j] = -np.eye(wdim_j)
                else:
                    W_dagger = np.linalg.pinv(Ws[i])

                    Aij = full_A[2*i:2*i+zdim_i,2*j:2*j+zdim_j]
                    Bij = full_B[2*i:2*i+zdim_i,B_col_st:B_col_end]

                    D_Aij[:, vdim_j+wdim_j:] = W_dagger.dot(Aij)
                    D_Bij[:, :vdim_j] = W_dagger.dot(Bij)

                B_col_st += self.v_partition[j]

                # build D_ij
                for k in range(T):
                    D_ij_row_st = k*wdim_i
                    D_ij_row_end = (k+1)*wdim_i

                    if k == 0:
                        D_ij_col_st = 0
                        D_ij_col_end = wdim_j + vdim_j + zdim_j

                        data = D_Bij
                    else:
                        D_ij_col_st = (k-1) * (wdim_j + vdim_j + zdim_j)
                        D_ij_col_end = D_ij_col_st + 2*(wdim_j + vdim_j + zdim_j)

                        data = np.hstack(( D_Aij, D_Bij ))

                    D_ij[D_ij_row_st:D_ij_row_end, D_ij_col_st:D_ij_col_end] = data

                # put D_ij into D
                D_col_end += T*(wdim_j + vdim_j + zdim_j)

                D[D_row_st:D_row_end, D_col_st:D_col_end] = D_ij

                D_col_st += T*(wdim_j + vdim_j + zdim_j)
                
            D_row_st += T*wdim_i

        return D

    def get_d(self, T):
        full_A, _ = self.get_AB()

        As = self.get_As()
        Bs = self.get_Bs()
        Ws = self.get_Ws()

        zinits = self.get_zinits()

        n_w = sum([W.shape[1] for W in Ws])
        d = np.zeros(( T*n_w ))

        d_row_st = 0
        d_row_end = 0

        for i in range(6):
            zdim_i = As[i].shape[1]
            wdim_i = Ws[i].shape[1]

            di = np.zeros( T*wdim_i )

            d_row_end += T*wdim_i

            for j in range(6):
                zdim_j = As[j].shape[1]
                wdim_j = Ws[j].shape[1]
                vdim_j = Bs[j].shape[1]

                D_Aij = np.zeros(( wdim_i, (wdim_j + zdim_j + vdim_j) ))

                if i != j:
                    W_dagger = np.linalg.pinv(Ws[i])

                    Aij = full_A[2*i:2*i+zdim_i,2*j:2*j+zdim_j]

                    D_Aij[:, vdim_j+wdim_j:] = W_dagger.dot(Aij)

                initial_condition = np.zeros(vdim_j+wdim_j+zdim_j)
                initial_condition[(vdim_j+wdim_j):] = zinits[j]

                di[:wdim_i] += -D_Aij.dot(initial_condition)

            d[d_row_st:d_row_end] = di

            d_row_st += T*wdim_i

        return d

    def get_Qis(self, T):
        I_T = np.eye(T)
        Qts = self.get_Q_tildes()
        Rts = self.get_R_tildes()
        Ws = self.get_Ws()

        Qs = []

        for i in range(6):
            cost = np.diag( np.hstack(( np.diag(Rts[i]), np.zeros(Ws[i].shape[1]), np.diag(Qts[i]) )) )

            Qs.append(np.kron(I_T, cost))

        return Qs

    def get_Pis(self, T, rho):
        Qis = self.get_Qis(T)

        Pis = []

        for i in range(len(Qis)):
            Pis += [np.linalg.inv((1/rho) * Qis[i] + 2*np.eye(Qis[i].shape[0]))]

        return Pis

    def get_dims(self, T):
        ydims = []
        yi_dims = []
        vdims = []

        As = self.get_As()
        Bs = self.get_Bs()
        Ws = self.get_Ws()

        for i in range(6):
            zdim = As[i].shape[1]
            vdim = Bs[i].shape[1]
            wdim = Ws[i].shape[1]

            ydims.append( T*(zdim+wdim+vdim) )
            yi_dims.append( zdim + vdim + wdim )
            vdims.append( vdim )

        return ydims, yi_dims, vdims
