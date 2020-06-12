from admm_warm_start import ADMMWarmStart
from scipy.sparse import csc_matrix

class ADMMSparse(ADMMWarmStart):
    def __init__(self, ydims, Pis, Cs, D, lower_bounds, upper_bounds, vdims, T, yi_dims, rho=0.5):
        super().__init__(ydims, Pis, Cs, D, lower_bounds, upper_bounds, vdims, T, yi_dims, rho)

        self.D_mat = csc_matrix(self.D_mat)
        self.E_mat = csc_matrix(self.E_mat)
