import numpy as np
import scipy.linalg as linalg
from .rla_utils import matmul
from scipy.sparse import csc_matrix, diags, hstack, vstack
from scipy.sparse.linalg import spsolve

def dual_residual(H, z, g, kappa, P, d, C, nu):
    return 2*H.dot(z) + g + kappa*P.T.dot(d) + C.T.dot(nu)

def primal_residual(C, z, b):
    return C.dot(z) - b

def build_d(h, P, z):
    return 1 / (h - P.dot(z))

def build_phi(H, kappa, P, z, h, use_sparse=False):
    if not use_sparse:
        d = build_d(h, P, z)
        first = 2*H
        second = kappa*P.T.dot((P.T * d**2).T)
    else:
        d = diags(build_d(h, P, z)**2)
        first = 2*H
        second = kappa * P.T.dot(d).dot(P)
    return first+second

def build_newton_A(H, kappa, P, z, C, h, use_sparse=False):
    if not use_sparse:
        row1 = np.hstack((build_phi(H, kappa, P, z, h), C.T))
        row2 = np.hstack(( C, np.zeros((C.shape[0], C.shape[0])) ))

        return np.vstack(( row1, row2 ))
    else:
        row1 = hstack(( build_phi(H, kappa, P, z, h, use_sparse), C.T ))
        row2 = hstack(( C, csc_matrix((C.shape[0], C.shape[0])) ))

        return vstack(( row1, row2 ))

def build_newton_b(H, z, g, kappa, P, C, nu, b, h):
    d = build_d(h, P, z)
    return -np.hstack(( dual_residual(H, z, g, kappa, P, d, C, nu), primal_residual(C, z, b) ))

def func(z, kappa, H, g, P, h):
    if np.all(h-P.dot(z) >= 0):
        return z.T.dot(H).dot(z) + + g.T.dot(z) - kappa * np.log(h - P.dot(z)).sum()
    else:
        return 1e10

def grad(z, kappa, H, g, P, h):
    d = build_d(h, P, z)
    return 2 * H.dot(z) + g + kappa * P.T.dot(d)

def check_feasibility(z, P, h):
    residual = h - P.dot(z)

    return np.all(residual > -1e-10)

def calculate_newton_step(z, nu, kappa, H, g, C, b, P, h):
    A_newt = build_newton_A(H, kappa, P, z, C, h)
    b_newt = build_newton_b(H, z, g, kappa, P, C, nu, b, h)

    return np.linalg.solve(A_newt, b_newt)

def calculate_diagonal_newton_step(z, nu, kappa, H, g, C, b, P, h, use_sparse=False):
    # useful Q, R are diagonal and S=0 because H is then diagonal
    # look at http://web.stanford.edu/class/ee364a/lectures/num-lin-alg.pdf slide 9-12
    if use_sparse:
        phi_inv = 1 / build_phi(H, kappa, P, z, h, use_sparse).diagonal()
    else:
        phi_inv = 1 / np.diag(build_phi(H, kappa, P, z, h, use_sparse))
    d = build_d(h, P, z)
    rd = dual_residual(H, z, g, kappa, P, d, C, nu)
    rp = primal_residual(C, z, b)

    if use_sparse:
        D = diags(phi_inv).dot(C.T)
    else:
        D = (C*phi_inv).T
    E = rd*phi_inv

    Y = C.dot(D)

    beta = -rp + C.dot(E)

    if use_sparse:
        dnu = spsolve(Y, -beta)
    else:
        dnu = np.linalg.solve(Y, -beta)
    dz = phi_inv * (-rd - C.T.dot(dnu))

    return np.hstack(( dz, dnu ))

def take_step(z, nu, kappa, H, g, C, b, P, h, use_diagonal=False):
    alpha = 0.10
    beta = 0.8

    if not use_diagonal:
        ns = calculate_newton_step(z, nu, kappa, H, g, C, b, P, h)
    else:
        ns = calculate_diagonal_newton_step(z, nu, kappa, H, g, C, b, P, h)

    # backtracking line search on norm of the residuals
    t = 1
    d0 = build_d(h, P, z)
    residual_0 = np.hstack(( dual_residual(H, z, g, kappa, P, d0, C, nu), primal_residual(C, z, b) ))
    grad_residual_0 = build_newton_A(H, kappa, P, z, C, h)

    next_z = z + t * ns[:z.shape[0]]
    next_nu = nu + t * ns[z.shape[0]:]
    next_d = build_d(h, P, next_z)
    residual_1 = np.hstack(( dual_residual(H, next_z, g, kappa, P, next_d, C, next_nu), primal_residual(C, next_z, b) ))
    feasibility = check_feasibility(next_z, P, h)
    
    while np.linalg.norm(residual_1) > np.linalg.norm( residual_0 + alpha * t * grad_residual_0.T.dot(ns) ) or (not feasibility):
        t *= beta
        next_z = z + t * ns[:z.shape[0]]
        next_nu = nu + t * ns[z.shape[0]:]
        next_d = build_d(h, P, next_z)
        residual_1 = np.hstack(( dual_residual(H, next_z, g, kappa, P, next_d, C, next_nu), primal_residual(C, next_z, b) ))
        feasibility = check_feasibility(next_z, P, h)

    return z + t * ns[:z.shape[0]], nu + t * ns[z.shape[0]:]

def take_sparse_step(z, nu, kappa, H, g, C, b, P, h):
    alpha = 0.10
    beta = 0.8
    sp = True

    # import time
    # st = time.time()
    # ns = calculate_diagonal_newton_step(z, nu, kappa, H, g, C, b, P, h)
    # print(time.time()-st)

    # H = csc_matrix(H)
    # P = csc_matrix(P)
    # C = csc_matrix(C)
    # st = time.time()
    # nss = calculate_diagonal_newton_step(z, nu, kappa, H, g, C, b, P, h, sp)
    # print(time.time()-st)
    # print(np.linalg.norm(nss-ns))
    # phi_s = build_phi(H, kappa, P, z, h, sp)
    # phi = build_phi(H, kappa, P, z, h)

    ns = calculate_diagonal_newton_step(z, nu, kappa, H, g, C, b, P, h, sp)

    # backtracking line search on norm of the residuals
    t = 1
    d0 = build_d(h, P, z)
    # import pdb; pdb.set_trace()
    residual_0 = np.hstack(( dual_residual(H, z, g, kappa, P, d0, C, nu), primal_residual(C, z, b) ))
    grad_residual_0 = build_newton_A(H, kappa, P, z, C, h, sp)

    next_z = z + t * ns[:z.shape[0]]
    next_nu = nu + t * ns[z.shape[0]:]
    next_d = build_d(h, P, next_z)
    residual_1 = np.hstack(( dual_residual(H, next_z, g, kappa, P, next_d, C, next_nu), primal_residual(C, next_z, b) ))
    feasibility = check_feasibility(next_z, P, h)
    
    while np.linalg.norm(residual_1) > np.linalg.norm( residual_0 + alpha * t * grad_residual_0.T.dot(ns) ) or (not feasibility):
        t *= beta
        next_z = z + t * ns[:z.shape[0]]
        next_nu = nu + t * ns[z.shape[0]:]
        next_d = build_d(h, P, next_z)
        residual_1 = np.hstack(( dual_residual(H, next_z, g, kappa, P, next_d, C, next_nu), primal_residual(C, next_z, b) ))
        feasibility = check_feasibility(next_z, P, h)

    return z + t * ns[:z.shape[0]], nu + t * ns[z.shape[0]:]

def get_residual_norm(H, z, g, kappa, P, C, nu, b, h):
    return np.linalg.norm(build_newton_b(H, z, g, kappa, P, C, nu, b, h))

def calculate_uniform_newton_step(z, nu, kappa, H, g, C, b, P, h, RH, D):
    A_newt = build_newton_A(H, kappa, P, z, C, h)
    b_newt = build_newton_b(H, z, g, kappa, P, C, nu, b, h)

    A_newt = RH.dot(((A_newt.T * D).T))
    b_newt = RH.dot(D * b_newt)

    return np.linalg.solve(A_newt, b_newt)

def take_uniform_random_step(z, nu, kappa, H, g, C, b, P, h, r, hadamard=None):
    alpha = 0.10
    beta = 0.8

    n = H.shape[0]

    if hadamard is None:
        hadamard = linalg.hadamard(n)
    
    R_locs = np.random.choice(n, size=r, replace=False)
    R = np.zeros((r, n))
    R[:, R_locs] = np.sqrt(n / r)

    RH = R.dot(hadamard)

    D = np.sign(np.random.rand(n) - 0.5)

    import pdb; pdb.set_trace()
    ns = calculate_uniform_newton_step(z, nu, kappa, H, g, C, b, P, h, RH, D)

    # backtracking line search on norm of the residuals
    t = 1
    d0 = build_d(h, P, z)
    residual_0 = np.hstack(( dual_residual(H, z, g, kappa, P, d0, C, nu), primal_residual(C, z, b) ))
    grad_residual_0 = build_newton_A(H, kappa, P, z, C, h)

    next_z = z + t * ns[:z.shape[0]]
    next_nu = nu + t * ns[z.shape[0]:]
    next_d = build_d(h, P, next_z)
    residual_1 = np.hstack(( dual_residual(H, next_z, g, kappa, P, next_d, C, next_nu), primal_residual(C, next_z, b) ))
    feasibility = check_feasibility(next_z, P, h)
    
    while np.linalg.norm(residual_1) > np.linalg.norm( residual_0 + alpha * t * grad_residual_0.T.dot(ns) ) or (not feasibility):
        t *= beta
        next_z = z + t * ns[:z.shape[0]]
        next_nu = nu + t * ns[z.shape[0]:]
        next_d = build_d(h, P, next_z)
        residual_1 = np.hstack(( dual_residual(H, next_z, g, kappa, P, next_d, C, next_nu), primal_residual(C, next_z, b) ))
        feasibility = check_feasibility(next_z, P, h)

    return z + t * ns[:z.shape[0]], nu + t * ns[z.shape[0]:]
