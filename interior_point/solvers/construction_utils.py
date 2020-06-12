import numpy as np

def make_H(Q, Qf, R, S, n, m, T):
    Q_size = T * (n+m)
    H = np.zeros((Q_size, Q_size))

    for i in range(T):
        R_start = i * (n+m)
        Q_start = R_start + m

        H[R_start:R_start+m, R_start:R_start+m] = R

        if i < T-1:
            H[Q_start:Q_start+n, Q_start:Q_start+n] = Q
            H[Q_start:Q_start+n, Q_start+n:Q_start+n+m] = S
            H[Q_start+n:Q_start+n+m, Q_start:Q_start+n] = S.T
        else:
            H[Q_start:Q_start+n, Q_start:Q_start+n] = Qf

    return H

def make_g(S, x0, r, q, qf, n, m, T):
    g = np.zeros((n+m)*T)

    for i in range(T):
        r_start = i * (n+m)
        q_start = r_start + m

        g[r_start:r_start+m] = r

        if i < T-1:
            g[q_start:q_start+n] = q
        else:
            g[q_start:q_start+n] = qf

    g[0:m] += 2*S.T.dot(x0)

    return g

def make_C(A, B, n, m, T):
    C = np.zeros((T*n, T*(n+m)))

    for i in range(T):
        row_start = i*n
        row_end = (i+1)*n

        if i == 0:
            col_start = 0
            col_end = n+m
        else:
            col_start = m + (i-1)*(n+m)
            col_end = m + n + i*(m+n)

        if i == 0:
            C[row_start:row_end, col_start:col_end] = np.hstack(( -B, np.eye(n) ))
        else:
            C[row_start:row_end, col_start:col_end] = np.hstack(( -A, -B, np.eye(n) ))

    return C

def make_b(A, x0, w, n, T):
    b = np.zeros(T*n)

    for i in range(T):
        b[i*n:(i+1)*n] = w

    b[:n] += A.dot(x0)

    return b

def make_P(Fx, Fu, n, m, T):
    P_rows = 2*((n+m)*(T) + n)
    P_cols = (n+m)*T
    P = np.zeros((P_rows, P_cols))

    for i in range(T+1):
        if i == 0:
            col_start = 0
            col_end = m
        elif i < T:
            col_start = (i-1)*(n+m) + m
            col_end = i*(n+m) + m
        else:
            col_start = (i-1)*(n+m) + m
            col_end = -1

        row_start = 2*i*(n+m)
        if i < T:
            row_end = 2*(i+1)*(n+m)
        else:
            row_end = -1

        if i == 0:
            P[row_start:row_end, col_start:col_end] = Fu
        elif i < T:
            P[row_start:row_end, col_start:col_end] = np.hstack(( Fx, Fu ))
        else:
            P[row_start:, col_start:] = Fx[:2*n, :]

    return P

def make_h(f, Fx, x0, n, m, T):
    h_rows = 2*((n+m)*(T) + n)
    h = np.zeros(h_rows)

    for i in range(T+1):
        row_start = 2*i*(n+m)
        if i < T:
            row_end = 2*(i+1)*(n+m)
        else:
            row_end = -1

        if i < T:
            h[row_start:row_end] = f.squeeze()
        else:
            h[row_start:] = f.squeeze()[:2*n]

        if i == 0:
            h[row_start:row_end] -= Fx.dot(x0)

    return h
