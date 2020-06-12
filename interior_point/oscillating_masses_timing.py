from solvers.newton_ip import NewtonIP
from solvers.inexact_newton import InexactNewton
from solvers.uniform_random_newton import UniformRandomNewton
from problems.oscillating_masses import OscillatingMasses
import numpy as np
import time
import cvxpy as cp
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import copy

Ts = [2, 5, 10, 20, 30, 50]

om = OscillatingMasses()

num_steps = 100
rates = np.zeros(( 5, len(Ts) ))

om0 = OscillatingMasses()

for num_t, T in enumerate(Ts):
    # MPC test for exact newton
    om = copy.copy(om0)
    exact_solver = NewtonIP(
        om.get_Q(),
        om.get_R(),
        om.get_S(),
        om.get_Qf(),
        om.get_q(),
        om.get_qf(),
        om.get_r(),
        om.get_AB()[0],
        om.get_AB()[1],
        om.get_w(),
        om.get_Fx(),
        om.get_Fu(),
        om.get_f(),
        om.x,
        om.n,
        om.m,
        T
    )

    results_exact_newton = np.zeros((num_steps+1, om.n))
    results_exact_newton[0] = om.x

    total_time = 0
    for step_i in range(num_steps):
        exact_solver.set_x(om.x)

        st = time.time()
        exact_solver.solve()
        total_time += time.time() - st

        u = exact_solver.get_command()
        x_pred = exact_solver.get_solution()[3:15]

        om.step(u)
        results_exact_newton[step_i+1] = om.x

        if (step_i + 1) % 50 == 0:
            print('Exact newton completed step %d/%d' % (step_i+1,num_steps))
    rate = total_time/num_steps
    print('Exact Newton for T=%d took %f seconds per step (%f Hz)' % (T, rate, 1/rate))
    rates[0,num_t] = rate

    # MPC test for inexact newton
    om = copy.copy(om0)
    inexact_solver = InexactNewton(
        om.get_Q(),
        om.get_R(),
        om.get_S(),
        om.get_Qf(),
        om.get_q(),
        om.get_qf(),
        om.get_r(),
        om.get_AB()[0],
        om.get_AB()[1],
        om.get_w(),
        om.get_Fx(),
        om.get_Fu(),
        om.get_f(),
        om.x,
        om.n,
        om.m,
        T,
        False
    )

    results_inexact_newton = np.zeros((num_steps+1, om.n))
    results_inexact_newton[0] = om.x

    total_time = 0
    for step_i in range(num_steps):
        inexact_solver.set_x(om.x)

        st = time.time()
        inexact_solver.solve(warm_start=False)
        total_time += time.time() - st

        u = inexact_solver.get_command()
        x_pred = inexact_solver.get_solution()[3:15]

        om.step(u)
        results_inexact_newton[step_i+1] = om.x

        if (step_i + 1) % 50 == 0:
            print('Inexact newton (no warm start) completed step %d/%d' % (step_i+1,num_steps))
    rate = total_time/num_steps
    print('Inexact Newton (no warm start) for T=%d took %f seconds per step (%f Hz)' % (T, rate, 1/rate))
    rates[1,num_t] = rate

    # MPC test for inexact newton with warm start
    om = copy.copy(om0)
    inexact_solver = InexactNewton(
        om.get_Q(),
        om.get_R(),
        om.get_S(),
        om.get_Qf(),
        om.get_q(),
        om.get_qf(),
        om.get_r(),
        om.get_AB()[0],
        om.get_AB()[1],
        om.get_w(),
        om.get_Fx(),
        om.get_Fu(),
        om.get_f(),
        om.x,
        om.n,
        om.m,
        T,
        False
    )

    results_inexact_newton_ws = np.zeros((num_steps+1, om.n))
    results_inexact_newton_ws[0] = om.x

    total_time = 0
    for step_i in range(num_steps):
        inexact_solver.set_x(om.x)

        st = time.time()
        inexact_solver.solve()
        total_time += time.time() - st

        u = inexact_solver.get_command()
        x_pred = inexact_solver.get_solution()[3:15]

        om.step(u)
        results_inexact_newton_ws[step_i+1] = om.x

        if (step_i + 1) % 50 == 0:
            print('Inexact newton (warm start) completed step %d/%d' % (step_i+1,num_steps))
    rate = total_time/num_steps
    print('Inexact Newton (warm start) for T=%d took %f seconds per step (%f Hz)' % (T, rate, 1/rate))
    rates[2,num_t] = rate

    # MPC test for inexact newton with warm start and sparse solution
    om = copy.copy(om0)
    inexact_sparse_solver = InexactNewton(
        om.get_Q(),
        om.get_R(),
        om.get_S(),
        om.get_Qf(),
        om.get_q(),
        om.get_qf(),
        om.get_r(),
        om.get_AB()[0],
        om.get_AB()[1],
        om.get_w(),
        om.get_Fx(),
        om.get_Fu(),
        om.get_f(),
        om.x,
        om.n,
        om.m,
        T,
        True
    )

    results_inexact_newton_ws_sp = np.zeros((num_steps+1, om.n))
    results_inexact_newton_ws_sp[0] = om.x

    total_time = 0
    for step_i in range(num_steps):
        inexact_sparse_solver.set_x(om.x)

        st = time.time()
        inexact_sparse_solver.solve()
        total_time += time.time() - st

        u = inexact_sparse_solver.get_command()

        om.step(u)
        results_inexact_newton_ws_sp[step_i+1] = om.x

        if (step_i + 1) % 50 == 0:
            print('Inexact newton (warm start, sparse) completed step %d/%d' % (step_i+1,num_steps))
    rate = total_time/num_steps
    print('Inexact Newton (warm start, sparse) for T=%d took %f seconds per step (%f Hz)' % (T, rate, 1/rate))
    rates[3,num_t] = rate

    # MPC test for cvxpy
    om = copy.copy(om0)

    results_cvxpy = np.zeros((num_steps+1, om.n))
    results_cvxpy[0] = om.x

    total_time = 0
    for step_i in range(num_steps):
        z_cvx = cp.Variable(inexact_solver.z.shape[0])

        obj = cp.Minimize( cp.quad_form(z_cvx, exact_solver.H) + cp.matmul(z_cvx.T, exact_solver.g) )
        cons = [
            cp.matmul( exact_solver.P, z_cvx ) <= exact_solver.h,
            cp.matmul( exact_solver.C, z_cvx ) == exact_solver.b,
        ]

        prob = cp.Problem( obj, cons )
        st = time.time()
        prob.solve()
        total_time += time.time() - st

        u = z_cvx.value[:om.m]

        om.step(u)
        results_cvxpy[step_i+1] = om.x

        if (step_i + 1) % 50 == 0:
            print('cvxpy completed step %d/%d' % (step_i+1,num_steps))
    rate = total_time/num_steps
    print('cvxpy for T=%d took %f seconds per step (%f Hz)' % (T, rate, 1/rate))
    rates[4,num_t] = rate

    print('Problem size')
    print('z: ' + str(inexact_solver.z.shape[0]))
    print('P: ' + str(inexact_solver.P.shape[0]))
    print('C: ' + str(inexact_solver.C.shape[0]))

fig, ax = plt.subplots()
ax.plot(Ts, rates[0], label='(1)')
ax.plot(Ts, rates[1], label='(2)')
ax.plot(Ts, rates[2], label='(3)')
ax.plot(Ts, rates[3], label='(4)')
ax.plot(Ts, rates[4], label='(5)')
ax.set_xlabel('Horizon Length')
ax.set_ylabel('Solution Time (s)')
ax.set_title('Effect of Horizon Length of Speed of Solution')
ax.legend()
plt.savefig('rates.png')
plt.show()