
# 1. Function Intersection

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

def system(vars):
    x, y = vars
    eq1 = x**2 + y**2 - 1 
    eq2 = x - y          
    return [eq1, eq2]

# 1) Solve using different initial guesses to capture different intersections.
sol1 = root(system, [0.5, 0.5])   
sol2 = root(system, [-0.5, -0.5]) 

print("Solution 1:", sol1.x)
print("Solution 2:", sol2.x)

# 2) Plot the two equations and their intersection points.

x_vals = np.linspace(-1.5, 1.5, 400)

y_upper = np.sqrt(np.clip(1 - x_vals**2, 0, None))
y_lower = -np.sqrt(np.clip(1 - x_vals**2, 0, None))

line_vals = x_vals

plt.figure(figsize=(6, 6))
plt.plot(x_vals, y_upper, 'b', label=r'$x^2 + y^2 = 1$ (upper)')
plt.plot(x_vals, y_lower, 'b', label=r'$x^2 + y^2 = 1$ (lower)')
plt.plot(x_vals, line_vals, 'r', label=r'$y = x$')

# Plot the solutions
plt.plot(sol1.x[0], sol1.x[1], 'ko', label='Intersection 1')
plt.plot(sol2.x[0], sol2.x[1], 'ko', label='Intersection 2')


plt.axis('equal')  
plt.xlim([-1.2, 1.2])
plt.ylim([-1.2, 1.2])
plt.xlabel('x')
plt.ylabel('y')
plt.title('Intersections of the System')
plt.legend()
plt.grid(True)

plt.show()


# 2 Job Search

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

def make_wage_distribution(w_min=10, w_max=60, alpha=100, beta_shape=50):
    wages = np.arange(w_min, w_max+1)
    # Map wage w in [w_min, w_max] to x in [0, 1].
    # E.g. x = (w - w_min)/(w_max - w_min)
    x_vals = (wages - w_min)/(w_max - w_min)
    
    # Evaluate Beta(alpha, beta_shape) pdf at each x
    pdf_vals = beta.pdf(x_vals, alpha, beta_shape)
    pmf = pdf_vals / pdf_vals.sum()
    
    return wages, pmf

def solve_finite_horizon_job_search(
    T=8,    # last period is 8, so total of 9 periods: t=0..8
    c=10,   # unemployment benefit
    beta_disc=0.9,
    wages=None,
    pmf=None
):

    if wages is None or pmf is None:
        
        wages, pmf = make_wage_distribution()
    
    # Precompute sum_{tau=0 to T-t} beta^tau for each t:
    # This factor is how many times log(w) gets counted if you accept in period t
    G = np.array([sum(beta_disc**k for k in range(T-t+1)) for t in range(T+1)])
    
    V = np.zeros(T+1)  
    r = np.zeros(T+1)  
    
    # 1) Last period T
    V[T] = np.sum(np.maximum(np.log(wages), np.log(c)) * pmf)
    # r[T]: solve G[T]*ln(r[T]) = ln(c)
    # because at T+1 there's no future, so V[T+1] = 0 in that formula
    # G[T] = sum_{k=0}^0 beta^k = 1
    r[T] = np.exp( (np.log(c) + beta_disc*0.0) / G[T] )
    
    # 2) Work backward t = T-1 down to 0
    for t in reversed(range(T)):
        # V[t] = sum_w p(w)* max( G[t]*ln(w), ln(c) + beta_disc*V[t+1] )
        accept_vals = G[t] * np.log(wages) 
        reject_val  = np.log(c) + beta_disc * V[t+1]
        
        inside_max = np.maximum(accept_vals, reject_val)
        V[t] = np.sum(inside_max * pmf)
        
        # reservation wage r[t] solves
        #     G[t]*ln(r[t]) = ln(c) + beta_disc*V[t+1]
        r[t] = np.exp((np.log(c) + beta_disc * V[t+1]) / G[t])
    
    return V, r, wages


if __name__ == "__main__":
    # Solve for T=8
    T = 8
    V, r, wages = solve_finite_horizon_job_search(T=T, c=10, beta_disc=0.9)
    
    print("Period |  Reservation Wage  | Value of Search")
    for t in range(T+1):
        print(f"{t:5d} | {r[t]:18.4f} | {V[t]:14.4f}")
    
    # Plot reservation wages over time
    plt.figure(figsize=(8,4))
    plt.plot(range(T+1), r, marker="o")
    plt.title("Reservation wages over time (finite horizon)")
    plt.xlabel("t (period)")
    plt.ylabel("r(t)")
    plt.grid(True)
    plt.show()


# 3 Market Equilibrium


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

# 1) Demand and supply functions
def D1(p1, p2):
    return 100 - 2.0*p1 + 0.5*(p2**2)

def S1(p1):
    return 20 + 0.1*(p1**2)

def D2(p1, p2):
    return 80 - 3.0*p2 + 0.2*(p1**2)

def S2(p2):
    return 30 + 0.05*(p2**2)

# 2) Homotopy system H = (H1, H2).
#    p = (p1, p2), lam in [0,1].
#    p1_ref, p2_ref are reference prices where H(...)=0 at lam=0.
p1_ref = 5.0
p2_ref = 5.0

def H(p, lam):
    p1, p2 = p
    H1 = lam*( S1(p1) - D1(p1,p2) ) + (1 - lam)*( p1 - p1_ref )
    H2 = lam*( S2(p2) - D2(p1,p2) ) + (1 - lam)*( p2 - p2_ref )
    return np.array([H1, H2])

def solve_for_prices(lam, p_init):
    sol = root(lambda p: H(p, lam), p_init, method='hybr')
    return sol.x


def main():
    # 3) Trace out solutions as lam goes from 0 to 1
    lam_values = np.linspace(0, 1, 21) 
    p_current = np.array([p1_ref, p2_ref])
    
    solutions = []
    
    for lam in lam_values:
        p_current = solve_for_prices(lam, p_current)
        solutions.append(p_current)
    
    solutions = np.array(solutions)
    p1_vals = solutions[:,0]
    p2_vals = solutions[:,1]
    
 
    print(f"Final (p1,p2) at lam=1: ({p1_vals[-1]:.4f}, {p2_vals[-1]:.4f})")

    plt.figure(figsize=(6,4))
    plt.plot(lam_values, p1_vals, 'o-', label='p1')
    plt.plot(lam_values, p2_vals, 's-', label='p2')
    plt.xlabel(r'$\lambda$')
    plt.ylabel('Prices')
    plt.title('Homotopy path for (p1, p2)')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
