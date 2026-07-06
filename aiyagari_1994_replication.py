# ============================================================
# Imports
# ============================================================

# Standard numerical libraries
import time
import numpy as np
import matplotlib.pyplot as plt

# Performance optimization
from numba.experimental import jitclass
from numba import jit, njit, prange, float64, int32

# Numerical optimization and interpolation
from quantecon.optimize.scalar_maximization import brent_max
from interpolation import interp



# ============================================================
# Aiyagari Model Class
# ============================================================
# Stores all structural parameters defining the household and
# firm environment. Using a jitclass allows Numba to compile
# the model efficiently for faster numerical computation.

data_type = [
    ('bet', float64),
    ('mu', float64),
    ('b', float64),
    ('alph', float64),
    ('delt', float64),
    ('Z', float64[:]),
    ('Pi', float64[:,:])
]

@jitclass(data_type)
class aiyagari94:
    """
    Container for the structural parameters of the Aiyagari (1994) economy.
 
    Bundles the household's preference parameters, the borrowing limit,
    the firm's production parameters, and the exogenous labor productivity
    process into a single Numba jitclass so that it can be passed
    efficiently into JIT-compiled functions (e.g. the Bellman objective
    and value function iteration).
 
    Parameters
    ----------
    bet : float
        Household's discount factor, beta (0 < bet < 1).
    mu : float
        Coefficient of relative risk aversion (CRRA utility parameter).
    b : float
        Ad hoc (exogenous) borrowing limit.
    alph : float
        Capital share in the firm's Cobb-Douglas production function.
    delt : float
        Capital depreciation rate.
    Z : np.ndarray of float64
        Grid of possible values for idiosyncratic labor productivity, z.
    Pi : np.ndarray of float64, shape (len(Z), len(Z))
        Transition probability matrix for the Markov chain governing z,
        where Pi[i, j] = P(z' = Z[j] | z = Z[i]).
    """
        def __init__(self, bet, mu, b, alph, delt, Z, Pi):
            self.bet, self.mu, self.b= bet, mu, b
            self.alph, self.delt = alph, delt
            self.Z, self.Pi = Z, Pi



# ============================================================
# Bellman Objective Function
# ============================================================
# Computes the value of choosing next-period assets (kp)
# given the current state (resources x and productivity z).
#
# The function evaluates:
#   current utility
# + discounted expected continuation value
#
# This objective is repeatedly maximized during value function
# iteration.

@njit
def bellman_objective(kp, r, w, ix, iz, x_vec, V_mat, aiyagari94_class):
    """
    Evaluate the household's Bellman objective for a candidate choice of
    next-period assets.
 
    Computes current-period CRRA utility from consumption plus the
    discounted expected continuation value, given the household is at
    total-resources grid point x_vec[ix] with productivity Z[iz] and
    chooses next-period (transformed) assets kp.
 
    Parameters
    ----------
    kp : float
        Candidate choice of next-period assets, k-hat'.
    r : float
        Interest rate (net return on capital).
    w : float
        Wage rate.
    ix : int
        Index of the current total-resources grid point (x_vec[ix] = x).
    iz : int
        Index of the current productivity state (aiyagari94_class.Z[iz] = z).
    x_vec : np.ndarray of float64
        Grid of total resources (state space for x).
    V_mat : np.ndarray of float64, shape (len(x_vec), len(Z))
        Current guess of the value function, V_mat[i, j] = V(x_vec[i], Z[j]).
    aiyagari94_class : aiyagari94
        Instance holding the model's structural parameters.
 
    Returns
    -------
    float
        The value of choosing kp: current utility plus discounted
        expected continuation value. Returns -1e100 if the implied
        consumption is non-positive (infeasible choice).
    """
      bet = aiyagari94_class.bet
      mu = aiyagari94_class.mu
      b = aiyagari94_class.b
      Z = aiyagari94_class.Z
      Pi = aiyagari94_class.Pi

      x = x_vec[ix]
      z = Z[iz]

      phi = min(b,w*Z[0]/r)

      c = x - kp
      if c <= 0:
        return -1e100

      u = (c**(1-mu)-1) / (1-mu)

      EV = 0
      for j in range(len(Z)):
        xp = w*Z[j] + (1+r)*kp - r*phi
        xp = min(max(xp,x_vec[0]), x_vec[-1])
        EV += Pi[iz,j]*interp(x_vec, V_mat[:,j], xp)

      W = u + bet*EV
      return W



# ============================================================
# Value Function Iteration
# ============================================================
# Solves the household optimization problem by repeatedly
# updating the Bellman equation until convergence.
#
# Outputs:
#   - Optimal value function
#   - Optimal savings policy function

@njit(parallel=True)
def update_value_function(r, w, x_vec, Vold_mat, aiyagari94_class):
    """
    Perform one Bellman-operator update of the value function.
 
    For every point on the (total resources, productivity) grid, searches
    over all feasible next-period asset choices on x_vec and picks the one
    that maximizes the Bellman objective. The search is parallelized over
    the total-resources grid using Numba's prange.
 
    Parameters
    ----------
    r : float
        Interest rate.
    w : float
        Wage rate.
    x_vec : np.ndarray of float64
        Grid of total resources; also used as the discrete choice set
        for next-period assets.
    Vold_mat : np.ndarray of float64, shape (len(x_vec), len(Z))
        Value function from the previous iteration.
    aiyagari94_class : aiyagari94
        Instance holding the model's structural parameters.
 
    Returns
    -------
    Vnew_mat : np.ndarray of float64, shape (len(x_vec), len(Z))
        Updated value function after one Bellman iteration.
    Gnew_kp_mat : np.ndarray of float64, shape (len(x_vec), len(Z))
        Corresponding optimal next-period asset choice for each grid point.
    """
        Nx = len(x_vec)
        Nz = len(aiyagari94_class.Z)

        Vnew_mat = np.empty((Nx, Nz), dtype=np.float64)
        Gnew_kp_mat = np.empty((Nx, Nz), dtype=np.float64)

        Z = aiyagari94_class.Z

        for ix in prange(Nx):
            x = x_vec[ix]

            for iz in range(Nz):

                best_val = -1e300
                best_kp = 0.0

                for ikp in range(Nx):

                    kp = x_vec[ikp]

                    if kp > x:
                        break

                    val = bellman_objective(kp, r, w, ix, iz, x_vec, Vold_mat, aiyagari94_class)

                    if val > best_val:
                        best_val = val
                        best_kp = kp

                Vnew_mat[ix, iz] = float(best_val)
                Gnew_kp_mat[ix, iz] = float(best_kp)

        return Vnew_mat, Gnew_kp_mat

@njit
def value_function_iteration(r, w, x_vec, V0_mat, aiyagari94_class, eps_v, max_iter, display):
    """
    Solve the household's dynamic programming problem via value function
    iteration.
 
    Repeatedly applies the Bellman operator (update_value_function) until
    the sup-norm distance between successive value functions falls below
    a tolerance, or a maximum number of iterations is reached.
 
    Parameters
    ----------
    r : float
        Interest rate.
    w : float
        Wage rate.
    x_vec : np.ndarray of float64
        Grid of total resources.
    V0_mat : np.ndarray of float64, shape (len(x_vec), len(Z))
        Initial guess for the value function.
    aiyagari94_class : aiyagari94
        Instance holding the model's structural parameters.
    eps_v : float
        Convergence tolerance on the sup-norm of successive value
        function iterates.
    max_iter : int
        Maximum number of iterations to perform.
    display : bool
        Reserved for optional iteration logging (currently unused).
 
    Returns
    -------
    V_mat : np.ndarray of float64, shape (len(x_vec), len(Z))
        Converged (or final) value function.
    G_kp_mat : np.ndarray of float64, shape (len(x_vec), len(Z))
        Optimal next-period asset policy function associated with V_mat.
    """
      V_mat = V0_mat.copy()
      iter_count = 0
      stop_crit = eps_v + 1

      while (stop_crit > eps_v) and (iter_count < max_iter):
        Vnew_mat, G_kp_mat = update_value_function(r, w, x_vec, V_mat, aiyagari94_class)
        stop_crit = np.max(np.abs(Vnew_mat - V_mat))
        V_mat = Vnew_mat.copy()
        iter_count += 1

      return V_mat, G_kp_mat




# ============================================================
# Household Simulation
# ============================================================
# Simulates household decisions using the computed policy
# function and a sequence of productivity shocks.
#
# Returns simulated paths for:
#   - Total resources
#   - Asset holdings

def simulate_households(r, w, zt, x_vec, V0_mat, max_iter_v, eps_v, aiyagari94_class):
    """
    Solve the household problem and simulate a panel path given a sequence
    of productivity realizations.
 
    First solves for the value and policy functions via value function
    iteration at prices (r, w), then uses the policy function to simulate
    the implied paths of total resources and (transformed) asset holdings
    along the exogenous productivity path zt.
 
    Parameters
    ----------
    r : float
        Interest rate.
    w : float
        Wage rate.
    zt : np.ndarray of float64
        Simulated sample path of the exogenous productivity process.
    x_vec : np.ndarray of float64
        Grid of total resources.
    V0_mat : np.ndarray of float64, shape (len(x_vec), len(Z))
        Initial guess for the value function.
    max_iter_v : int
        Maximum number of value function iterations.
    eps_v : float
        Convergence tolerance for value function iteration.
    aiyagari94_class : aiyagari94
        Instance holding the model's structural parameters.
 
    Returns
    -------
    V_mat : np.ndarray of float64, shape (len(x_vec), len(Z))
        Converged value function at prices (r, w).
    kp_hat_mat : np.ndarray of float64, shape (len(x_vec), len(Z))
        Optimal next-period (transformed) asset policy function.
    xt : np.ndarray of float64, shape (len(zt),)
        Simulated path of total resources.
    kt_hat : np.ndarray of float64, shape (len(zt),)
        Simulated path of (transformed) asset holdings.
    """
      V_mat, kp_hat_mat = value_function_iteration(r, w, x_vec, V0_mat, aiyagari94_class, eps_v, max_iter_v, False)

      b = aiyagari94_class.b
      Z = aiyagari94_class.Z
      Nsim = len(zt)
      xt = np.zeros(Nsim)
      kt_hat = np.zeros(Nsim)

      phi = min(b,w*Z[0]/r)
      xt[0] = w*zt[0] + (1+r)*0 - r*phi
      kt_hat[0] = 0
      Z = aiyagari94_class.Z

      for t in range(Nsim-1):
        iz = np.argmin(np.abs(Z-zt[t]))
        kt_hat[t+1] = interp(x_vec, kp_hat_mat[:,iz], xt[t])
        xt[t+1] = w*zt[t+1] + (1+r)*kt_hat[t+1] - r*phi

      return V_mat, kp_hat_mat, xt, kt_hat



# ============================================================
# Markov Chain Utilities
# ============================================================
# Generates stationary distributions and simulated productivity
# paths used throughout the equilibrium computation.

def draw_markov_state(rand_num,cdf):
    """
    Draw a discrete Markov chain state given a uniform random draw and a
    cumulative distribution function.
 
    Finds the smallest index i such that cdf[i] >= rand_num, which
    corresponds to sampling from the discrete distribution implied by cdf.
 
    Parameters
    ----------
    rand_num : float
        A draw from a Uniform(0, 1) distribution.
    cdf : np.ndarray of float64
        Cumulative distribution function over the discrete state space.
 
    Returns
    -------
    int
        Index of the sampled state.
    """
        i = 0
        while rand_num>cdf[i]:
            i+=1
        return i

class markov_chain:
    """
    Utilities for working with a discrete-state Markov chain.
 
    Wraps a state space and transition matrix and provides methods to
    compute the stationary distribution, propagate an initial distribution
    forward in time, and simulate a sample path of the chain.
 
    Parameters
    ----------
    s_vec : np.ndarray
        Vector of possible states.
    tran_mat : np.ndarray, shape (len(s_vec), len(s_vec))
        Transition probability matrix, where tran_mat[i, j] is the
        probability of moving from state i to state j.
 
    Attributes
    ----------
    S : np.ndarray
        The state space (same as s_vec).
    Pi : np.ndarray
        The transition matrix (same as tran_mat).
    p : np.ndarray
        The stationary distribution, set after calling stationary_dist().
    """
        def __init__(self, s_vec, tran_mat):
            self.S, self.Pi = s_vec, tran_mat

        def stationary_dist(self):
            """
            Compute the stationary distribution of the Markov chain.
 
            Approximates the stationary distribution by raising the
            transpose of the transition matrix to a large power (N_inf)
            and applying it to an arbitrary initial distribution. Stores
            the result in self.p.
 
            Returns
            -------
            None
                The stationary distribution is stored as the instance
                attribute `self.p`.
            """
                # step # 1: initialize p0
                p0 = np.zeros(len(self.S))
                p0[0]= 1

                # step # 2: compute transpose
                M = self.Pi.T

                # step # 3: define infinity
                N_inf = 1000

                # step # 4: compute stationary dist
                p = np.linalg.matrix_power(M,N_inf) @ p0

                # step # 5: store p as attribute in class
                self.p = p


        def tran_uncon_dists(self,p0,Nt):
            """
            Compute the sequence of unconditional distributions implied by
            repeatedly applying the transition matrix.
 
            Parameters
            ----------
            p0 : np.ndarray
                Initial distribution over states at t=0.
            Nt : int
                Number of periods (including t=0) to compute.
 
            Returns
            -------
            np.ndarray, shape (len(self.S), Nt)
                Matrix whose t-th column is the distribution over states at
                period t.
            """
                # step 1: initialize matrix to store sequence of distributions
                pt_mat = np.zeros((len(self.S),Nt))

                # step 2: starting period
                pt_mat[:,0] = p0

                # step 3: compute distribution for each period t=1...Nt
                for t in range(1,Nt):
                    pt_mat[:,t] = self.Pi.T @ pt_mat[:,t-1]

                # Last step: return sequence of distributions
                return pt_mat

        def sample_path(self,p0,Nt):
            """
            Simulate a sample path of the Markov chain.
 
            Draws an initial state from p0, then repeatedly draws subsequent
            states according to the transition matrix Pi.
 
            Parameters
            ----------
            p0 : np.ndarray
                Initial distribution over states used to draw the state at
                t=0 (typically the stationary distribution).
            Nt : int
                Number of periods to simulate.
 
            Returns
            -------
            np.ndarray, shape (Nt,)
                Simulated sample path of realized state values (in levels,
                not indices).
            """
                # Step 1:
                rand_unif = np.random.uniform(low=0.,high=1.,size=Nt)

                # Step 2&4a:
                cdf = np.cumsum(p0)

                # Step 3:
                Pi_cdf = np.cumsum(self.Pi,axis=1)

                # Step 4b:
                x = np.zeros(Nt)

                # Step 5:
                for t in range(Nt):
                    # Get realization
                    s = draw_markov_state(rand_num=rand_unif[t],cdf=cdf)

                    # Transform to actual state value
                    x[t] =  self.S[s]

                    # Update cdf given state realization
                    cdf = Pi_cdf[s,:]

                return x



# ============================================================
# General Equilibrium Solver
# ============================================================
# Uses bisection to find the equilibrium interest rate.
#
# At each candidate interest rate:
#   1. Solve the household problem.
#   2. Simulate household savings.
#   3. Compute aggregate capital supply.
#   4. Compare supply with firm demand.
#
# The algorithm stops when excess demand is approximately zero.

def compute_equilibrium(r0, zt, x_vec, V0_mat, max_iter_v, eps_v, max_iter_r, eps_r, aiyagari94_class):
    """
    Solve for the stationary general equilibrium interest rate via
    bisection on the capital market.
 
    At each candidate interest rate r, computes the firm's capital demand
    Kd from its first-order condition, solves and simulates the household
    problem to obtain aggregate capital supply Ks, and updates the
    bisection bracket based on the sign of excess demand (Kd - Ks) until
    the bracket width falls below a tolerance.
 
    Parameters
    ----------
    r0 : float
        Initial guess for the equilibrium interest rate.
    zt : np.ndarray of float64
        Simulated sample path of the exogenous productivity process,
        used to compute aggregate capital supply via simulation.
    x_vec : np.ndarray of float64
        Grid of total resources.
    V0_mat : np.ndarray of float64, shape (len(x_vec), len(Z))
        Initial guess for the value function.
    max_iter_v : int
        Maximum number of value function iterations per household solve.
    eps_v : float
        Convergence tolerance for value function iteration.
    max_iter_r : int
        Maximum number of bisection iterations.
    eps_r : float
        Convergence tolerance on the bisection bracket width.
    aiyagari94_class : aiyagari94
        Instance holding the model's structural parameters.
 
    Returns
    -------
    r_e : float
        Equilibrium interest rate.
    V_mat : np.ndarray of float64, shape (len(x_vec), len(Z))
        Value function evaluated at the equilibrium prices.
    kp_hat_mat : np.ndarray of float64, shape (len(x_vec), len(Z))
        Policy function evaluated at the equilibrium prices.
    xt : np.ndarray of float64, shape (len(zt),)
        Simulated path of total resources at the equilibrium prices.
    kt_hat : np.ndarray of float64, shape (len(zt),)
        Simulated path of (transformed) asset holdings at the
        equilibrium prices.
    """
      alph = aiyagari94_class.alph
      delt = aiyagari94_class.delt
      bet = aiyagari94_class.bet
      b = aiyagari94_class.b
      Z = aiyagari94_class.Z
      Pi = aiyagari94_class.Pi

      K0 = ((r0+delt)/alph)**(1/(alph-1))
      w0 = (1-alph)*K0**(alph)

      phi0 = min(b,w0*Z[0]/r0)
      V_mat, kp_hat_mat, xt, kt_hat = simulate_households(r0, w0, zt, x_vec, V0_mat, max_iter_v, eps_v, aiyagari94_class)
      Ks = np.mean(kt_hat) - phi0
      r1 = alph * Ks**(alph-1) - delt

      rL = min(r0,r1)
      rR = max(r0,r1)

      stop_crit = eps_r + 1
      iter_count = 1

      while (stop_crit > eps_r) and (iter_count < max_iter_r):
        r = (rL + rR)/2
        Kd = ((r+delt)/alph)**(1/(alph-1))
        w = (1-alph)*Kd**alph

        phi = min(b,w*Z[0]/r)
        V_mat, kp_hat_mat, xt, kt_hat = simulate_households(r, w, zt, x_vec, V_mat, max_iter_v, eps_v, aiyagari94_class)

        Ks = np.mean(kt_hat) - phi
        excess_demand = Kd - Ks

        if excess_demand < 0:
          rR = r
        else:
          rL = r

        stop_crit = abs(rR-rL)
        iter_count += 1

      r_e = (rL + rR)/2

      return r_e, V_mat, kp_hat_mat, xt, kt_hat



# ============================================================
# Model Calibration and Grid Construction
# ============================================================

from quantecon.markov.approximation import tauchen
# AR(1) process for labor (z)
rho = .6
sig = .2

# Approximate AR(1) process with Markov chain (Tauchen)
m_stdevs = 3
n_states = 7
mc_z = tauchen(rho=rho, sigma=sig*(1-rho**2)**(1/2), mu=0, n_std=m_stdevs, n=n_states)



# Markov chain state and transition probability
Z = np.exp(mc_z._state_values)
Pi = mc_z.P

# Household's parameters
β = .96
μ = 5.
b = 0.

# Firm's parameters
α = .36
δ = .08

# Create economy
e1 = aiyagari94(bet=β, mu=μ, b=b, alph=α, delt=δ, Z=Z, Pi=Pi)



# Minimum wage rate (corresponds to r=(1-bet)/bet)
λ = (1-β)/β
k_λ = (α/(λ+δ))**(1/(1-α))     # from firm's FOC on capital (k that corresponds to r=(1-bet)/bet)
w_λ = (1-α)*k_λ**(α)           # from firm's FOC on labor (w that corresponds to r=(1-bet)/bet)

# Total resources grid parameters
Nx = 200
x_max = δ**(1/(α-1))
x_min = w_λ*np.min(Z)

# Exponentially spaced grid
log_x_vec = np.linspace(np.log(x_min),np.log(x_max),Nx)
x_vec = np.exp(log_x_vec)



# ============================================================
# Solve Household Problem at a Reference Interest Rate
# ============================================================

# VFI convergence parameters
eps_v, max_iter_v = 1e-4, 3000
r = 0.035857
K = ((r + δ) / α) ** (1 / (α - 1))
w = (1 - α) * K ** α

V0_mat = np.zeros((Nx,len(Z)))
start_time = time.time()
V_mat, G_kp_mat = value_function_iteration(r, w, x_vec, V0_mat, e1, eps_v, max_iter_v, False)
end_time = time.time()
print('VFI took: ' + str(end_time-start_time) + ' seconds')



# ============================================================
# Figure 1: Household Policy Functions
# ============================================================
# Plot optimal savings decisions for low- and high-productivity
# households.

plt.figure(figsize=(10, 6))
plt.plot(x_vec, G_kp_mat[:, 0], label=f'$z_{{min}}$ = {Z[0]:.3f}', linewidth=2)
plt.plot(x_vec, G_kp_mat[:, -1], label=f'$z_{{max}}$ = {Z[-1]:.3f}', linewidth=2)
plt.plot(x_vec, x_vec, 'k--', alpha=0.5, label='45° line')
plt.xlabel('Total Resources', fontsize=10)
plt.ylabel("Asset Holdings", fontsize=10)
plt.title('Optimal Policy Functions', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()



# ============================================================
# Aggregate Capital Demand and Supply Across Interest Rates
# ============================================================

# VFI convergence parameters
eps_v, max_iter_v = 1e-4, 3000

# Grid of interest rates
Nr = 10  # Can change to 20+ if it doesn't take too long to run
r_min = 1e-10
r_max = .95*λ
r_vec = np.linspace(r_min,r_max,Nr)

Nsim = 20000
mc = markov_chain(Z, Pi)
mc.stationary_dist()
zt = mc.sample_path(mc.p, Nsim)
Kd_vec = np.zeros(Nr)
Ks_vec = np.zeros(Nr)

V0_mat = np.zeros((len(x_vec), len(Z)))
for i, r in enumerate(r_vec):
  Kd = ((r + δ) / α) ** (1 / (α - 1))
  w = (1 - α) * Kd ** α

  phi = min(b, w * Z[0] / r)
  V_mat, kp_hat_mat, xt, kt_hat = simulate_households(r, w, zt, x_vec, V0_mat, max_iter_v, eps_v, e1)
  Ks = np.mean(kt_hat) - phi

  Kd_vec[i] = Kd
  Ks_vec[i] = Ks

  V0_mat = V_mat.copy()



# ============================================================
# Figure 2: Aggregate Capital Market
# ============================================================
# Compare aggregate capital demand from firms with aggregate
# capital supply from households across interest rates.

plt.figure(figsize=(10, 6))
plt.plot(Kd_vec, r_vec, 'b-', linewidth=2, label='Capital Demand ($K_d$)')
plt.plot(Ks_vec, r_vec, 'r-', linewidth=2, label='Capital Supply ($K_s$)')
plt.ylabel('Net Return to Capital', fontsize=12)
plt.xlabel('Capital', fontsize=12)
plt.title('Aggregate Capital Demand and Supply', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()



# ============================================================
# Solve for General Equilibrium
# ============================================================

# VFI convergence parameters
eps_v, max_iter_v = 1e-4, 3000

# Bisection convergence parameters
eps_r, max_iter_r = 1e-6, 500

# Initial guess for r
r0 = .95*λ

Nsim = 20000
mc = markov_chain(Z, Pi)
mc.stationary_dist()
zt = mc.sample_path(mc.p, Nsim)
V0_mat = np.zeros((len(x_vec), len(Z)))
r_e, V_mat, kp_hat_mat, xt, kt_hat = compute_equilibrium(r0, zt, x_vec, V0_mat, max_iter_v, eps_v, max_iter_r, eps_r, e1)
print(f"Equilibrium interest rate: r = {r_e:.4f}")
print(f"Compare to Aiyagari (1994) Table II: r ≈ 0.0359")



# ============================================================
# Inequality Measures: Consumption, Wealth, and Income
# ============================================================

K_e = ((r_e + δ) / α) ** (1 / (α - 1))
w_e = (1 - α) * K_e ** α
phi_e = min(b, w_e * Z[0] / r_e)

ct = xt[:-1] - kt_hat[1:]
kt = kt_hat - phi_e
yt = w_e * zt + r_e * kt

def lorenz_curve(data):
    """
    Compute the Lorenz curve for a distribution of values.
 
    Parameters
    ----------
    data : np.ndarray
        Sample of values (e.g. consumption, wealth, or income) from
        which to compute the empirical Lorenz curve.
 
    Returns
    -------
    p : np.ndarray
        Cumulative population share, from 0 to 1.
    lorenz : np.ndarray
        Corresponding cumulative share of the variable held by the
        bottom p fraction of the population.
    """
        data = np.sort(data)
        n = len(data)
        cum_data = np.cumsum(data)
        lorenz = cum_data / cum_data[-1]
        lorenz = np.insert(lorenz, 0, 0)
        p = np.linspace(0, 1, len(lorenz))
        return p, lorenz

def gini_coefficient(data):
    """
    Compute the Gini coefficient of a distribution of values.
 
    Uses the standard closed-form expression based on the sorted sample:
    G = (2 * sum(i * x_i)) / (n * sum(x)) - (n + 1) / n, for i = 1, ..., n
    on ascending-sorted data.
 
    Parameters
    ----------
    data : np.ndarray
        Sample of values (e.g. consumption, wealth, or income).
 
    Returns
    -------
    float
        The Gini coefficient, between 0 (perfect equality) and 1
        (maximal inequality).
    """
        sorted_data = np.sort(data)
        n = len(sorted_data)
        cumsum = np.cumsum(sorted_data)

        gini = (2 * np.sum((np.arange(1, n + 1)) * sorted_data)) / (n * cumsum[-1]) - (n + 1) / n

        return gini

pop_c, lorenz_c = lorenz_curve(ct)
pop_k, lorenz_k = lorenz_curve(kt)
pop_y, lorenz_y = lorenz_curve(yt)

gini_c = gini_coefficient(ct)
gini_k = gini_coefficient(kt)
gini_y = gini_coefficient(yt)


# ============================================================
# Figure 3: Lorenz Curves
# ============================================================
# Measure inequality in equilibrium consumption, wealth,
# and income using Lorenz curves and Gini coefficients.

plt.figure(figsize=(10, 6))
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Equality')
plt.plot(pop_c, lorenz_c, linewidth=2, label=f'Consumption (Gini = {gini_c:.3f})')
plt.plot(pop_k, lorenz_k, linewidth=2, label=f'Wealth (Gini = {gini_k:.3f})')
plt.plot(pop_y, lorenz_y, linewidth=2, label=f'Income (Gini = {gini_y:.3f})')
plt.xlabel('Cumulative Share of Population', fontsize=12)
plt.ylabel('Cumulative Share of Variable', fontsize=12)
plt.title('Lorenz Curves', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
