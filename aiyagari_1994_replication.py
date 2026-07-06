```python
# Load standard libraries
import time
import numpy as np
import matplotlib.pyplot as plt
from numba.experimental import jitclass
from numba import jit, njit, prange, float64, int32
# Load add-on libraries
from quantecon.optimize.scalar_maximization import brent_max
from interpolation import interp
```

```python
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
    def __init__(self, bet, mu, b, alph, delt, Z, Pi):
        self.bet, self.mu, self.b= bet, mu, b
        self.alph, self.delt = alph, delt
        self.Z, self.Pi = Z, Pi
```

```python
@njit
def obj_fnc(kp, r, w, ix, iz, x_vec, V_mat, aiyagari94_class):
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

@njit(parallel=True)
def Vnew_mat_fnc(r, w, x_vec, Vold_mat, aiyagari94_class):
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

                val = obj_fnc(kp, r, w, ix, iz, x_vec, Vold_mat, aiyagari94_class)

                if val > best_val:
                    best_val = val
                    best_kp = kp

            Vnew_mat[ix, iz] = float(best_val)
            Gnew_kp_mat[ix, iz] = float(best_kp)

    return Vnew_mat, Gnew_kp_mat

@njit
def VFI_fnc(r, w, x_vec, V0_mat, aiyagari94_class, eps_v, max_iter, display):
  V_mat = V0_mat.copy()
  iter_count = 0
  stop_crit = eps_v + 1

  while (stop_crit > eps_v) and (iter_count < max_iter):
    Vnew_mat, G_kp_mat = Vnew_mat_fnc(r, w, x_vec, V_mat, aiyagari94_class)
    stop_crit = np.max(np.abs(Vnew_mat - V_mat))
    V_mat = Vnew_mat.copy()
    iter_count += 1

  return V_mat, G_kp_mat
```

```python
def simulate_fnc(r, w, zt, x_vec, V0_mat, max_iter_v, eps_v, aiyagari94_class):
  V_mat, kp_hat_mat = VFI_fnc(r, w, x_vec, V0_mat, aiyagari94_class, eps_v, max_iter_v, False)

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
```

```python
def discrete_realization(rand_num,cdf):
    i = 0
    while rand_num>cdf[i]:
        i+=1
    return i

class markov_chain:
    def __init__(self, s_vec, tran_mat):
        self.S, self.Pi = s_vec, tran_mat

    def stationary_dist(self):
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
            s = discrete_realization(rand_num=rand_unif[t],cdf=cdf)

            # Transform to actual state value
            x[t] =  self.S[s]

            # Update cdf given state realization
            cdf = Pi_cdf[s,:]

        return x
```

```python
def compute_equil(r0, zt, x_vec, V0_mat, max_iter_v, eps_v, max_iter_r, eps_r, aiyagari94_class):
  alph = aiyagari94_class.alph
  delt = aiyagari94_class.delt
  bet = aiyagari94_class.bet
  b = aiyagari94_class.b
  Z = aiyagari94_class.Z
  Pi = aiyagari94_class.Pi

  K0 = ((r0+delt)/alph)**(1/(alph-1))
  w0 = (1-alph)*K0**(alph)

  phi0 = min(b,w0*Z[0]/r0)
  V_mat, kp_hat_mat, xt, kt_hat = simulate_fnc(r0, w0, zt, x_vec, V0_mat, max_iter_v, eps_v, aiyagari94_class)
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
    V_mat, kp_hat_mat, xt, kt_hat = simulate_fnc(r, w, zt, x_vec, V_mat, max_iter_v, eps_v, aiyagari94_class)

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
```

```python
from quantecon.markov.approximation import tauchen
# AR(1) process for labor (z)
rho = .6
sig = .2

# Approximate AR(1) process with Markov chain (Tauchen)
m_stdevs = 3
n_states = 7
mc_z = tauchen(rho=rho, sigma=sig*(1-rho**2)**(1/2), mu=0, n_std=m_stdevs, n=n_states)
```

```python
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
```

```python
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
```

```python
# VFI convergence parameters
eps_v, max_iter_v = 1e-4, 3000
r = 0.035857
K = ((r + δ) / α) ** (1 / (α - 1))
w = (1 - α) * K ** α

# YOUR CODE HERE
V0_mat = np.zeros((Nx,len(Z)))
start_time = time.time()
V_mat, G_kp_mat = VFI_fnc(r, w, x_vec, V0_mat, e1, eps_v, max_iter_v, False)
end_time = time.time()
print('VFI took: ' + str(end_time-start_time) + ' seconds')
```

```python
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
```

```python
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
  V_mat, kp_hat_mat, xt, kt_hat = simulate_fnc(r, w, zt, x_vec, V0_mat, max_iter_v, eps_v, e1)
  Ks = np.mean(kt_hat) - phi

  Kd_vec[i] = Kd
  Ks_vec[i] = Ks

  V0_mat = V_mat.copy()
```

```python
plt.figure(figsize=(10, 6))
plt.plot(Kd_vec, r_vec, 'b-', linewidth=2, label='Capital Demand ($K_d$)')
plt.plot(Ks_vec, r_vec, 'r-', linewidth=2, label='Capital Supply ($K_s$)')
plt.ylabel('Net Return to Capital', fontsize=12)
plt.xlabel('Capital', fontsize=12)
plt.title('Aggregate Capital Demand and Supply', fontsize=14)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
```

```python
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
r_e, V_mat, kp_hat_mat, xt, kt_hat = compute_equil(r0, zt, x_vec, V0_mat, max_iter_v, eps_v, max_iter_r, eps_r, e1)
print(f"Equilibrium interest rate: r = {r_e:.4f}")
print(f"Compare to Aiyagari (1994) Table II: r ≈ 0.0359")
```

```python
K_e = ((r_e + δ) / α) ** (1 / (α - 1))
w_e = (1 - α) * K_e ** α
phi_e = min(b, w_e * Z[0] / r_e)

ct = xt[:-1] - kt_hat[1:]
kt = kt_hat - phi_e
yt = w_e * zt + r_e * kt

def lorenz_curve(data):
    data = np.sort(data)
    n = len(data)
    cum_data = np.cumsum(data)
    lorenz = cum_data / cum_data[-1]
    lorenz = np.insert(lorenz, 0, 0)
    p = np.linspace(0, 1, len(lorenz))
    return p, lorenz

def gini_coefficient(data):
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
```

```python
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
```
