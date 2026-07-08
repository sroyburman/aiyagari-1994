# Aiyagari (1994) Replication

A Python implementation of the Aiyagari (1994) heterogeneous-agent incomplete markets model using dynamic programming, value function iteration, simulation, and equilibrium computation.

## Overview

The Aiyagari (1994) model is a foundational heterogeneous-agent macroeconomic model where households face idiosyncratic labor income risk and save to self-insure against future uncertainty. Unlike the Huggett model, the Aiyagari economy includes aggregate capital accumulation and production, allowing the equilibrium interest rate, wage, and capital stock to be determined endogenously.

This project focuses on the computational solution of the model. The implementation solves the household savings problem, simulates household asset paths, computes aggregate capital supply, compares it to firm capital demand, and solves for the equilibrium interest rate using bisection.

## Objectives

The goals of this project are to:

1. Solve the household dynamic programming problem.
2. Compute optimal savings policy functions.
3. Simulate household income and asset paths.
4. Construct aggregate capital supply.
5. Compare capital supply with firm capital demand.
6. Solve for the general equilibrium interest rate.
7. Analyze inequality using Lorenz curves and Gini coefficients.

## Model

Households choose next period asset holdings to maximize expected lifetime utility:

$$
V(x, z; r, w) = \max_{c, \hat{k}'} \; \frac{c^{1-\mu} - 1}{1-\mu} + \beta \sum_{z' \in \mathbf{Z}} \pi(z'|z) \, V(x', z'; r, w)
$$

subject to

$$
\phi \equiv \min\{b,\frac{wz^1}{r}\}
$$

$$
\begin{aligned}
c &\ge 0, \\
\hat{k}' &\ge 0, \\
c + \hat{k}' &= x, \\
x' &= wz' + (1+r)\hat{k}' - r\phi.
\end{aligned}
$$

Labor productivity follows the finite-state Markov process

$$
z, z' \in Z = \{z^1,\dots,z^N\}
$$

with transition matrix

$$
\Pi =
\begin{bmatrix}
\pi_{11} & \pi_{12} & \cdots & \pi_{1N} \\
\vdots & \vdots & \ddots & \vdots \\
\pi_{i1} & \pi_{ij} & \cdots & \pi_{iN} \\
\vdots & \vdots & \ddots & \vdots \\
\pi_{N1} & \pi_{Nj} & \cdots & \pi_{NN}
\end{bmatrix}.
$$

Here:

- $x$ is total resources,
- $k'$ is next-period asset holdings,
- $z$ is labor productivity,
- $w$ is the wage,
- $r$ is the interest rate,
- $\phi$ is the borrowing constraint,
- $\beta$ is the discount factor,
- $\mu$ is the coefficient of relative risk aversion.

Labor productivity follows a discretized AR(1) process approximated with Tauchen's method.

## Parameters

Benchmark household parameters:

| Parameter | Value |
|----------|------:|
| Discount factor $\beta$ | 0.96 |
| Risk aversion $\mu$ | 5.0 |
| Borrowing constraint $b$ | 0.0 |

Firm parameters:

| Parameter | Value |
|----------|------:|
| Capital share $\alpha$ | 0.36 |
| Depreciation $\delta$ | 0.08 |

Labor productivity process:

```python
rho = 0.6
sigma = 0.2
n_states = 7
```

## Computational Approach

This implementation consists of several main components.

### 1. Productivity Process

Labor productivity is modeled as an AR(1) process and discretized into a finite-state Markov chain using Tauchen's method.

### 2. Household Problem

Given an interest rate and wage, the household dynamic programming problem is solved using value function iteration.

The code uses:

- Asset/resource grids
- Interpolation
- Discrete search over next-period assets
- Numba JIT compilation
- Parallelized loops

### 3. Simulation

After solving the household problem, the model simulates a long path of productivity shocks and household asset choices. This generates simulated series' for:

- Total resources
- Asset holdings
- Consumption
- Income

### 4. Capital Supply and Demand

For a grid of interest rates, the model computes:

- Household capital supply from simulated savings behavior
- Firm capital demand from the production function

The intersection of capital supply and demand determines the general equilibrium interest rate.

### 5. Equilibrium Computation

The equilibrium interest rate is solved using bisection. At each candidate interest rate, the model solves the household problem, simulates capital holdings, and compares capital supply to firm capital demand. 

### 6. Inequality Analysis

Using the simulated equilibrium allocation, the project computes Lorenz curves and Gini coefficients for:

- Consumption
- Wealth
- Income

## Results

The project produces numerous key outputs:

1. Optimal household savings policy functions
2. Aggregate capital supply and demand curves
3. General equilibrium interest rate
4. Simulated household wealth, consumption, and income distributions
5. Lorenz curves and Gini coefficients

For the benchmark calibration, the computed equilibrium interest rate is close to the value reported in Aiyagari (1994) Table II.

## Visualizations

The repository includes depictions of:

- Optimal policy functions
- Aggregate supply and demand curves
- Lorenz curves for consumption, wealth, and income

## Skills Demonstrated

- Python programming
- Object-oriented programming
- Dynamic Programming
- Value function iteration
- Macroeconomic modeling
- Bisection root-finding
- Markov chains
- Tauchen discretization
- Interpolation
- Numba acceleration

## Packages Used

`numpy`; `matplotlib`; `numba`; `quantecon`; `interpolation`

## Potential Extensions

Possible future improvements include:

- Adding alternative borrowing constraints
- Computing different risk aversion values
- Refactoring into separate model, solver, simulation, and plotting modules
- Computing the stationary distribution directly rather than through simulation
- Adding convergence diagnostics for value function iteration

## References

Aiyagari, S. Rao. (1994). _Uninsured Idiosyncratic Risk and Aggregate Saving_. Quarterly Journal of Economics, 109(3), 659–684.
