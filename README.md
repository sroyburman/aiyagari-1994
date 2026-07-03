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

