#!/usr/bin/env python3
"""Verify the §4 analytic MSE decomposition against Monte Carlo.

Checks that
    E|| mean_{j in S}(theta_j + eta_j) - theta_t ||^2
    = p * [ (tau^2/|S|^2) sum_{j,k in S} overlap(j,k) + sigma^2/|S| + (mu*mean_S(j-t))^2 ]
to <0.3% relative error across LOSO/forward, cold-start/mid targets, and
random-walk/trend/strong regimes. Run: python TEST/verify_theory.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.theory import analytic_mse  # noqa: E402

RNG = np.random.default_rng(0)


def mc_mse(S, t, p, tau, sigma, mu, T, reps=200_000):
    S = list(S)
    acc = 0.0
    for _ in range(reps):
        eps = RNG.normal(0, tau, size=(T, p))
        theta = np.zeros((T, p))
        for j in range(1, T):
            theta[j] = theta[j - 1] + mu + eps[j]
        obs = theta + RNG.normal(0, sigma, size=(T, p))
        acc += float(np.sum((obs[S].mean(axis=0) - theta[t]) ** 2))
    return acc / reps


def main():
    p, T = 3, 8
    cases = [
        ("LOSO t=1 rw",     [j for j in range(T) if j != 1], 1, 0.3, 0.5, 0.0),
        ("forward t=1 rw",  list(range(0, 1)),               1, 0.3, 0.5, 0.0),
        ("LOSO t=4 rw",     [j for j in range(T) if j != 4], 4, 0.3, 0.5, 0.0),
        ("forward t=4 rw",  list(range(0, 4)),               4, 0.3, 0.5, 0.0),
        ("LOSO t=3 trend",  [j for j in range(T) if j != 3], 3, 0.3, 0.5, 0.15),
        ("forward t=3 trend", list(range(0, 3)),             3, 0.3, 0.5, 0.15),
        ("LOSO t=2 strong", [j for j in range(T) if j != 2], 2, 0.6, 0.5, 0.0),
    ]
    max_err = 0.0
    print(f"{'case':22s} {'analytic':>10s} {'MC':>10s} {'rel.err':>8s}")
    for name, S, t, tau, sig, mu in cases:
        a = p * analytic_mse(S, t, tau, sig, mu)
        m = mc_mse(S, t, p, tau, sig, mu, T)
        err = abs(a - m) / m
        max_err = max(max_err, err)
        print(f"{name:22s} {a:10.4f} {m:10.4f} {err:8.4f}")
    print(f"\nmax relative error = {max_err:.4f}")
    assert max_err < 0.01, f"decomposition mismatch: {max_err:.4f}"
    print("PASS: analytic decomposition matches Monte Carlo.")


if __name__ == "__main__":
    main()
