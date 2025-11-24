"""
Problema 2 — Cajeros automáticos con disponibilidad variable
M/M/k + cadena de Markov discreta
"""

import numpy as np
import pandas as pd
from math import factorial
import argparse


# -------------------------------------------------------------------
# Funciones de Cálculo
# -------------------------------------------------------------------

def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """
    Calcula la distribución estacionaria π de una cadena de Markov.
    Resuelve πP = π junto con sum(π) = 1.
    """
    n_states = P.shape[0]

    A = (P.T - np.identity(n_states))
    A[-1, :] = 1.0

    b = np.zeros(n_states)
    b[-1] = 1.0

    try:
        pi = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        print("Error: Matriz singular. No se pudo calcular π.")
        return np.full(n_states, np.nan)

    return pi


def mmk_metrics(lamb: float, mu: float, k: int):
    """
    Calcula métricas estándar de M/M/k: rho, L, W.
    Si rho >= 1, retorna métricas infinitas.
    """
    mu_sys_rate = k * mu
    rho = lamb / mu_sys_rate

    if rho >= 1 or k <= 0:
        return {'rho': rho, 'L': np.inf, 'W': np.inf}

    a = lamb / mu  # parámetro carga por servidor

    sum_term = sum((a**n) / factorial(n) for n in range(k))
    queue_term = (a**k / factorial(k)) * (1 / (1 - rho))

    P0 = 1 / (sum_term + queue_term)

    # Lq fórmula estándar
    Lq = (P0 * lamb * mu * (a**k)) / (factorial(k - 1) * (mu_sys_rate - lamb)**2)

    Wq = Lq / lamb
    W = Wq + (1 / mu)
    L = lamb * W

    return {'rho': rho, 'L': L, 'W': W}


def weighted_metrics(pi, lamb, mu, states):
    """
    Métricas ponderadas por π.
    """
    data = []

    for i in range(len(pi)):
        k = states[i]
        metrics = mmk_metrics(lamb, mu, k)

        metrics['pi'] = pi[i]
        metrics['k'] = k
        metrics['Estado'] = f"S{i+1}"

        data.append(metrics)

    df = pd.DataFrame(data)

    saturated = (df['L'] == np.inf) & (df['pi'] > 1e-9)
    if saturated.any():
        Lw = np.inf
        Ww = np.inf
    else:
        Lw = np.nansum(df['L'].replace(np.inf, np.nan) * df['pi'])
        Ww = np.nansum(df['W'].replace(np.inf, np.nan) * df['pi'])

    rho_w = np.nansum(df['rho'].replace(np.inf, np.nan) * df['pi'])

    resumen = {
        'rho_ponderado': rho_w,
        'L_ponderado': Lw,
        'W_ponderado': Ww
    }

    df = df[['Estado', 'k', 'pi', 'rho', 'L', 'W']]
    return df, resumen


def simulate_markov_chain(P, days, initial_state=0, seed=None):
    """
    Simula una cadena de Markov por 'days' pasos.
    """
    if seed is not None:
        np.random.seed(seed)

    n_states = P.shape[0]
    states = np.zeros(days, dtype=int)
    states[0] = initial_state

    for t in range(1, days):
        current = states[t - 1]
        probs = P[current]
        states[t] = np.random.choice(np.arange(n_states), p=probs)

    return states


def empirical_distribution(sim_states, n_states):
    counts = np.bincount(sim_states, minlength=n_states)
    return counts / counts.sum()


# -------------------------------------------------------------------
# MAIN con argparse
# -------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Problema 2 — Cajeros automáticos M/M/k + Markov"
    )

    parser.add_argument("--lambda_", type=float, default=8.0,
                        help="Tasa de llegada λ (clientes/min). Default = 8.0")

    parser.add_argument("--mu", type=float, default=4.0,
                        help="Tasa de servicio por cajero μ (clientes/min). Default = 4.0")

    parser.add_argument("--dias", type=int, default=30,
                        help="Número de días a simular. Default = 30")

    parser.add_argument("--seed", type=int, default=None,
                        help="Semilla opcional para reproducibilidad.")

    args = parser.parse_args()

    # Estados fijos del problema
    STATES = {0: 3, 1: 2, 2: 1}

    P = np.array([
        [0.6, 0.3, 0.1],
        [0.2, 0.6, 0.2],
        [0.1, 0.2, 0.7],
    ])

    # Mostrar parámetros elegidos
    print("\n---------------------------------------")
    print("Usted eligió los siguientes parámetros:")
    print(f"λ (lambda)           : {args.lambda_}")
    print(f"μ por cajero         : {args.mu}")
    print(f"Días a simular       : {args.dias}")
    print(f"Semilla (opcional)   : {args.seed}")
    print("---------------------------------------\n")

    # (a) Distribución estacionaria
    pi = stationary_distribution(P)
    print("Distribución estacionaria π:")
    for i, p in enumerate(pi):
        print(f"  S{i+1}: {p:.4f}")

    print()

    # (b) Métricas ponderadas
    df, resumen = weighted_metrics(pi, args.lambda_, args.mu, STATES)

    print("Métricas por estado:")
    print(df.to_string(index=False))

    print("\nMétricas ponderadas por π:")
    for k, v in resumen.items():
        print(f"  {k}: {v}")

    print()

    # (c) Simulación de la cadena durante n días
    sim = simulate_markov_chain(P, args.dias, seed=args.seed)
    dist_emp = empirical_distribution(sim, len(pi))

    print(f"Estados simulados ({args.dias} días):")
    print(sim)

    print("\nDistribución empírica:")
    for i, p in enumerate(dist_emp):
        print(f"  S{i+1}: {p:.4f}")

    print()
