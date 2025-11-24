# ---------------------------------------------------------------
# Problema 2 — Cajeros automáticos con disponibilidad variable
# M/M/k + Cadena de Markov discreta
# ---------------------------------------------------------------
# Este código implementa:
#  (a) Cálculo de la distribución estacionaria π de una cadena de Markov.
#  (b) Cálculo de métricas de teoría de colas M/M/k por estado.
#  (c) Cálculo ponderado de métricas usando π.
#  (d) Simulación de 30 días de la cadena de Markov.
# Además incorpora argparse para permitir cambiar parámetros desde consola.
# ---------------------------------------------------------------

import numpy as np
import pandas as pd
from math import factorial
import argparse

# -------------------------------------------------------------------
# (a) Cálculo de la distribución estacionaria
# -------------------------------------------------------------------
# Se resuelve el sistema πP = π con la condición sum(π) = 1.
# Para ello se construye un sistema lineal equivalente basado en (P^T − I).
# La última ecuación se reemplaza por la normalización de π.
# -------------------------------------------------------------------

def stationary_distribution(P: np.ndarray) -> np.ndarray:
    n_states = P.shape[0]

    # Matriz del sistema A = P^T − I
    A = (P.T - np.identity(n_states))

    # La última fila se reemplaza por [1,1,...,1] para imponer sum(pi)=1
    A[-1, :] = 1.0

    # Vector del lado derecho: solo el último elemento es 1
    b = np.zeros(n_states)
    b[-1] = 1.0

    try:
        pi = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        print("Error: Matriz singular. No se pudo calcular π.")
        return np.full(n_states, np.nan)

    return pi

# -------------------------------------------------------------------
# (b) Cálculo de métricas M/M/k
# -------------------------------------------------------------------
# Para cada estado se modela un sistema M/M/k.
# Si rho >= 1, el sistema está saturado y L, W → ∞.
# -------------------------------------------------------------------

def mmk_metrics(lamb: float, mu: float, k: int):
    mu_sys_rate = k * mu
    rho = lamb / mu_sys_rate

    # Si rho >= 1, el sistema no es estable
    if rho >= 1 or k <= 0:
        return {'rho': rho, 'L': np.inf, 'W': np.inf}

    # a = carga ofrecida al sistema por cada servidor
    a = lamb / mu

    # Cálculo de P0 (probabilidad de 0 clientes en el sistema)
    sum_term = sum((a**n) / factorial(n) for n in range(k))
    queue_term = (a**k / factorial(k)) * (1 / (1 - rho))
    P0 = 1 / (sum_term + queue_term)

    # Fórmula estándar de Lq para M/M/k
    Lq = (P0 * lamb * mu * (a**k)) / (factorial(k - 1) * (mu_sys_rate - lamb)**2)

    # Tiempos de espera
    Wq = Lq / lamb
    W = Wq + (1 / mu)
    L = lamb * W

    return {'rho': rho, 'L': L, 'W': W}

# -------------------------------------------------------------------
# (b) Métricas ponderadas usando la distribución estacionaria π
# -------------------------------------------------------------------
# Para cada estado se calcula M/M/k y luego se pondera por π.
# Si algún estado con π>0 está saturado, los valores agregados se vuelven ∞.
# -------------------------------------------------------------------

def weighted_metrics(pi, lamb, mu, states):
    data = []

    for i in range(len(pi)):
        k = states[i]  # número de cajeros en el estado
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

# -------------------------------------------------------------------
# (c) Simulación de la cadena de Markov
# -------------------------------------------------------------------
# Se simulan cambios de estado durante 'days' pasos.
# En cada paso se elige el siguiente estado según las probabilidades de P.
# -------------------------------------------------------------------

def simulate_markov_chain(P, days, initial_state=0, seed=None):
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

# Distribución empírica

def empirical_distribution(sim_states, n_states):
    counts = np.bincount(sim_states, minlength=n_states)
    return counts / counts.sum()

# -------------------------------------------------------------------
# MAIN — argparse para parámetros configurables
# -------------------------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Problema 2 — Cajeros automáticos M/M/k + Markov"
    )

    # Parámetros ajustables desde consola
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

    # Matriz de transición
    P = np.array([
        [0.6, 0.3, 0.1],
        [0.2, 0.6, 0.2],
        [0.1, 0.2, 0.7],
    ])

    # Mostrar parámetros usados
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

    # (b) Métricas por estado + ponderadas
    df, resumen = weighted_metrics(pi, args.lambda_, args.mu, STATES)

    print("Métricas por estado:")
    print(df.to_string(index=False))

    print("\nMétricas ponderadas por π:")
    for k, v in resumen.items():
        print(f"  {k}: {v}")
    print()

    # (c) Simulación
    sim = simulate_markov_chain(P, args.dias, seed=args.seed)
    dist_emp = empirical_distribution(sim, len(pi))

    print(f"Estados simulados ({args.dias} días):")
    print(sim)

    print("\nDistribución empírica:")
    for i, p in enumerate(dist_emp):
        print(f"  S{i+1}: {p:.4f}")

    print()
