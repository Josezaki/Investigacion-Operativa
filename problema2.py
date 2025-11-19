"""
Problema 2 — Cajeros automáticos con disponibilidad variable
M/M/k + cadena de Markov discreta

- Estados:
    S1: 3 cajeros operativos
    S2: 2 cajeros operativos
    S3: 1 cajero operativo

- Parámetros:
    mu_por_cajero = 4 clientes/min
    lambda_total   = 8 clientes/min

- Matriz de transición P (3x3):
    P[1->*] = [0.6, 0.3, 0.1]
    P[2->*] = [0.2, 0.6, 0.2]
    P[3->*] = [0.1, 0.2, 0.7]
"""

import numpy as np
import pandas as pd
from math import factorial



# Parámetros del problema

LAMBDA = 8.0           # clientes/min
MU_POR_CAJERO = 4.0    # clientes/min por cajero

# Estados: número de cajeros operativos en cada estado
STATES = {
    0: 3,  # S1
    1: 2,  # S2
    2: 1,  # S3
}

# Matriz de transición P (orden: S1, S2, S3)
P = np.array([
    [0.6, 0.3, 0.1],   # desde S1
    [0.2, 0.6, 0.2],   # desde S2
    [0.1, 0.2, 0.7],   # desde S3
])


# (a) Cálculo de la distribución estacionaria π

def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """
    Calcula la distribución estacionaria π de una cadena de Markov
    a partir de la matriz de transición P.
    
    Se resuelve el sistema de ecuaciones lineales:
    π P = π  =>  π (P - I) = 0
    junto con la condición de normalización: sum(π_i) = 1

    Esto se formula como A * π_T = b, donde A es (P^T - I) modificada
    en la última fila para la normalización, y b es [0, ..., 1]^T.
    """
    n_states = P.shape[0]

    # 1. Matriz de coeficientes A = P^T - I
    A = (P.T - np.identity(n_states))

    # 2. Reemplazar la última fila con la condición de normalización (suma de pi es 1)
    A[-1, :] = 1.0

    # 3. Vector del lado derecho b: [0, 0, ..., 1]^T
    b = np.zeros(n_states)
    b[-1] = 1.0

    # 4. Resolver A * pi_T = b
    try:
        pi = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        print("Advertencia: La matriz es singular. No se pudo calcular la distribución estacionaria.")
        return np.full(n_states, np.nan)

    # El resultado ya debería estar normalizado
    return pi


# -------------------------------------------------------------------
# (b) Métricas de la cola M/M/k para un estado dado
# -------------------------------------------------------------------

def mmk_metrics(lamb: float, mu_per_server: float, k: int):
    """
    Calcula métricas estándar de la cola M/M/k (capacidad infinita):

    IMPORTANTE: Las fórmulas clásicas solo son válidas si rho < 1.
    Si rho >= 1 se retornan np.inf para las métricas correspondientes.
    """
     # Tasa de servicio del sistema (k * mu_por_cajero)
    mu_sys_rate = k * mu_per_server

    # Parámetro de intensidad de tráfico para 1 servidor (a = lambda/mu)
    a = lamb / mu_per_server
    
    # Utilización del sistema (rho_sys = lambda / k*mu)
    rho_sys = lamb / mu_sys_rate

    # Revisar condición de no saturación
    if rho_sys >= 1 or k <= 0:
        return {'rho': rho_sys, 'L': np.inf, 'W': np.inf}

    # 1. Cálculo de P0 (Probabilidad de 0 clientes en el sistema)
    
    # Sumatoria (n=0 a k-1) de (a^n / n!)
    sum_term = sum((a**n) / factorial(n) for n in range(k))
    
    # Término de la cola: (a^k / k!) * (k*mu / (k*mu - lambda))
    # Simplificando: (a^k / k!) * (1 / (1 - rho_sys))
    queue_term = (a**k / factorial(k)) * (1.0 / (1.0 - rho_sys))
    
    P0_inverse = sum_term + queue_term
    P0 = 1.0 / P0_inverse

    # 2. Cálculo de Lq (Número medio de clientes en la cola)
    
    # Usando la fórmula de la diapositiva: 
    # Lq = [lambda * mu * (lambda/mu)^k * P0] / [(k-1)! * (k*mu - lambda)^2]
    Lq_slide_num = lamb * mu_per_server * (a**k) * P0
    Lq_slide_den = factorial(k - 1) * (mu_sys_rate - lamb)**2
    Lq = Lq_slide_num / Lq_slide_den

    # 3. Cálculo de Wq (Tiempo medio de espera en la cola)
    Wq = Lq / lamb
    
    # 4. Cálculo de W (Tiempo medio en el sistema)
    W = Wq + (1.0 / mu_per_server)
    
    # 5. Cálculo de L (Número medio de clientes en el sistema)
    L = lamb * W
    
    # La Utilización (rho) solicitada en (b) es rho_sys
    return {'rho': rho_sys, 'L': L, 'W': W}



def weighted_metrics(pi: np.ndarray):
    """
    Calcula las métricas de manera ponderada por la distribución estacionaria π.
    Para cada estado i se computa M/M/k con k = número de cajeros en ese estado.
    """

    n_states = pi.shape[0]
    data = []

    for i in range(n_states):
        # USAR EL DICCIONARIO CORRECTO
        k = STATES[i]

        # USAR LAS VARIABLES GLOBALES CORRECTAS
        metrics = mmk_metrics(LAMBDA, MU_POR_CAJERO, k)

        metrics['pi'] = pi[i]
        metrics['k'] = k
        metrics['Estado'] = f"S{i+1}"
        data.append(metrics)

    df_states = pd.DataFrame(data)

    # Manejo de saturaciones
    is_saturated_in_state = np.any(df_states['L'] == np.inf) and np.any(df_states['pi'][df_states['L'] == np.inf] > 1e-9)

    weighted_L = np.nansum(df_states['L'].replace(np.inf, np.nan) * df_states['pi'])
    weighted_W = np.nansum(df_states['W'].replace(np.inf, np.nan) * df_states['pi'])
    weighted_rho = np.nansum(df_states['rho'].replace(np.inf, np.nan) * df_states['pi'])

    if is_saturated_in_state:
        weighted_L = np.inf
        weighted_W = np.inf

    resumen = {
        'rho_ponderado': weighted_rho,
        'L_ponderado (Clientes en sistema)': weighted_L,
        'W_ponderado (Tiempo en sistema)': weighted_W,
    }

    df_states = df_states[['Estado', 'k', 'pi', 'rho', 'L', 'W']]

    return df_states, resumen

        
    df_states = pd.DataFrame(data)
    
    # Cálculo de las métricas ponderadas (promedio de las métricas por pi)
    
    # Reemplazar np.inf con np.nan para un cálculo correcto con nansum
    # Si algún estado con pi > 0 está saturado (infinito), el ponderado debe ser infinito.
    
    is_saturated_in_state = np.any(df_states['L'] == np.inf) and np.any(df_states['pi'][df_states['L'] == np.inf] > 1e-9)
    
    weighted_L = np.nansum(df_states['L'].replace(np.inf, np.nan) * df_states['pi'])
    weighted_W = np.nansum(df_states['W'].replace(np.inf, np.nan) * df_states['pi'])
    weighted_rho = np.nansum(df_states['rho'].replace(np.inf, np.nan) * df_states['pi'])
    
    if is_saturated_in_state:
        weighted_L = np.inf
        weighted_W = np.inf
        
    resumen = {
        'rho_ponderado': weighted_rho,
        'L_ponderado (Clientes en sistema)': weighted_L,
        'W_ponderado (Tiempo en sistema)': weighted_W,
    }
    
    # Reordenar columnas para una mejor presentación
    df_states = df_states[['Estado', 'k', 'pi', 'rho', 'L', 'W']]
    
    return df_states, resumen
 


# -------------------------------------------------------------------
# (c) Simulación de la cadena de Markov por 30 días
# -------------------------------------------------------------------

def simulate_markov_chain(P: np.ndarray, days: int, initial_state: int = 0) -> np.ndarray:
    """
    Simula la cadena de Markov por 'days' pasos (un paso = 1 día).

    - P: matriz de transición
    - days: número de días a simular
    - initial_state: índice del estado inicial (0, 1 o 2)

    Retorna un arreglo con los índices de estados visitados.
    """
    n_states = P.shape[0]
    states = np.zeros(days, dtype=int)
    states[0] = initial_state

    rng = np.random.default_rng()

    for t in range(1, days):
        current = states[t - 1]
        probs = P[current]
        states[t] = rng.choice(np.arange(n_states), p=probs)

    return states


def empirical_distribution(states: np.ndarray, n_states: int) -> np.ndarray:
    """
    Calcula la distribución empírica de estados a partir de la simulación.
    """
    counts = np.bincount(states, minlength=n_states)
    return counts / counts.sum()


# -------------------------------------------------------------------
# Ejecución de prueba (para ir viendo resultados)
# -------------------------------------------------------------------

if __name__ == "__main__":
    # (a) Distribución estacionaria
    pi = stationary_distribution(P)
    print("Distribución estacionaria π:")
    for i, p in enumerate(pi):
        print(f"  S{i+1}: {p:.4f}")
    print()

    # (b) Métricas por estado y ponderadas
    df_states, resumen = weighted_metrics(pi)
    print("Métricas por estado (M/M/k):")
    print(df_states.to_string(index=False))
    print("\nMétricas ponderadas por π:")
    for k, v in resumen.items():
        print(f"  {k}: {v}")

    # (c) Simulación de 30 días
    dias = 30
    estados_simulados = simulate_markov_chain(P, dias, initial_state=0)
    dist_empirica = empirical_distribution(estados_simulados, n_states=P.shape[0])

    print("\nEstados simulados (0=S1, 1=S2, 2=S3):")
    print(estados_simulados)
    print("\nDistribución empírica después de 30 días:")
    for i, p in enumerate(dist_empirica):
        print(f"  S{i+1}: {p:.4f}")
