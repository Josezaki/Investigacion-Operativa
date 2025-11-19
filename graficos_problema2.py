import numpy as np
import matplotlib.pyplot as plt

# Importar funciones y datos desde el archivo principal
from problema2 import (
    stationary_distribution, simulate_markov_chain, empirical_distribution,
    P
)


# ---------------------------------------------
# (1) Distribución estacionaria
# ---------------------------------------------
pi = stationary_distribution(P)


# ---------------------------------------------
# (2) Simulación 30 días
# ---------------------------------------------
dias = 30
estados_simulados = np.array([0, 0, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 0, 0])
dist_empirica = empirical_distribution(estados_simulados, n_states=P.shape[0])


# =============================================
# FUNCIÓN AUXILIAR PARA GUARDAR FIGURAS
# =============================================

def guardar_png(nombre_archivo):
    """
    Guarda la figura actual en PNG, sobrescribiendo el archivo si existe.
    """
    plt.savefig(
        nombre_archivo,
        dpi=300,          # alta calidad
        bbox_inches='tight'
    )
    print(f"✔ Imagen guardada: {nombre_archivo}")


# =============================================
# GRÁFICO 1 — Evolución del estado (30 días)
# =============================================

plt.figure(figsize=(10,4))
plt.plot(estados_simulados, marker='o')
plt.yticks([0,1,2], ["S1 (3 cajeros)", "S2 (2 cajeros)", "S3 (1 cajero)"])
plt.xlabel("Día")
plt.ylabel("Estado")
plt.title("Evolución del estado de cajeros durante 30 días")
plt.grid(True)
plt.tight_layout()

guardar_png("grafico_estados_30_dias.png")
plt.close()


# =============================================
# GRÁFICO 2 — Distribución estacionaria vs empírica
# =============================================

plt.figure(figsize=(7,4))
x = np.arange(3)
width = 0.35

plt.bar(x - width/2, pi, width, label='π estacionaria', alpha=0.7)
plt.bar(x + width/2, dist_empirica, width, label='Distribución empírica', alpha=0.7)

plt.xticks(x, ["S1", "S2", "S3"])
plt.ylabel("Probabilidad")
plt.title("Estacionaria vs Empírica (30 días)")
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

guardar_png("grafico_distribuciones.png")
plt.close()


# =============================================
# GRÁFICO 3 — Heatmap de la matriz de transición
# =============================================

plt.figure(figsize=(5,4))
plt.imshow(P, cmap='Blues', interpolation='nearest')
plt.colorbar(label="Probabilidad")
plt.xticks([0,1,2], ["S1","S2","S3"])
plt.yticks([0,1,2], ["S1","S2","S3"])
plt.title("Matriz de transición P")
plt.tight_layout()

guardar_png("heatmap_matriz_P.png")
plt.close()
