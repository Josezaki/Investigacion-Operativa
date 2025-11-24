**TAREA: Cajeros automáticos con Disponibilidad variable (M/M/k + Markov)**
En este repositorio se encuentra la solución al Problema 2 de la tarea de Investigación Operativa, que analiza el desempeño de un sistema de cajeros automáticos sujetos a una variabilidad operativa modelada por una Cadena de Markov discreta.

El script *problema2.py* calcula la distribución estacionaria (pi) y las métricas de Teoría de Colas (rho, L, W) ponderadas por esta distribución, simulando finalmente la operación del sistema durante 30 días.

# Requisitos del Sistema
Para ejecutar el código, es necesario tener instalado Python 3.8 o superior, junto con las siguientes librerías:
 - numpy
 - panda
 - math (Librería estándar de Python)

# Instalación de dependencias:
>pip install numpy pandas

# Ejecución del Código
El script está diseñado para ejecutarse directamente desde la línea de comandos:
>python3 problema2.py

El programa entrega en consola:

1. Distribución estacionaria (π)

Cálculo teórico de la probabilidad de encontrarse en cada estado de disponibilidad de cajeros (Punto a).

2. Métricas por Estado (M/M/k)

Tabla que muestra, para cada estado S1, S2 y S3:

- ρ (utilización)

- L (número esperado de clientes en el sistema)

- W (tiempo esperado en el sistema)

Los estados S2 y S3 presentan saturación (ρ ≥ 1), por lo que L y W se reportan como ∞.

3. Métricas ponderadas por π

Cálculo del rendimiento promedio general del sistema (Punto b).
Debido a la saturación parcial, algunas métricas pueden resultar infinitas.

4. Simulación de 30 días

Genera:
- La secuencia de estados visitados día a día.
- La distribución empírica resultante (Punto c).

# Parámetros Configurables

El archivo problema2.py permite modificar fácilmente los valores clave en la sección inicial, lo cual es útil para analizar escenarios alternativos e incluir propuestas en el Punto (d) de la tarea. A continuacion, se muestran los parametros que estan por predeterminados en el codigo: 

> LAMBDA = 8.0
Tasa de llegada de clientes (lambda) [clientes/min]

> MU_POR_CAJERO = 4.0
Tasa de servicio por cajero (mu) [clientes/min]

> P 
Matriz de Transición de Markov (3 x 3)

> STATES = {0: 3, 1: 2, 2: 1}
Mapeo de estado a número de cajeros operativos (k)

> dias = 30
Número de pasos de la simulación de Markov (Punto c)

> np.random.seed() = 42
Semilla para reproducibilidad de la simulación

Para probar estos parametros configurables, se muestra a continuacion un ejemplo de ejecucion, con sus respectivos parametros

>  python3 problema2.py --lambda_ 7.0 --mu 5.0 --dias 20 --seed 1
# Descripción del Modelado y Supuestos

###  Modelo

El sistema se modela como una Cadena de Markov discreta en tiempo para la disponibilidad de servidores, acoplada a un modelo de Teoría de Colas M/M/k con capacidad infinita para la atención al cliente dentro de cada estado.

### Supuestos Clave

1. Modelo de Colas (M/M/k): Se asume que las llegadas de clientes siguen una distribución de Poisson (M) y los tiempos de servicio siguen una distribución Exponencial (M).

2. Independencia: La tasa de llegada (lambda) y la tasa de servicio por cajero (mu) se mantienen constantes independientemente del estado.

3. Saturación (rho >= 1): De acuerdo con la Teoría de Colas, si la utilización del sistema (lambda / k*mu) es igual o mayor a 1 (como ocurre en S2 y S3), las métricas de congestión (L y W) se reportan como infinitas, reflejando la inestabilidad a largo plazo.

