"""
Agrupamiento Jerárquico con datos de países
Ejercicio 3 del Lab01 - Inteligencia Artificial 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform


def distancia_hamming(u, v):
    """Calcula la distancia de Hamming entre dos vectores binarios."""
    return np.sum(u != v) / len(u)


def matriz_distancias_hamming(data):
    """Calcula la matriz de distancias de Hamming para datos binarios."""
    n = len(data)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist = distancia_hamming(data[i], data[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    return dist_matrix


# =============================================================================
# CARGAR DATOS
# =============================================================================
print("="*70)
print("AGRUPAMIENTO JERÁRQUICO - PAÍSES")
print("="*70)

df = pd.read_excel('../datos/countries_binary.xlsx')
print(f"\nDimensiones del dataset: {df.shape}")
print(f"Países: {df['Country'].tolist()}")

# Extraer nombres de países y datos numéricos
paises = df['Country'].values
datos = df.drop('Country', axis=1).values.astype(float)

print(f"\nNúmero de características: {datos.shape[1]}")
print(f"Número de países: {datos.shape[0]}")

# Verificar si los datos son binarios
valores_unicos = np.unique(datos)
print(f"Valores únicos en los datos: {valores_unicos}")
es_binario = np.all(np.isin(datos, [0, 1]))
print(f"¿Datos binarios? {es_binario}")


# =============================================================================
# CONFIGURACIÓN DE MÉTODOS Y MÉTRICAS
# =============================================================================
metodos = ['single', 'complete', 'average', 'ward']
metodos_nombres = {
    'single': 'Simple (Single Linkage)',
    'complete': 'Completo (Complete Linkage)', 
    'average': 'Promedio (Average Linkage)',
    'ward': 'Ward'
}

metricas = ['euclidean', 'hamming']
metricas_nombres = {
    'euclidean': 'Euclideana',
    'hamming': 'Hamming'
}


# =============================================================================
# GENERAR DENDROGRAMAS
# =============================================================================
# Nota: Ward solo funciona con distancia euclideana
# Para otras combinaciones método-métrica usamos las demás

fig, axes = plt.subplots(4, 2, figsize=(16, 20))
fig.suptitle('Dendrogramas de Agrupamiento Jerárquico - Países', fontsize=16, fontweight='bold')

for i, metodo in enumerate(metodos):
    for j, metrica in enumerate(metricas):
        ax = axes[i, j]
        
        # Ward solo funciona con euclideana
        if metodo == 'ward' and metrica == 'hamming':
            ax.text(0.5, 0.5, 'Ward solo funciona\ncon distancia Euclideana', 
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title(f'{metodos_nombres[metodo]} + {metricas_nombres[metrica]}')
            ax.axis('off')
            continue
        
        try:
            # Calcular linkage
            if metrica == 'hamming':
                # Calcular distancias de Hamming condensadas
                dist_condensed = pdist(datos, metric='hamming')
                Z = linkage(dist_condensed, method=metodo)
            else:
                Z = linkage(datos, method=metodo, metric=metrica)
            
            # Crear dendrograma
            dendrogram(Z, labels=paises, ax=ax, leaf_rotation=90, leaf_font_size=8)
            ax.set_title(f'{metodos_nombres[metodo]} + {metricas_nombres[metrica]}', fontsize=11)
            ax.set_xlabel('País')
            ax.set_ylabel('Distancia')
            
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {str(e)[:50]}', 
                   ha='center', va='center', fontsize=10, transform=ax.transAxes)
            ax.set_title(f'{metodos_nombres[metodo]} + {metricas_nombres[metrica]}')

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('../figuras/dendrogramas_paises.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nFigura guardada en: dendrogramas_paises.png")


# =============================================================================
# ANÁLISIS MÁS DETALLADO
# =============================================================================
print("\n" + "="*70)
print("ANÁLISIS DE AGRUPAMIENTOS")
print("="*70)

# Función para obtener clusters a cierta altura
from scipy.cluster.hierarchy import fcluster

print("\n--- Comparación de agrupamientos con k=3 clusters ---\n")

for metodo in metodos:
    for metrica in metricas:
        if metodo == 'ward' and metrica == 'hamming':
            continue
            
        try:
            if metrica == 'hamming':
                dist_condensed = pdist(datos, metric='hamming')
                Z = linkage(dist_condensed, method=metodo)
            else:
                Z = linkage(datos, method=metodo, metric=metrica)
            
            # Obtener 3 clusters
            clusters = fcluster(Z, t=3, criterion='maxclust')
            
            print(f"{metodos_nombres[metodo]} + {metricas_nombres[metrica]}:")
            for c in range(1, 4):
                paises_cluster = paises[clusters == c]
                print(f"  Cluster {c}: {', '.join(paises_cluster)}")
            print()
            
        except Exception as e:
            print(f"{metodo} + {metrica}: Error - {e}\n")


# =============================================================================
# VISUALIZACIÓN INDIVIDUAL MÁS GRANDE
# =============================================================================
# Crear dendrogramas individuales más grandes para mejor visualización

fig2, axes2 = plt.subplots(2, 2, figsize=(16, 14))
fig2.suptitle('Dendrogramas con Distancia Euclideana', fontsize=14, fontweight='bold')

for idx, metodo in enumerate(metodos):
    ax = axes2[idx // 2, idx % 2]
    Z = linkage(datos, method=metodo, metric='euclidean')
    dendrogram(Z, labels=paises, ax=ax, leaf_rotation=45, leaf_font_size=9)
    ax.set_title(f'{metodos_nombres[metodo]}', fontsize=12)
    ax.set_xlabel('País')
    ax.set_ylabel('Distancia')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('../figuras/dendrogramas_euclidean.png', dpi=150, bbox_inches='tight')
plt.show()


fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6))
fig3.suptitle('Dendrogramas con Distancia de Hamming', fontsize=14, fontweight='bold')

for idx, metodo in enumerate(['single', 'complete', 'average']):
    ax = axes3[idx]
    dist_condensed = pdist(datos, metric='hamming')
    Z = linkage(dist_condensed, method=metodo)
    dendrogram(Z, labels=paises, ax=ax, leaf_rotation=45, leaf_font_size=9)
    ax.set_title(f'{metodos_nombres[metodo]}', fontsize=12)
    ax.set_xlabel('País')
    ax.set_ylabel('Distancia')

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('../figuras/dendrogramas_hamming.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nFiguras adicionales guardadas:")
print("- dendrogramas_euclidean.png")
print("- dendrogramas_hamming.png")


# =============================================================================
# RESUMEN
# =============================================================================
print("\n" + "="*70)
print("RESUMEN Y OBSERVACIONES")
print("="*70)

print("""
MÉTODOS DE AGRUPAMIENTO:
------------------------
1. Simple (Single Linkage): 
   - Usa la distancia mínima entre puntos de dos clusters.
   - Tiende a crear clusters alargados (efecto cadena).
   
2. Completo (Complete Linkage):
   - Usa la distancia máxima entre puntos de dos clusters.
   - Crea clusters más compactos y esféricos.
   
3. Promedio (Average Linkage):
   - Usa el promedio de todas las distancias entre pares de puntos.
   - Compromiso entre simple y completo.
   
4. Ward:
   - Minimiza la varianza dentro de los clusters.
   - Tiende a crear clusters de tamaño similar.
   - Solo funciona con distancia euclideana.

MÉTRICAS:
---------
1. Euclideana: 
   - Distancia geométrica estándar.
   - Sensible a la escala de las variables.
   
2. Hamming:
   - Proporción de atributos que difieren.
   - Ideal para datos binarios como este dataset.
   - Valores entre 0 (idénticos) y 1 (totalmente diferentes).

OBSERVACIONES SOBRE LOS DATOS DE PAÍSES:
----------------------------------------
- Los datos son binarios (0 y 1), por lo que Hamming es apropiada.
- Con Single Linkage se observa el efecto cadena.
- Complete y Ward tienden a dar clusters más balanceados.
- Los países con características similares se agrupan juntos.
""")
