"""
Análisis K-means de datos de países
Ejercicio 4 del Lab01 - Inteligencia Artificial 2026
Comparación con agrupamiento jerárquico
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, silhouette_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

# Importar nuestro k-means desde cero
from kmeans_from_scratch import kmeans


# =============================================================================
# CARGAR DATOS
# =============================================================================
print("="*70)
print("ANÁLISIS K-MEANS - PAÍSES")
print("="*70)

df = pd.read_excel('../datos/countries_binary.xlsx')
paises = df['Country'].values
datos = df.drop('Country', axis=1).values.astype(float)

print(f"\nDimensiones: {datos.shape}")
print(f"Países: {list(paises)}")


# =============================================================================
# MÉTODO DEL CODO PARA ELEGIR K
# =============================================================================
print("\n" + "="*70)
print("MÉTODO DEL CODO")
print("="*70)

inercias = []
silhouettes = []
ks = range(2, 10)

for k in ks:
    km = SklearnKMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(datos)
    inercias.append(km.inertia_)
    silhouettes.append(silhouette_score(datos, labels))
    
print("\nInercias por k:")
for k, iner in zip(ks, inercias):
    print(f"  k={k}: {iner:.2f}")

print("\nSilhouette scores por k:")
for k, sil in zip(ks, silhouettes):
    print(f"  k={k}: {sil:.4f}")

mejor_k_silhouette = ks[np.argmax(silhouettes)]
print(f"\nMejor k según Silhouette: {mejor_k_silhouette}")


# =============================================================================
# K-MEANS CON K=3 (para comparar con jerárquico)
# =============================================================================
print("\n" + "="*70)
print("K-MEANS CON K=3")
print("="*70)

# K-means propio
labels_propio, centroids_propio = kmeans(datos.tolist(), k=3, seed=42)
labels_propio = np.array(labels_propio)

# K-means sklearn
km_sklearn = SklearnKMeans(n_clusters=3, random_state=42, n_init=10)
labels_sklearn = km_sklearn.fit_predict(datos)

print("\n--- K-means Propio (k=3) ---")
for c in range(3):
    paises_cluster = paises[labels_propio == c]
    print(f"  Cluster {c}: {', '.join(paises_cluster)}")

print("\n--- K-means Sklearn (k=3) ---")
for c in range(3):
    paises_cluster = paises[labels_sklearn == c]
    print(f"  Cluster {c}: {', '.join(paises_cluster)}")

# ARI entre propio y sklearn
ari_propio_sklearn = adjusted_rand_score(labels_propio, labels_sklearn)
print(f"\nARI entre k-means propio y sklearn: {ari_propio_sklearn:.4f}")


# =============================================================================
# COMPARACIÓN CON AGRUPAMIENTO JERÁRQUICO
# =============================================================================
print("\n" + "="*70)
print("COMPARACIÓN CON AGRUPAMIENTO JERÁRQUICO")
print("="*70)

# Obtener clusters jerárquicos con diferentes métodos
metodos_jerarquico = {
    'Ward (Euclidean)': linkage(datos, method='ward', metric='euclidean'),
    'Complete (Euclidean)': linkage(datos, method='complete', metric='euclidean'),
    'Complete (Hamming)': linkage(pdist(datos, metric='hamming'), method='complete'),
    'Average (Hamming)': linkage(pdist(datos, metric='hamming'), method='average'),
}

print("\n--- Comparación de agrupamientos (k=3) ---\n")

resultados_comparacion = {}

for nombre, Z in metodos_jerarquico.items():
    clusters_jer = fcluster(Z, t=3, criterion='maxclust')
    ari = adjusted_rand_score(labels_sklearn, clusters_jer)
    resultados_comparacion[nombre] = {
        'clusters': clusters_jer,
        'ari_vs_kmeans': ari
    }
    
    print(f"{nombre}:")
    for c in range(1, 4):
        paises_cluster = paises[clusters_jer == c]
        print(f"  Cluster {c}: {', '.join(paises_cluster)}")
    print(f"  ARI vs K-means: {ari:.4f}\n")

print("\n--- K-means sklearn (referencia) ---")
for c in range(3):
    paises_cluster = paises[labels_sklearn == c]
    print(f"  Cluster {c}: {', '.join(paises_cluster)}")


# =============================================================================
# VISUALIZACIÓN
# =============================================================================
from sklearn.decomposition import PCA

# Reducir a 2D para visualización
pca = PCA(n_components=2)
datos_2d = pca.fit_transform(datos)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Comparación K-means vs Agrupamiento Jerárquico - Países', fontsize=14, fontweight='bold')

# K-means
axes[0, 0].scatter(datos_2d[:, 0], datos_2d[:, 1], c=labels_sklearn, cmap='viridis', s=100)
for i, pais in enumerate(paises):
    axes[0, 0].annotate(pais, (datos_2d[i, 0], datos_2d[i, 1]), fontsize=8)
axes[0, 0].set_title('K-means (k=3)')
axes[0, 0].set_xlabel('PC1')
axes[0, 0].set_ylabel('PC2')

# Jerárquico Ward
axes[0, 1].scatter(datos_2d[:, 0], datos_2d[:, 1], 
                   c=resultados_comparacion['Ward (Euclidean)']['clusters'], cmap='viridis', s=100)
for i, pais in enumerate(paises):
    axes[0, 1].annotate(pais, (datos_2d[i, 0], datos_2d[i, 1]), fontsize=8)
axes[0, 1].set_title('Jerárquico Ward')
axes[0, 1].set_xlabel('PC1')
axes[0, 1].set_ylabel('PC2')

# Jerárquico Complete Euclidean
axes[0, 2].scatter(datos_2d[:, 0], datos_2d[:, 1], 
                   c=resultados_comparacion['Complete (Euclidean)']['clusters'], cmap='viridis', s=100)
for i, pais in enumerate(paises):
    axes[0, 2].annotate(pais, (datos_2d[i, 0], datos_2d[i, 1]), fontsize=8)
axes[0, 2].set_title('Jerárquico Complete (Euclidean)')
axes[0, 2].set_xlabel('PC1')
axes[0, 2].set_ylabel('PC2')

# Jerárquico Complete Hamming
axes[1, 0].scatter(datos_2d[:, 0], datos_2d[:, 1], 
                   c=resultados_comparacion['Complete (Hamming)']['clusters'], cmap='viridis', s=100)
for i, pais in enumerate(paises):
    axes[1, 0].annotate(pais, (datos_2d[i, 0], datos_2d[i, 1]), fontsize=8)
axes[1, 0].set_title('Jerárquico Complete (Hamming)')
axes[1, 0].set_xlabel('PC1')
axes[1, 0].set_ylabel('PC2')

# Jerárquico Average Hamming
axes[1, 1].scatter(datos_2d[:, 0], datos_2d[:, 1], 
                   c=resultados_comparacion['Average (Hamming)']['clusters'], cmap='viridis', s=100)
for i, pais in enumerate(paises):
    axes[1, 1].annotate(pais, (datos_2d[i, 0], datos_2d[i, 1]), fontsize=8)
axes[1, 1].set_title('Jerárquico Average (Hamming)')
axes[1, 1].set_xlabel('PC1')
axes[1, 1].set_ylabel('PC2')

# Método del codo
axes[1, 2].plot(list(ks), inercias, 'bo-')
axes[1, 2].axvline(x=3, color='r', linestyle='--', label='k=3')
axes[1, 2].set_xlabel('k')
axes[1, 2].set_ylabel('Inercia')
axes[1, 2].set_title('Método del Codo')
axes[1, 2].legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('../figuras/kmeans_paises_comparacion.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nFigura guardada en: kmeans_paises_comparacion.png")


# =============================================================================
# RESUMEN
# =============================================================================
print("\n" + "="*70)
print("RESUMEN DE COMPARACIÓN")
print("="*70)

print("""
RESULTADOS:
-----------
Los agrupamientos NO son exactamente iguales entre K-means y jerárquico.

ARI (Adjusted Rand Index) vs K-means:""")

for nombre, res in resultados_comparacion.items():
    print(f"  - {nombre}: {res['ari_vs_kmeans']:.4f}")

print("""
JUSTIFICACIÓN DE LAS DIFERENCIAS:
---------------------------------
1. DIFERENCIA EN EL ALGORITMO:
   - K-means minimiza la inercia (suma de distancias al cuadrado al centroide).
   - Jerárquico construye clusters de forma aglomerativa según el criterio de enlace.

2. MÉTRICAS DE DISTANCIA:
   - K-means usa distancia euclideana implícitamente.
   - Jerárquico puede usar Hamming, más apropiada para datos binarios.

3. SENSIBILIDAD A LA INICIALIZACIÓN:
   - K-means depende de la inicialización aleatoria de centroides.
   - Jerárquico es determinístico (mismo resultado siempre).

4. FORMA DE LOS CLUSTERS:
   - K-means asume clusters esféricos de tamaño similar.
   - Jerárquico puede capturar estructuras más complejas.

5. PARA DATOS BINARIOS:
   - La distancia de Hamming (jerárquico) es más apropiada.
   - K-means con euclideana puede no capturar bien la similitud binaria.

CONCLUSIÓN:
-----------
Las agrupaciones son SIMILARES pero NO IDÉNTICAS. Los métodos jerárquicos 
con distancia Hamming (ARI más bajo vs K-means) capturan una estructura 
diferente más apropiada para datos binarios. K-means y Ward con euclideana 
dan resultados más parecidos porque ambos minimizan varianza/inercia.
""")
