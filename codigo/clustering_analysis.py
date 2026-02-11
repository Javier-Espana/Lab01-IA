"""
Clustering: K-Means vs Jerárquico.
Ejercicio 5 del Lab01 - Inteligencia Artificial 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. GENERAR DATOS SINTÉTICOS
# ==============================================================================
print("Generando datos...")

# Generar datos con make_moons (200 muestras, 100 por cluster)
X_moons, y_moons = make_moons(n_samples=200, noise=0.05, random_state=42)

# Generar datos con make_circles (200 muestras, 100 por cluster)
X_circles, y_circles = make_circles(n_samples=200, noise=0.05, factor=0.5, random_state=42)

# Normalizar los datos
scaler = StandardScaler()
X_moons = scaler.fit_transform(X_moons)
X_circles = scaler.fit_transform(X_circles)

print("Datos generados: moons={}, circles={}".format(X_moons.shape, X_circles.shape))

# ==============================================================================
# 2. IMPLEMENTAR CLUSTERING
# ==============================================================================
print("\nEjecutando clustering...")

# K-Means para ambos datasets
kmeans_moons = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans_circles = KMeans(n_clusters=2, random_state=42, n_init=10)

labels_kmeans_moons = kmeans_moons.fit_predict(X_moons)
labels_kmeans_circles = kmeans_circles.fit_predict(X_circles)

# Agrupamiento Jerárquico con diferentes enlaces
linkage_methods = ['ward', 'complete', 'average', 'single']
hierarchical_results = {}

for linkage in linkage_methods:
    hierarchical_moons = AgglomerativeClustering(n_clusters=2, linkage=linkage)
    hierarchical_circles = AgglomerativeClustering(n_clusters=2, linkage=linkage)
    
    labels_hier_moons = hierarchical_moons.fit_predict(X_moons)
    labels_hier_circles = hierarchical_circles.fit_predict(X_circles)
    
    hierarchical_results[linkage] = {
        'moons': labels_hier_moons,
        'circles': labels_hier_circles
    }

# ==============================================================================
# 3. CALCULAR MÉTRICAS DE EVALUACIÓN
# ==============================================================================
print("\nCalculando métricas...")

def calculate_metrics(X, labels, algorithm_name, dataset_name):
    """Calcula métricas de validación para un clustering"""
    silhouette = silhouette_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    calinski_harabasz = calinski_harabasz_score(X, labels)
    
    return {
        'algorithm': algorithm_name,
        'dataset': dataset_name,
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'calinski_harabasz': calinski_harabasz
    }

# Métricas para K-Means
metrics_kmeans_moons = calculate_metrics(X_moons, labels_kmeans_moons, "K-Means", "Moons")
metrics_kmeans_circles = calculate_metrics(X_circles, labels_kmeans_circles, "K-Means", "Circles")

# Métricas para Agrupamiento Jerárquico
hierarchical_metrics = {}

for linkage in linkage_methods:
    metrics_moons = calculate_metrics(
        X_moons, 
        hierarchical_results[linkage]['moons'], 
        "Hierarchical ({})".format(linkage), 
        "Moons"
    )
    
    metrics_circles = calculate_metrics(
        X_circles, 
        hierarchical_results[linkage]['circles'], 
        "Hierarchical ({})".format(linkage), 
        "Circles"
    )
    
    hierarchical_metrics[linkage] = {
        'moons': metrics_moons,
        'circles': metrics_circles
    }

# Resumen mínimo en consola
print("\nResumen (Silhouette):")
print("  K-Means     | Moons: {:.4f} | Circles: {:.4f}".format(
    metrics_kmeans_moons['silhouette'], metrics_kmeans_circles['silhouette']))
for linkage in linkage_methods:
    print("  Hier-{:7} | Moons: {:.4f} | Circles: {:.4f}".format(
        linkage,
        hierarchical_metrics[linkage]['moons']['silhouette'],
        hierarchical_metrics[linkage]['circles']['silhouette']))

# ==============================================================================
# 4. VISUALIZACIÓN
# ==============================================================================
print("\nGenerando visualización...")

fig, axes = plt.subplots(2, 6, figsize=(24, 8))
fig.suptitle('Comparacion de Algoritmos de Clustering', fontsize=16, fontweight='bold')

# Fila 1: Moons
# Datos originales
axes[0, 0].scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap='viridis', alpha=0.6)
axes[0, 0].set_title('Moons (Datos Originales)')
axes[0, 0].set_xticks([])
axes[0, 0].set_yticks([])

# K-Means
axes[0, 1].scatter(X_moons[:, 0], X_moons[:, 1], c=labels_kmeans_moons, cmap='viridis', alpha=0.6)
axes[0, 1].scatter(kmeans_moons.cluster_centers_[:, 0], kmeans_moons.cluster_centers_[:, 1], 
                   c='red', marker='X', s=200, edgecolors='black', linewidths=2)
axes[0, 1].set_title('K-Means')
axes[0, 1].set_xticks([])
axes[0, 1].set_yticks([])

# Agrupamiento Jerárquico con diferentes enlaces
for idx, linkage in enumerate(linkage_methods):
    axes[0, idx + 2].scatter(X_moons[:, 0], X_moons[:, 1], 
                             c=hierarchical_results[linkage]['moons'], 
                             cmap='viridis', alpha=0.6)
    axes[0, idx + 2].set_title('Hierarchical ({})'.format(linkage))
    axes[0, idx + 2].set_xticks([])
    axes[0, idx + 2].set_yticks([])

# Fila 2: Circles
# Datos originales
axes[1, 0].scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, cmap='plasma', alpha=0.6)
axes[1, 0].set_title('Circles (Datos Originales)')
axes[1, 0].set_xticks([])
axes[1, 0].set_yticks([])

# K-Means
axes[1, 1].scatter(X_circles[:, 0], X_circles[:, 1], c=labels_kmeans_circles, cmap='plasma', alpha=0.6)
axes[1, 1].scatter(kmeans_circles.cluster_centers_[:, 0], kmeans_circles.cluster_centers_[:, 1], 
                   c='red', marker='X', s=200, edgecolors='black', linewidths=2)
axes[1, 1].set_title('K-Means')
axes[1, 1].set_xticks([])
axes[1, 1].set_yticks([])

# Agrupamiento Jerárquico con diferentes enlaces
for idx, linkage in enumerate(linkage_methods):
    axes[1, idx + 2].scatter(X_circles[:, 0], X_circles[:, 1], 
                             c=hierarchical_results[linkage]['circles'], 
                             cmap='plasma', alpha=0.6)
    axes[1, idx + 2].set_title('Hierarchical ({})'.format(linkage))
    axes[1, idx + 2].set_xticks([])
    axes[1, idx + 2].set_yticks([])

plt.tight_layout()
output_path = 'clustering_comparison.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print("Gráfico guardado: {}".format(output_path))
plt.close()
