"""
Evaluación del algoritmo k-means desde cero vs sklearn
Ejercicio 2 del Lab01 - Inteligencia Artificial 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
import seaborn as sns

# Importar nuestro k-means desde cero
from kmeans_from_scratch import kmeans


def comparar_kmeans(data, k, nombre_dataset, seed=42):
    """
    Compara nuestro k-means vs sklearn.
    
    Args:
        data: numpy array o lista de datos (n x d)
        k: número de clústers
        nombre_dataset: nombre para mostrar en resultados
        seed: semilla aleatoria
    
    Returns:
        dict con resultados de comparación
    """
    # Convertir a lista para nuestro algoritmo
    data_list = data.tolist() if isinstance(data, np.ndarray) else data
    data_np = np.array(data_list)
    
    # Nuestro k-means
    labels_propio, centroids_propio = kmeans(data_list, k=k, seed=seed)
    labels_propio = np.array(labels_propio)
    centroids_propio = np.array(centroids_propio)
    
    # Sklearn k-means
    sklearn_km = SklearnKMeans(n_clusters=k, random_state=seed, n_init=10)
    labels_sklearn = sklearn_km.fit_predict(data_np)
    centroids_sklearn = sklearn_km.cluster_centers_
    
    # Métricas de comparación
    ari = adjusted_rand_score(labels_propio, labels_sklearn)
    nmi = normalized_mutual_info_score(labels_propio, labels_sklearn)
    
    # Inercia (suma de distancias al cuadrado a centroides)
    def calcular_inercia(data, labels, centroids):
        inercia = 0
        for i, point in enumerate(data):
            centroid = centroids[labels[i]]
            inercia += np.sum((np.array(point) - centroid) ** 2)
        return inercia
    
    inercia_propio = calcular_inercia(data_list, labels_propio, centroids_propio)
    inercia_sklearn = sklearn_km.inertia_
    
    print(f"\n{'='*60}")
    print(f"DATASET: {nombre_dataset}")
    print(f"{'='*60}")
    print(f"Número de muestras: {len(data_list)}")
    print(f"Número de características: {len(data_list[0])}")
    print(f"Número de clústers (k): {k}")
    print(f"\n--- Métricas de comparación ---")
    print(f"Adjusted Rand Index (ARI): {ari:.4f}")
    print(f"  (1.0 = asignaciones idénticas, 0.0 = aleatorio)")
    print(f"Normalized Mutual Information (NMI): {nmi:.4f}")
    print(f"  (1.0 = información compartida perfecta)")
    print(f"\n--- Inercia (menor es mejor) ---")
    print(f"Inercia k-means propio:  {inercia_propio:.2f}")
    print(f"Inercia sklearn:         {inercia_sklearn:.2f}")
    
    # Distribución de clústers
    print(f"\n--- Distribución de clústers ---")
    print(f"K-means propio:  {np.bincount(labels_propio)}")
    print(f"Sklearn:         {np.bincount(labels_sklearn)}")
    
    return {
        'labels_propio': labels_propio,
        'labels_sklearn': labels_sklearn,
        'centroids_propio': centroids_propio,
        'centroids_sklearn': centroids_sklearn,
        'ari': ari,
        'nmi': nmi,
        'inercia_propio': inercia_propio,
        'inercia_sklearn': inercia_sklearn
    }


def metodo_codo(data, max_k=10, seed=42):
    """Método del codo para elegir k óptimo."""
    data_list = data.tolist() if isinstance(data, np.ndarray) else data
    data_np = np.array(data_list)
    
    inercias = []
    ks = range(1, max_k + 1)
    
    for k in ks:
        km = SklearnKMeans(n_clusters=k, random_state=seed, n_init=10)
        km.fit(data_np)
        inercias.append(km.inertia_)
    
    return list(ks), inercias


# =============================================================================
# a) DATASET IRIS
# =============================================================================
print("\n" + "="*70)
print("EVALUACIÓN CON DATASET IRIS")
print("="*70)

iris = load_iris()
df_iris = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df_iris['target'] = iris.target

print(f"\nInformación del dataset:")
print(f"- Clases reales: {iris.target_names}")
print(f"- Distribución: {np.bincount(iris.target)}")

# Normalizar datos para mejor rendimiento
scaler_iris = StandardScaler()
iris_scaled = scaler_iris.fit_transform(iris.data)

# Método del codo
ks_iris, inercias_iris = metodo_codo(iris_scaled, max_k=10)

# Elegimos k=3 (sabemos que hay 3 especies)
resultados_iris = comparar_kmeans(iris_scaled, k=3, nombre_dataset="Iris (normalizado)")

# Comparar con etiquetas reales
print(f"\n--- Comparación con etiquetas reales ---")
ari_real_propio = adjusted_rand_score(iris.target, resultados_iris['labels_propio'])
ari_real_sklearn = adjusted_rand_score(iris.target, resultados_iris['labels_sklearn'])
print(f"ARI vs etiquetas reales (k-means propio): {ari_real_propio:.4f}")
print(f"ARI vs etiquetas reales (sklearn):        {ari_real_sklearn:.4f}")


# =============================================================================
# b) DATASET PENGUINS
# =============================================================================
print("\n" + "="*70)
print("EVALUACIÓN CON DATASET PENGUINS")
print("="*70)

penguins = sns.load_dataset("penguins")
print(f"\nInformación del dataset:")
print(f"- Shape original: {penguins.shape}")
print(f"- Columnas: {list(penguins.columns)}")
print(f"- Valores nulos por columna:\n{penguins.isnull().sum()}")

# Eliminar filas con valores nulos y seleccionar solo columnas numéricas
penguins_clean = penguins.dropna()
penguins_numeric = penguins_clean.select_dtypes(include=[np.number])
print(f"- Shape después de limpiar: {penguins_numeric.shape}")
print(f"- Especies: {penguins_clean['species'].unique()}")
print(f"- Distribución: {penguins_clean['species'].value_counts().values}")

# Normalizar
scaler_penguins = StandardScaler()
penguins_scaled = scaler_penguins.fit_transform(penguins_numeric)

# Método del codo
ks_penguins, inercias_penguins = metodo_codo(penguins_scaled, max_k=10)

# Elegimos k=3 (hay 3 especies)
resultados_penguins = comparar_kmeans(penguins_scaled, k=3, nombre_dataset="Penguins (normalizado)")

# Comparar con etiquetas reales
species_labels = pd.Categorical(penguins_clean['species']).codes
print(f"\n--- Comparación con etiquetas reales ---")
ari_real_propio = adjusted_rand_score(species_labels, resultados_penguins['labels_propio'])
ari_real_sklearn = adjusted_rand_score(species_labels, resultados_penguins['labels_sklearn'])
print(f"ARI vs etiquetas reales (k-means propio): {ari_real_propio:.4f}")
print(f"ARI vs etiquetas reales (sklearn):        {ari_real_sklearn:.4f}")


# =============================================================================
# c) DATASET WINE QUALITY RED
# =============================================================================
print("\n" + "="*70)
print("EVALUACIÓN CON DATASET WINE QUALITY RED")
print("="*70)

# Cargar datos (separador es ;)
wine_red = pd.read_csv("../datos/wine_quality/winequality-red.csv", sep=";")
print(f"\nInformación del dataset:")
print(f"- Shape: {wine_red.shape}")
print(f"- Columnas: {list(wine_red.columns)}")
print(f"- Valores de calidad: {sorted(wine_red['quality'].unique())}")
print(f"- Distribución de calidad:\n{wine_red['quality'].value_counts().sort_index()}")

# Usar solo características (sin la columna quality)
wine_features = wine_red.drop('quality', axis=1)

# Normalizar
scaler_wine = StandardScaler()
wine_scaled = scaler_wine.fit_transform(wine_features)

# Método del codo
ks_wine, inercias_wine = metodo_codo(wine_scaled, max_k=10)

# Elegimos k=3 (agrupación natural de calidad: baja, media, alta)
# También probaremos k=6 (número de valores únicos de calidad)
print("\n--- Prueba con k=3 ---")
resultados_wine_3 = comparar_kmeans(wine_scaled, k=3, nombre_dataset="Wine Red (k=3)")

print("\n--- Prueba con k=6 ---")
resultados_wine_6 = comparar_kmeans(wine_scaled, k=6, nombre_dataset="Wine Red (k=6)")

# Comparar con etiquetas de calidad
print(f"\n--- Comparación con etiquetas de calidad ---")
ari_real_propio_3 = adjusted_rand_score(wine_red['quality'], resultados_wine_3['labels_propio'])
ari_real_sklearn_3 = adjusted_rand_score(wine_red['quality'], resultados_wine_3['labels_sklearn'])
print(f"k=3: ARI vs calidad (k-means propio): {ari_real_propio_3:.4f}")
print(f"k=3: ARI vs calidad (sklearn):        {ari_real_sklearn_3:.4f}")

ari_real_propio_6 = adjusted_rand_score(wine_red['quality'], resultados_wine_6['labels_propio'])
ari_real_sklearn_6 = adjusted_rand_score(wine_red['quality'], resultados_wine_6['labels_sklearn'])
print(f"k=6: ARI vs calidad (k-means propio): {ari_real_propio_6:.4f}")
print(f"k=6: ARI vs calidad (sklearn):        {ari_real_sklearn_6:.4f}")


# =============================================================================
# VISUALIZACIONES
# =============================================================================
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Método del codo
axes[0, 0].plot(ks_iris, inercias_iris, 'bo-')
axes[0, 0].axvline(x=3, color='r', linestyle='--', label='k=3')
axes[0, 0].set_xlabel('k')
axes[0, 0].set_ylabel('Inercia')
axes[0, 0].set_title('Método del Codo - Iris')
axes[0, 0].legend()

axes[0, 1].plot(ks_penguins, inercias_penguins, 'go-')
axes[0, 1].axvline(x=3, color='r', linestyle='--', label='k=3')
axes[0, 1].set_xlabel('k')
axes[0, 1].set_ylabel('Inercia')
axes[0, 1].set_title('Método del Codo - Penguins')
axes[0, 1].legend()

axes[0, 2].plot(ks_wine, inercias_wine, 'ro-')
axes[0, 2].axvline(x=3, color='b', linestyle='--', label='k=3')
axes[0, 2].axvline(x=6, color='g', linestyle='--', label='k=6')
axes[0, 2].set_xlabel('k')
axes[0, 2].set_ylabel('Inercia')
axes[0, 2].set_title('Método del Codo - Wine Red')
axes[0, 2].legend()

# Scatter plots comparativos (usando PCA para 2D si es necesario)
from sklearn.decomposition import PCA

# Iris
pca_iris = PCA(n_components=2)
iris_2d = pca_iris.fit_transform(iris_scaled)
axes[1, 0].scatter(iris_2d[:, 0], iris_2d[:, 1], c=resultados_iris['labels_propio'], 
                   cmap='viridis', alpha=0.6, label='Propio')
axes[1, 0].set_title('Iris - K-means Propio')
axes[1, 0].set_xlabel('PC1')
axes[1, 0].set_ylabel('PC2')

# Penguins
pca_penguins = PCA(n_components=2)
penguins_2d = pca_penguins.fit_transform(penguins_scaled)
axes[1, 1].scatter(penguins_2d[:, 0], penguins_2d[:, 1], c=resultados_penguins['labels_propio'], 
                   cmap='viridis', alpha=0.6)
axes[1, 1].set_title('Penguins - K-means Propio')
axes[1, 1].set_xlabel('PC1')
axes[1, 1].set_ylabel('PC2')

# Wine
pca_wine = PCA(n_components=2)
wine_2d = pca_wine.fit_transform(wine_scaled)
axes[1, 2].scatter(wine_2d[:, 0], wine_2d[:, 1], c=resultados_wine_3['labels_propio'], 
                   cmap='viridis', alpha=0.6)
axes[1, 2].set_title('Wine Red - K-means Propio (k=3)')
axes[1, 2].set_xlabel('PC1')
axes[1, 2].set_ylabel('PC2')

plt.tight_layout()
plt.savefig('../figuras/comparacion_kmeans.png', dpi=150, bbox_inches='tight')
plt.show()

# =============================================================================
# RESUMEN Y DISCUSIÓN
# =============================================================================
print("\n" + "="*70)
print("RESUMEN Y DISCUSIÓN")
print("="*70)

print("""
SEMEJANZAS entre k-means propio y sklearn:
1. Ambos algoritmos convergen a soluciones similares (ARI cercano a 1).
2. Las distribuciones de clústers son muy parecidas.
3. La inercia (suma de distancias al cuadrado) es comparable.

DIFERENCIAS:
1. Sklearn usa 'k-means++' por defecto para inicialización (más inteligente),
   mientras que nuestro algoritmo usa inicialización aleatoria simple.
2. Sklearn ejecuta múltiples inicializaciones (n_init=10) y elige la mejor,
   nuestro algoritmo solo hace una ejecución.
3. Por lo anterior, sklearn tiende a obtener inercias ligeramente menores.
4. Las asignaciones pueden diferir debido a la naturaleza estocástica y
   la sensibilidad de k-means a la inicialización.

OBSERVACIONES POR DATASET:
- Iris: Buen rendimiento en ambos. k=3 es apropiado (3 especies).
- Penguins: Similar a Iris, k=3 funciona bien para las 3 especies.
- Wine: El clustering no corresponde bien con la calidad del vino,
  ya que la calidad no forma grupos esféricos naturales en el espacio
  de características. k=3 o k=6 capturan estructura diferente a la calidad.
""")

print("\nFigura guardada en: comparacion_kmeans.png")
