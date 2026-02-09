import math
import random
from typing import List, Tuple, Optional


def _euclidean_distance_sq(a: List[float], b: List[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _mean_point(points: List[List[float]], d: int) -> List[float]:
    if not points:
        return [0.0] * d
    sums = [0.0] * d
    for p in points:
        for i in range(d):
            sums[i] += p[i]
    count = float(len(points))
    return [s / count for s in sums]


def kmeans(
    data: List[List[float]],
    k: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    seed: Optional[int] = None,
) -> Tuple[List[int], List[List[float]]]:
    """
    K-means desde cero (sin librerías de datos ni funciones especializadas).

    Args:
        data: matriz n x d (lista de listas de floats).
        k: número de clústers.
        max_iter: número máximo de iteraciones.
        tol: tolerancia para detener (cambio máximo de centroides).
        seed: semilla aleatoria para reproducibilidad.

    Returns:
        labels: lista de tamaño n con la clase de cada dato.
        centroids: lista k x d con los centroides.
    """
    if k <= 0:
        raise ValueError("k debe ser mayor que 0")
    if not data:
        raise ValueError("data no puede ser vacío")

    n = len(data)
    d = len(data[0])
    if any(len(row) != d for row in data):
        raise ValueError("todas las filas de data deben tener la misma dimensión")
    if k > n:
        raise ValueError("k no puede ser mayor que el número de datos")

    rng = random.Random(seed)
    initial_indices = rng.sample(range(n), k)
    centroids = [list(data[idx]) for idx in initial_indices]

    labels = [0] * n

    for _ in range(max_iter):
        # Asignación de puntos a centroides
        for i, point in enumerate(data):
            best_cluster = 0
            best_dist = _euclidean_distance_sq(point, centroids[0])
            for c in range(1, k):
                dist = _euclidean_distance_sq(point, centroids[c])
                if dist < best_dist:
                    best_dist = dist
                    best_cluster = c
            labels[i] = best_cluster

        # Recalcular centroides
        clusters = [[] for _ in range(k)]
        for label, point in zip(labels, data):
            clusters[label].append(point)

        new_centroids = []
        for c in range(k):
            if clusters[c]:
                new_centroids.append(_mean_point(clusters[c], d))
            else:
                # Re-inicializar centroides vacíos con un punto aleatorio
                new_centroids.append(list(data[rng.randrange(n)]))

        # Comprobar convergencia
        max_shift = 0.0
        for old, new in zip(centroids, new_centroids):
            shift = math.sqrt(_euclidean_distance_sq(old, new))
            if shift > max_shift:
                max_shift = shift
        centroids = new_centroids

        if max_shift <= tol:
            break

    return labels, centroids


if __name__ == "__main__":
    # Ejemplo mínimo de uso
    data_example = [
        [1.0, 2.0],
        [1.5, 1.8],
        [5.0, 8.0],
        [8.0, 8.0],
        [1.0, 0.6],
        [9.0, 11.0],
    ]
    labels_out, centroids_out = kmeans(data_example, k=2, seed=42)
    print("labels:", labels_out)
    print("centroids:", centroids_out)
