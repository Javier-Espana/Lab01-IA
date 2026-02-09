# Lab 01 - Métodos de Clustering
## Inteligencia Artificial 2026

**Fecha:** 8 de febrero de 2026

---


## 📝 Ejercicios Completados

### Ejercicio 1: K-means desde cero
**Archivo:** `codigo/kmeans_from_scratch.py`

Implementación del algoritmo k-means sin librerías especializadas:
- **Input:** Matriz de datos (n × d), número de clusters (k)
- **Output:** Vector de labels (n), matriz de centroides (k × d)

### Ejercicio 2: Evaluación con datasets
**Archivo:** `codigo/evaluar_kmeans.py`  
**Figura:** `figuras/comparacion_kmeans.png`

Evaluación comparativa con tres datasets:
| Dataset | Muestras | Características | k elegido |
|---------|----------|-----------------|-----------|
| Iris | 150 | 4 | 3 |
| Penguins | 333 | 4 | 3 |
| Wine Red | 1599 | 11 | 3 |

### Ejercicio 3: Agrupamiento jerárquico
**Archivo:** `codigo/agrupamiento_jerarquico.py`  
**Figuras:** `figuras/dendrogramas_*.png`

Dendrogramas variando:
- **Métodos:** Simple, Completo, Promedio, Ward
- **Métricas:** Euclideana, Hamming

### Ejercicio 4: K-means países + comparación
**Archivo:** `codigo/kmeans_paises.py`  
**Figura:** `figuras/kmeans_paises_comparacion.png`

Comparación K-means vs Jerárquico (ver análisis detallado abajo).

---

## 🔬 Análisis Ejercicio 4: ¿Son iguales las agrupaciones?

### Resultados con k=3

**K-means:**
- Cluster 0: Brazil, Burma, Egypt, Indonesia, Jordan
- Cluster 1: India, Israel, Netherlands, UK, USA
- Cluster 2: China, Cuba, Poland, USSR

**Jerárquico (Ward/Complete con Hamming):**
- Cluster 1: Israel, Netherlands, UK, USA
- Cluster 2: China, Cuba, Poland, USSR
- Cluster 3: Brazil, Burma, Egypt, India, Indonesia, Jordan

**ARI entre métodos:** 0.76 (similar pero no idéntico)

### ¿Por qué son diferentes?

| Factor | K-means | Jerárquico |
|--------|---------|------------|
| Algoritmo | Iterativo, particional | Aglomerativo |
| Inicialización | Aleatoria | Determinístico |
| Métrica | Euclideana implícita | Puede usar Hamming |
| Forma clusters | Esféricos | Cualquier forma |
| Optimización | Global (inercia) | Local (greedy) |

### Conclusión

Las agrupaciones son **similares pero no idénticas**. Ambos identifican correctamente:
- ✅ Bloque occidental: Israel, Netherlands, UK, USA
- ✅ Bloque comunista: China, Cuba, Poland, USSR

La diferencia principal está en los países en desarrollo. Para datos binarios, **Hamming es más apropiada**, por lo que el jerárquico con Hamming captura mejor la estructura.

---

## ⚙️ Ejecución

```bash
# Crear entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependencias
pip install numpy pandas matplotlib seaborn scikit-learn scipy openpyxl

# Ejecutar scripts (desde la carpeta raíz)
cd codigo
python kmeans_from_scratch.py
python evaluar_kmeans.py
python agrupamiento_jerarquico.py
python kmeans_paises.py
```

---

## 📦 Dependencias

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- scipy
- openpyxl
