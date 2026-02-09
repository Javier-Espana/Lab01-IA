from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Sequence, Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_sample_image
from kmeans_from_scratch import kmeans

try:
    from PIL import Image
except ImportError:
    Image = None

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CUSTOM_FOLDER = BASE_DIR / "figuras" / "cuantizacion_colores" / "imagenes"
DEFAULT_OUTPUT_DIR = BASE_DIR / "figuras" / "cuantizacion_colores"

def _slugify(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")

def _ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    arr = img.astype(np.float64)
    if arr.max() <= 1.0:
        arr *= 255.0
    arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)

def _ensure_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim != 3:
        raise ValueError("Debe tener 3 canales (RGB)")
    if img.shape[2] == 4:
        return img[:, :, :3]
    if img.shape[2] != 3:
        raise ValueError("Se espera imagen RGB o RGBA.")
    return img

def _load_image_from_path(path: Path) -> np.ndarray:
    if Image is not None:
        with Image.open(path) as pil_img:
            pil_img = pil_img.convert("RGB")
            return np.array(pil_img)
    arr = plt.imread(path)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return _ensure_rgb(_ensure_uint8(arr))


def _maybe_resize(image: np.ndarray, max_side: int | None) -> np.ndarray:
    if max_side is None:
        return image
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return image
    scale = max_side / float(longest)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    if Image is not None:
        pil_img = Image.fromarray(image)
        try:
            resample = Image.Resampling.LANCZOS
        except AttributeError:
            resample = Image.LANCZOS
        pil_img = pil_img.resize(new_size, resample=resample)
        return np.array(pil_img)
    step = int(np.ceil(longest / max_side))
    return image[::step, ::step, :]


def _generate_demo_images() -> List[Tuple[str, np.ndarray]]:
    demo = [
        ("china_sample", load_sample_image("china.jpg")),
        ("flower_sample", load_sample_image("flower.jpg")),
    ]
    size = 256
    x = np.linspace(0, 1, size)
    y = np.linspace(0, 1, size)
    xv, yv = np.meshgrid(x, y)
    grad_r = (xv * 255).astype(np.uint8)
    grad_g = (yv * 255).astype(np.uint8)
    grad_b = ((1 - xv) * 255).astype(np.uint8)
    demo.append(("gradient", np.stack([grad_r, grad_g, grad_b], axis=-1)))
    return demo


def _list_images_in_dir(folder: Path) -> List[Tuple[str, np.ndarray]]:
    if not folder.exists():
        raise FileNotFoundError(f"No se encontró la carpeta de imágenes: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"La ruta no es una carpeta: {folder}")

    soportados = {".png", ".jpg", ".jpeg", ".bmp"}
    imagenes: List[Tuple[str, np.ndarray]] = []
    for file in sorted(folder.iterdir()):
        if not file.is_file():
            continue
        if file.suffix.lower() not in soportados:
            continue
        arr = _load_image_from_path(file)
        imagenes.append((file.stem, arr))

    if not imagenes:
        raise ValueError(
            f"No se encontraron archivos de imagen válidos en {folder}. Revisar extensiones soportadas: {', '.join(sorted(soportados))}."
        )
    return imagenes


def _ask_image_source() -> str:
    prompt = (
        "Selecciona el origen de las imágenes:\n"
        "1 - Conjunto estándar (china, flower, gradiente)\n"
        "2 - Carpeta personalizada (procesa todas las imágenes disponibles)\n"
        "Ingresa 1 o 2: "
    )
    while True:
        choice = input(prompt).strip()
        if choice in {"1", "2"}:
            return "demo" if choice == "1" else "folder"
        print("Entrada inválida. Debes escribir 1 o 2.")


def _assign_to_centroids(pixels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    diff = pixels[:, None, :] - centroids[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    return np.argmin(dist_sq, axis=1)


def cuantizar_imagen(
    image: np.ndarray,
    k: int,
    seed: int,
    sample_pixels: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    rgb = _ensure_rgb(_ensure_uint8(image))
    h, w, c = rgb.shape
    flat_pixels = rgb.reshape(-1, c).astype(float)

    if sample_pixels is not None and sample_pixels < flat_pixels.shape[0]:
        rng = np.random.default_rng(seed)
        subset_idx = rng.choice(flat_pixels.shape[0], size=sample_pixels, replace=False)
        train_data = flat_pixels[subset_idx]
    else:
        train_data = flat_pixels

    etiquetas, centroides = kmeans(train_data.tolist(), k=k, seed=seed)
    centroids_arr = np.array(centroides)

    full_labels = _assign_to_centroids(flat_pixels, centroids_arr)
    labels_arr = full_labels.reshape(h, w)
    cuantizada = centroids_arr[full_labels].reshape(h, w, c)
    cuantizada = np.clip(cuantizada, 0, 255).astype(np.uint8)
    return labels_arr, cuantizada

def guardar_resultados(
    nombre: str,
    labels: np.ndarray,
    cuantizada: np.ndarray,
    original: np.ndarray,
    k: int,
    output_dir: Path,
) -> None:
    slug = _slugify(nombre)
    output_dir.mkdir(parents=True, exist_ok=True)

    cuantizada_path = output_dir / f"{slug}_k{k}_quantizada.png"
    plt.imsave(cuantizada_path, cuantizada)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle(f"Cuantización de colores - {nombre} (k={k})", fontsize=12)

    axes[0].imshow(original)
    axes[0].set_title("Imagen original")
    axes[0].axis("off")

    axes[1].imshow(labels, cmap="tab20", interpolation="nearest")
    axes[1].set_title("Mapa de clases")
    axes[1].axis("off")

    axes[2].imshow(cuantizada)
    axes[2].set_title("Imagen cuantizada")
    axes[2].axis("off")

    fig.tight_layout()
    figura_path = output_dir / f"{slug}_k{k}_resumen.png"
    fig.savefig(figura_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Guardado mapa/clases en: {figura_path.resolve()}")
    print(f"Guardada imagen cuantizada en: {cuantizada_path.resolve()}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cuantización de colores usando k-means propio",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--images",
        nargs="*",
        help="Rutas a imágenes RGB personalizadas. Si se omite, se usan muestras de sklearn.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=8,
        help="Número de colores (clusters) para la cuantización.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla aleatoria para reproducibilidad.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Carpeta donde se guardarán las figuras.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=512,
        help="Redimensiona la imagen para que su lado mayor no exceda este valor.",
    )
    parser.add_argument(
        "--sample-pixels",
        type=int,
        default=20000,
        help="Número máximo de píxeles usados para entrenar k-means (resto se asigna por distancia).",
    )
    parser.add_argument(
        "--source",
        choices=["demo", "folder"],
        help="Permite fijar sin interacción si se usan imágenes demo o una carpeta completa.",
    )
    parser.add_argument(
        "--folder-path",
        type=Path,
        default=DEFAULT_CUSTOM_FOLDER,
        help="Carpeta que contiene las imágenes personalizadas a cuantizar cuando se elige 'folder'.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.k < 2:
        raise ValueError("k debe ser al menos 2 para cuantizar colores")

    if args.images:
        imagenes = []
        for ruta in args.images:
            path = Path(ruta)
            if not path.exists():
                raise FileNotFoundError(f"No se encontró la imagen: {path}")
            arr = _load_image_from_path(path)
            imagenes.append((path.stem, arr))
    else:
        source = args.source or _ask_image_source()
        if source == "demo":
            print("Usando conjunto demo (china, flower, gradiente).")
            imagenes = _generate_demo_images()
        else:
            print(f"Procesando todas las imágenes en {args.folder_path.resolve()}.")
            imagenes = _list_images_in_dir(args.folder_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for nombre, img in imagenes:
        print(f"\nProcesando '{nombre}' (k={args.k})...")
        reduced = _maybe_resize(_ensure_uint8(img), args.max_side)
        labels, cuantizada = cuantizar_imagen(
            reduced,
            args.k,
            args.seed,
            sample_pixels=args.sample_pixels,
        )
        guardar_resultados(nombre, labels, cuantizada, reduced, args.k, args.output_dir)

    print("\nProceso completado. Revisa la carpeta de salida para las figuras.")

if __name__ == "__main__":
    main()
