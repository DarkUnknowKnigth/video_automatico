from pathlib import Path
from collections import deque
from PIL import Image
import numpy as np


def remove_white_background(img_path: Path, out_path: Path):
    """Elimina el fondo blanco usando Flood-Fill desde el exterior para no perforar ojos, dientes ni pecho."""
    arr = np.array(Image.open(img_path).convert("RGBA"), dtype=np.float32)
    H, W, _ = arr.shape
    rgb = arr[:, :, :3]

    min_c = rgb.min(axis=2)
    max_diff = rgb.max(axis=2) - min_c
    is_bg_candidate = (min_c > 212) & (max_diff < 28)

    visited = np.zeros((H, W), dtype=bool)
    q = deque()

    # Semillas en los 4 bordes de la imagen
    for x in range(W):
        if is_bg_candidate[0, x]:
            visited[0, x] = True
            q.append((0, x))
        if is_bg_candidate[H - 1, x]:
            visited[H - 1, x] = True
            q.append((H - 1, x))
    for y in range(H):
        if is_bg_candidate[y, 0]:
            visited[y, 0] = True
            q.append((y, 0))
        if is_bg_candidate[y, W - 1]:
            visited[y, W - 1] = True
            q.append((y, W - 1))

    # Semilla adicional en el hueco cerrado entre el brazo izquierdo/libro y el chaleco
    for sy, sx in [(645, 755), (635, 750)]:
        if 0 <= sy < H and 0 <= sx < W and is_bg_candidate[sy, sx] and not visited[sy, sx]:
            visited[sy, sx] = True
            q.append((sy, sx))

    while q:
        y, x = q.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < H and 0 <= nx < W and not visited[ny, nx] and is_bg_candidate[ny, nx]:
                visited[ny, nx] = True
                q.append((ny, nx))

    alpha = np.where(visited, 0.0, 255.0)

    # Suavizado (anti-aliasing y de-fringe) en el borde exterior de 2 píxeles
    dilated_1 = (
        np.pad(visited[1:, :], ((0, 1), (0, 0)))
        | np.pad(visited[:-1, :], ((1, 0), (0, 0)))
        | np.pad(visited[:, 1:], ((0, 0), (0, 1)))
        | np.pad(visited[:, :-1], ((0, 0), (1, 0)))
    )
    edge_1 = (~visited) & dilated_1
    edge_whiteness = np.clip((min_c - 175.0) / (248.0 - 175.0), 0.0, 1.0)
    alpha[edge_1] = 255.0 * (1.0 - edge_whiteness[edge_1])

    # Descontaminar el halo blanco en el borde (de-fringe)
    a_norm = np.clip(alpha / 255.0, 0.05, 1.0)
    for c in range(3):
        arr[edge_1, c] = np.clip((arr[edge_1, c] - (1.0 - a_norm[edge_1]) * 255.0) / a_norm[edge_1], 0.0, 255.0)

    arr[:, :, 3] = alpha
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGBA").save(out_path, "PNG")
    print(f"Procesado sin huecos internos: {out_path.name}")


if __name__ == "__main__":
    avatar_dir = Path(__file__).parent.absolute() / "avatar"
    names = ["1", "2", "3", "4", "5", "6", "7", "inhala", "exala"]
    for name in names:
        img_path = avatar_dir / f"{name}.png"
        out_path = avatar_dir / f"{name}_sin_fondo.png"
        if img_path.exists():
            remove_white_background(img_path, out_path)
