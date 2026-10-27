import subprocess
from pathlib import Path
from PIL import Image
import numpy as np

# Posición vertical del mentón por cada pose (labio superior fijo en y_top=265)
CHIN_Y = {
    "5": 315.0,  # Boca cerrada (M, B, P, silencio)
    "1": 332.0,  # Sonrisa / dientes (I, S, Z, C, F, V)
    "4": 332.0,  # Boca redonda media (O)
    "6": 330.0,  # Labios fruncidos (U, W, Q)
    "3": 338.0,  # Boca media abierta (E, K, G, J)
    "2": 352.0,  # Lengua arriba (L, N, R, D, T)
    "7": 392.0,  # Boca muy abierta (A)
}

# Escala horizontal de las comisuras respecto al centro (x=626)
WIDTH_X = {
    "5": 1.00,
    "1": 1.02,
    "3": 1.00,
    "2": 1.03,
    "4": 0.85,
    "6": 0.80,
    "7": 1.12,
}


def bilinear_sample(img: np.ndarray, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    H, W, _ = img.shape
    x0 = np.clip(np.floor(X).astype(np.int32), 0, W - 1)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y0 = np.clip(np.floor(Y).astype(np.int32), 0, H - 1)
    y1 = np.clip(y0 + 1, 0, H - 1)
    wx = (X - x0)[..., None]
    wy = (Y - y0)[..., None]
    return (
        img[y0, x0] * (1.0 - wx) * (1.0 - wy)
        + img[y0, x1] * wx * (1.0 - wy)
        + img[y1, x0] * (1.0 - wx) * wy
        + img[y1, x1] * wx * wy
    )


def warp_mouth_to_target(img: np.ndarray, y_src: float, y_dst: float, w_src: float, w_dst: float) -> np.ndarray:
    H, W, _ = img.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    # Máscara horizontal centrada en el hocico/mentón (x=626)
    wx = np.exp(-0.5 * ((xx - 626.0) / 34.0) ** 2) * np.clip(1.0 - np.abs(xx - 626.0) / 80.0, 0.0, 1.0)
    y_top, y_bot = 265.0, 418.0

    dy = np.zeros_like(yy)
    mask_upper = (yy >= y_top) & (yy <= y_dst)
    dy[mask_upper] = (yy[mask_upper] - y_top) * ((y_src - y_top) / (y_dst - y_top) - 1.0)
    mask_lower = (yy > y_dst) & (yy <= y_bot)
    dy[mask_lower] = (y_bot - yy[mask_lower]) * ((y_src - y_dst) / (y_bot - y_dst))

    wy = np.clip((yy - 258.0) / 18.0, 0.0, 1.0) * np.clip((422.0 - yy) / 18.0, 0.0, 1.0)

    wy_mouth = np.exp(-0.5 * ((yy - 300.0) / 35.0) ** 2)
    scale_ratio = w_src / w_dst
    dx = (xx - 626.0) * (scale_ratio - 1.0) * wx * wy_mouth

    src_X = xx + dx
    src_Y = yy + dy * wx * wy
    return bilinear_sample(img, src_X, src_Y)


def ensure_inbetweens(avatar_dir: Path) -> Path:
    """Genera (si no existen) las poses estabilizadas y todos los cuadros intermedios en avatar/inbetweens."""
    out_dir = avatar_dir / "inbetweens"
    out_dir.mkdir(exist_ok=True)

    sentinel = out_dir / "resp_75.png"
    if sentinel.exists() and (out_dir / "mid_5_7.png").exists():
        return out_dir

    print("Generando cuadros intermedios (Mesh-Warp + Optical Flow) para suavizar el Avatar...")
    base_body = np.array(Image.open(avatar_dir / "5_sin_fondo.png").convert("RGBA"), dtype=np.float32)
    H, W, _ = base_body.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

    # Máscara suave exclusiva del hocico/mandíbula para que el cuerpo y las orejas no vibren al hablar
    mouth_roi_mask = np.clip(
        1.0 - (((xx - 626.0) / 88.0) ** 2 + ((yy - 332.0) / 92.0) ** 2 - 0.55) / 0.45,
        0.0,
        1.0,
    )[..., None]

    poses = ["1", "2", "3", "4", "5", "6", "7"]
    pose_imgs = {}
    for p in poses:
        raw = np.array(Image.open(avatar_dir / f"{p}_sin_fondo.png").convert("RGBA"), dtype=np.float32)
        if p == "5":
            stabilized = base_body
        else:
            stabilized = base_body * (1.0 - mouth_roi_mask) + raw * mouth_roi_mask
        pose_imgs[p] = stabilized
        Image.fromarray(np.clip(stabilized, 0, 255).astype(np.uint8), "RGBA").save(out_dir / f"pose_{p}.png")

    # 1. Generar intermedios entre cada par de poses vocales (Mesh-Warp bidireccional)
    for i, pa in enumerate(poses):
        for pb in poses[i + 1 :]:
            img_a = pose_imgs[pa]
            img_b = pose_imgs[pb]
            ya, yb = CHIN_Y[pa], CHIN_Y[pb]
            wa, wb = WIDTH_X[pa], WIDTH_X[pb]

            yt = 0.5 * (ya + yb)
            wt = 0.5 * (wa + wb)

            warped_a = warp_mouth_to_target(img_a, ya, yt, wa, wt)
            warped_b = warp_mouth_to_target(img_b, yb, yt, wb, wt)

            mouth_blend = 0.5 * warped_a + 0.5 * warped_b
            final_mid = base_body * (1.0 - mouth_roi_mask) + mouth_blend * mouth_roi_mask
            mid_pil = Image.fromarray(np.clip(final_mid, 0, 255).astype(np.uint8), "RGBA")

            mid_pil.save(out_dir / f"mid_{pa}_{pb}.png")
            mid_pil.save(out_dir / f"mid_{pb}_{pa}.png")

    # 2. Generar 3 pasos intermedios de respiración (5 <-> inhala) usando FFmpeg Optical Flow (minterpolate)
    img_5 = avatar_dir / "5_sin_fondo.png"
    img_inhala = avatar_dir / "inhala_sin_fondo.png"
    tmp_pattern = out_dir / "tmp_resp_%02d.png"
    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(img_5),
        "-i", str(img_5),
        "-i", str(img_inhala),
        "-i", str(img_inhala),
        "-filter_complex",
        "[0:v][1:v][2:v][3:v]concat=n=4:v=1:a=0,settb=1/100,setpts=N*100,format=yuva444p,"
        "minterpolate=fps=4:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1:scd=none,format=rgba",
        str(tmp_pattern),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Los frames 06, 07 y 08 corresponden a 25%, 50% y 75% de transición entre 5 e inhala
    mapping = {
        "tmp_resp_06.png": "resp_25.png",
        "tmp_resp_07.png": "resp_50.png",
        "tmp_resp_08.png": "resp_75.png",
    }
    for src_name, dst_name in mapping.items():
        src_p = out_dir / src_name
        dst_p = out_dir / dst_name
        if src_p.exists():
            if dst_p.exists():
                dst_p.unlink()
            src_p.rename(dst_p)

    for tmp_f in out_dir.glob("tmp_resp_*.png"):
        tmp_f.unlink()

    print(f"¡Cuadros intermedios listos en {out_dir}!")
    return out_dir


if __name__ == "__main__":
    av_dir = Path(__file__).parent.absolute() / "avatar"
    # Limpiar previos para regenerar limpio
    inb = av_dir / "inbetweens"
    if inb.exists():
        for f in inb.glob("*.png"):
            f.unlink()
    ensure_inbetweens(av_dir)
