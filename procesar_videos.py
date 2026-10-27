import os
import subprocess
from pathlib import Path
import concurrent.futures
import argparse

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Instalando faster-whisper...")
    subprocess.check_call(["pip", "install", "faster-whisper"])
    from faster_whisper import WhisperModel

from generar_intermedios import ensure_inbetweens


def format_timestamp(seconds: float) -> str:
    milliseconds = int((seconds - int(seconds)) * 1000)
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"


def generate_srt(segments, srt_path):
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, segment in enumerate(segments, start=1):
            start = format_timestamp(segment.start)
            end = format_timestamp(segment.end)
            f.write(f"{i}\n")
            f.write(f"{start} --> {end}\n")
            f.write(f"{segment.text.strip()}\n\n")


# Mapa fonético corregido a las 7 poses reales del ocelote:
# '5': Boca cerrada neutra (M, B, P y reposo)
# '1': Dientes juntos / comisura abierta (I, Y, S, Z, C, F, V)
# '2': Lengua en paladar/dientes (L, N, Ñ, R, D, T)
# '3': Boca media abierta (E, K, G, J, X)
# '4': Boca redonda media (O)
# '6': Labios fruncidos (U, W, Q)
# '7': Boca muy abierta (A)
CHAR_TO_POSE = {
    "a": "7", "á": "7",
    "e": "3", "é": "3",
    "i": "1", "í": "1", "y": "1",
    "o": "4", "ó": "4",
    "u": "6", "ú": "6", "ü": "6", "w": "6",
    "m": "5", "b": "5", "p": "5",
    "f": "1", "v": "1", "s": "1", "z": "1", "c": "1",
    "l": "2", "n": "2", "ñ": "2", "r": "2", "d": "2", "t": "2",
    "k": "3", "g": "3", "j": "3", "x": "3",
    "q": "6",
}

VOWELS = set("aeiouáéíóúü")


def word_to_visemes(raw_word: str, duration: float) -> list[str]:
    """Convierte una palabra en una secuencia de poses fonéticas naturales sin efecto metralleta."""
    w = "".join(c.lower() for c in raw_word if c.isalpha())
    if not w:
        return []

    # Limpiar letras mudas y dígrafos del español
    w = w.replace("qu", "k").replace("gue", "ge").replace("gui", "gi")
    w = w.replace("ch", "c").replace("ll", "y").replace("rr", "r").replace("h", "")
    if not w:
        return ["3"]

    raw_poses = [CHAR_TO_POSE.get(c, "3") for c in w]

    # Colapsar poses idénticas consecutivas
    dedup = [raw_poses[0]]
    for p in raw_poses[1:]:
        if p != dedup[-1]:
            dedup.append(p)

    # Limitar la cantidad máxima de poses según la duración de la palabra (~11 poses/seg máx)
    # para evitar el efecto "Muppet" en palabras rápidas
    max_poses = max(1, int(round(duration / 0.085)))
    if len(dedup) > max_poses:
        # Priorizar vocales y consonantes labiales ('5') que dan el ritmo silábico visual
        priority_poses = []
        for idx, ch in enumerate(w):
            p = CHAR_TO_POSE.get(ch, "3")
            if ch in VOWELS or p == "5" or idx == 0 or idx == len(w) - 1:
                if not priority_poses or priority_poses[-1] != p:
                    priority_poses.append(p)
        if len(priority_poses) > max_poses:
            # Submuestreo uniforme conservando inicio y fin
            indices = [int(round(i * (len(priority_poses) - 1) / (max_poses - 1))) for i in range(max_poses)] if max_poses > 1 else [0]
            dedup = [priority_poses[i] for i in indices]
        else:
            dedup = priority_poses if priority_poses else dedup[:max_poses]

    return dedup


def generate_text_based_avatar(segments, output_txt_path: Path, base_dir: Path) -> bool:
    avatar_dir = base_dir / "avatar"
    if not (avatar_dir / "5_sin_fondo.png").exists():
        return False

    inb_dir = ensure_inbetweens(avatar_dir)

    def pose_path(p: str) -> str:
        cand = inb_dir / f"pose_{p}.png"
        return cand.as_posix() if cand.exists() else (avatar_dir / f"{p}_sin_fondo.png").as_posix()

    def mid_path(pa: str, pb: str) -> str | None:
        if pa == pb:
            return None
        cand = inb_dir / f"mid_{pa}_{pb}.png"
        return cand.as_posix() if cand.exists() else None

    img_inhala = (avatar_dir / "inhala_sin_fondo.png").as_posix()
    r25 = (inb_dir / "resp_25.png").as_posix()
    r50 = (inb_dir / "resp_50.png").as_posix()
    r75 = (inb_dir / "resp_75.png").as_posix()

    # Ciclo suave de respiración con cuadros intermedios por Optical Flow
    breath_cycle = [
        (pose_path("5"), 0.35),
        (r25, 0.12),
        (r50, 0.12),
        (r75, 0.12),
        (img_inhala, 0.40),
        (r75, 0.12),
        (r50, 0.12),
        (r25, 0.12),
    ]

    entries: list[tuple[str, float]] = []
    curr_pose = "5"

    def append_frame(img_p: str, dur: float):
        if dur <= 0.005:
            return
        if entries and entries[-1][0] == img_p:
            entries[-1] = (img_p, entries[-1][1] + dur)
        else:
            entries.append((img_p, dur))

    def transition_to(target_pose: str, total_dur: float):
        nonlocal curr_pose
        if target_pose == curr_pose or total_dur < 0.045:
            append_frame(pose_path(target_pose), total_dur)
            curr_pose = target_pose
            return

        # Si salta de cerrada ('5') a muy abierta ('7') y hay tiempo suficiente, pasar por media ('3')
        if {curr_pose, target_pose} == {"5", "7"} and total_dur >= 0.13:
            t_mid1 = total_dur * 0.18
            t_pose3 = total_dur * 0.20
            t_mid2 = total_dur * 0.18
            t_main = total_dur - (t_mid1 + t_pose3 + t_mid2)
            m1 = mid_path(curr_pose, "3")
            m2 = mid_path("3", target_pose)
            if m1:
                append_frame(m1, t_mid1)
            append_frame(pose_path("3"), t_pose3)
            if m2:
                append_frame(m2, t_mid2)
            append_frame(pose_path(target_pose), t_main)
            curr_pose = target_pose
            return

        # Insertar cuadro intermedio (Mesh-Warp) al inicio de la transición
        mid_img = mid_path(curr_pose, target_pose)
        if mid_img:
            mid_dur = min(0.038, total_dur * 0.38)
            main_dur = total_dur - mid_dur
            append_frame(mid_img, mid_dur)
            append_frame(pose_path(target_pose), main_dur)
        else:
            append_frame(pose_path(target_pose), total_dur)
        curr_pose = target_pose

    t = 0.0
    for segment in segments:
        if not hasattr(segment, "words") or not segment.words:
            continue

        for word_obj in segment.words:
            # Manejo de silencios / pausas entre palabras
            if word_obj.start > t:
                gap = word_obj.start - t
                if gap <= 0.30:
                    # Pausa breve entre palabras: cerrar suavemente a boca neutra '5' sin cabecear
                    transition_to("5", gap)
                else:
                    # Pausa larga: primero cerrar suavemente la boca a '5'
                    close_time = min(0.12, gap * 0.25)
                    transition_to("5", close_time)
                    rem = gap - close_time
                    idx = 0
                    while rem > 0.005:
                        b_img, b_dur = breath_cycle[idx]
                        step = min(b_dur, rem)
                        append_frame(b_img, step)
                        rem -= step
                        idx = (idx + 1) % len(breath_cycle)
                    curr_pose = "5"
                t = word_obj.start

            word_dur = max(0.04, word_obj.end - word_obj.start)
            visemes = word_to_visemes(word_obj.word, word_dur)
            if visemes:
                step_dur = word_dur / len(visemes)
                for v_pose in visemes:
                    transition_to(v_pose, step_dur)
            else:
                transition_to("5", word_dur)

            t = word_obj.end

    # Cerrar boca suavemente al finalizar el video
    transition_to("5", 0.15)

    with open(output_txt_path, "w", encoding="utf-8") as f:
        for img_p, dur in entries:
            f.write(f"file '{img_p}'\n")
            f.write(f"duration {dur:.3f}\n")
        f.write(f"file '{pose_path('5')}'\n")

    return True


print("\nCargando modelo de IA para subtítulos (Word-Level Lip-Sync + In-Betweens activo)...")
model = WhisperModel("small", device="cpu", compute_type="int8")


def process_single_video(video_path, output_dir, base_dir, logo1, logo2, logo3, use_avatar, use_branding, use_left):
    audio_path = output_dir / f"{video_path.stem}.wav"
    srt_path = output_dir / f"{video_path.stem}.srt"
    avatar_txt_path = output_dir / f"{video_path.stem}_avatar.txt"
    final_video_path = output_dir / f"{video_path.stem}_final.mp4"

    print(f"[{video_path.name}] Iniciando procesamiento...")

    # 1. Extraer audio
    if not audio_path.exists() and (not srt_path.exists() or srt_path.stat().st_size == 0 or use_avatar):
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                str(audio_path),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    # 2. Generar Transcripción con timestamps por palabra
    segments_gen, info = model.transcribe(str(audio_path), language="es", beam_size=5, word_timestamps=True)
    segments = list(segments_gen)

    if not srt_path.exists() or srt_path.stat().st_size == 0:
        generate_srt(segments, str(srt_path))

    # 3. Generar secuencia de Avatar fluida con cuadros intermedios
    has_avatar = False
    if use_avatar:
        has_avatar = generate_text_based_avatar(segments, avatar_txt_path, base_dir)

    # Limpiar audio temporal
    if audio_path.exists():
        audio_path.unlink()

    # 4. Superponer Logos, Avatar y Subtítulos
    rel_srt = srt_path.name
    ffmpeg_cmd = ["ffmpeg", "-y", "-i", str(video_path)]

    if use_branding:
        ffmpeg_cmd.extend(["-i", str(logo1), "-i", str(logo2), "-i", str(logo3)])

    if has_avatar:
        ffmpeg_cmd.extend(["-f", "concat", "-safe", "0", "-i", str(avatar_txt_path)])

    filter_complex = ""
    curr_stream = "[0:v]"

    if use_branding:
        s3_x = "W-w-10" if use_left else "10"
        l3_x = "W-w-20" if use_left else "20"

        filter_complex += (
            "[1:v]scale=90:-1,split[l1a][l1b];"
            "[l1a]pad=w=iw+20:h=ih+20:x=10:y=10:color=black@0,colorchannelmixer=0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0.7,gblur=sigma=4[s1];"
            "[2:v]scale=90:-1,split[l2a][l2b];"
            "[l2a]pad=w=iw+20:h=ih+20:x=10:y=10:color=black@0,colorchannelmixer=0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0.7,gblur=sigma=4[s2];"
            "[3:v]scale=90:-1,split[l3a][l3b];"
            "[l3a]pad=w=iw+20:h=ih+20:x=10:y=10:color=black@0,colorchannelmixer=0:0:0:0:0:0:0:0:0:0:0:0:0:0:0:0.7,gblur=sigma=4[s3];"
            f"{curr_stream}[s1]overlay=10:10[v1a];"
            "[v1a][s2]overlay=W-w-10:10[v2a];"
            f"[v2a][s3]overlay={s3_x}:H-h-10[v3a];"
            "[v3a][l1b]overlay=20:20[v1b];"
            "[v1b][l2b]overlay=W-w-20:20[v2b];"
            f"[v2b][l3b]overlay={l3_x}:H-h-20[curr];"
        )
        curr_stream = "[curr]"

    if has_avatar:
        avatar_idx = 4 if use_branding else 1
        av_x = "0" if use_left else "W-w"
        filter_complex += (
            f"[{avatar_idx}:v]scale=400:-1,format=rgba[av];"
            f"{curr_stream}[av]overlay={av_x}:H-h:format=auto:eof_action=repeat[curr_av];"
        )
        curr_stream = "[curr_av]"

    if filter_complex:
        filter_complex += f"{curr_stream}subtitles={rel_srt}[vout]"
        ffmpeg_cmd.extend(["-filter_complex", filter_complex, "-map", "[vout]"])
    else:
        ffmpeg_cmd.extend(["-vf", f"subtitles={rel_srt}", "-map", "0:v"])

    ffmpeg_cmd.extend([
        "-map", "0:a",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "23",
        "-c:a", "aac",
        "-b:a", "192k",
        str(final_video_path),
    ])

    subprocess.run(ffmpeg_cmd, cwd=str(output_dir), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if avatar_txt_path.exists():
        avatar_txt_path.unlink()

    return video_path.name


def main():
    parser = argparse.ArgumentParser(description="Procesador automático de videos con logos, subtítulos y avatar.")
    parser.add_argument("--avatar", action="store_true", help="Si se especifica, procesa y añade el avatar al video.")
    parser.add_argument("--branding", action="store_true", help="Si se especifica, añade los 3 logos al video.")
    parser.add_argument("--left", action="store_true", help="Coloca el avatar a la izquierda y el logo 3 a la derecha.")
    args = parser.parse_args()

    base_dir = Path(__file__).parent.absolute()
    input_dir = base_dir / "input"
    logos_dir = base_dir / "logos"
    output_dir = base_dir / "output"

    if args.avatar:
        ensure_inbetweens(base_dir / "avatar")

    video_files = list(input_dir.glob("*.mp4")) + list(input_dir.glob("*.mkv")) + list(input_dir.glob("*.mov"))

    if not video_files:
        print("--- No se encontraron videos en la carpeta input ---")
        return

    logo1, logo2, logo3 = None, None, None
    if args.branding:
        logo_files = list(logos_dir.glob("*.png"))
        if len(logo_files) < 3:
            print("ADVERTENCIA: Faltan logos. Se necesitan 3 para --branding.")
            return
        logo1, logo2, logo3 = logo_files[:3]

    max_workers = min(4, len(video_files))
    print(f"\n==========================================")
    print(f"Iniciando procesamiento PARALELO ({max_workers} videos al mismo tiempo)...")
    print(f"Modo Avatar: {'Activado' if args.avatar else 'Desactivado'}")
    print(f"Modo Branding (Logos): {'Activado' if args.branding else 'Desactivado'}")
    print(f"Posición Avatar: {'Izquierda' if args.left else 'Derecha'}")
    print(f"==========================================\n")

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for video_path in video_files:
            futures.append(
                executor.submit(
                    process_single_video,
                    video_path,
                    output_dir,
                    base_dir,
                    logo1,
                    logo2,
                    logo3,
                    args.avatar,
                    args.branding,
                    args.left,
                )
            )

        for future in concurrent.futures.as_completed(futures):
            try:
                name = future.result()
                print(f"¡Completado con éxito! -> {name}")
            except Exception as e:
                print(f"Error procesando un video: {e}")

    print("\n--- TODOS LOS VIDEOS HAN SIDO PROCESADOS ---")


if __name__ == "__main__":
    main()
