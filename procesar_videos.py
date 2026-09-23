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

def generate_text_based_avatar(segments, output_txt_path, base_dir):
    avatar_dir = base_dir / "avatar"
    img_silence = (avatar_dir / "inhala_sin_fondo.png").as_posix() # Inhala como boca cerrada y silencio
    img_inhala = (avatar_dir / "inhala_sin_fondo.png").as_posix()
    img_exala = (avatar_dir / "exala_sin_fondo.png").as_posix()
    
    # Mapa de visemas (Letra -> Imagen)
    VISEME_MAP = {
        # A, I (Boca abierta/sonrisa)
        'a': '2_sin_fondo.png', 'i': '2_sin_fondo.png', 'y': '2_sin_fondo.png', 'h': '2_sin_fondo.png',
        # O (Boca redonda)
        'o': '4_sin_fondo.png',
        # U, W (Boca fruncida)
        'u': '1_sin_fondo.png', 'w': '1_sin_fondo.png',
        # E, S, C y consonantes semicerradas (Dientes visibles / Lengua)
        'e': '6_sin_fondo.png', 's': '6_sin_fondo.png', 'c': '6_sin_fondo.png', 'g': '6_sin_fondo.png', 
        'j': '6_sin_fondo.png', 'q': '6_sin_fondo.png', 'x': '6_sin_fondo.png', 'z': '6_sin_fondo.png', 
        'k': '6_sin_fondo.png', 'f': '6_sin_fondo.png', 'v': '6_sin_fondo.png', 'l': '6_sin_fondo.png', 
        'd': '6_sin_fondo.png', 't': '6_sin_fondo.png', 'n': '6_sin_fondo.png', 'r': '6_sin_fondo.png',
        # M, B, P (Labios juntos)
        'm': 'inhala_sin_fondo.png', 'b': 'inhala_sin_fondo.png', 'p': 'inhala_sin_fondo.png'
    }
    
    if not Path(img_silence).exists():
        return False
        
    t = 0.0
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for segment in segments:
            if not hasattr(segment, 'words') or not segment.words:
                continue
                
            for word_obj in segment.words:
                # Si hay espacio entre la última palabra y esta, es silencio (Idle animation)
                if word_obj.start > t:
                    dur = word_obj.start - t
                    dur_remaining = dur
                    idle_cycle = [img_silence, img_inhala, img_exala]
                    cycle_idx = 0
                    
                    while dur_remaining > 0:
                        chunk = min(0.6, dur_remaining) # Cada estado de respiración dura 0.6s
                        
                        # Si falta la imagen de respiración, usar silencio como fallback
                        img_to_use = idle_cycle[cycle_idx]
                        if not Path(img_to_use).exists():
                            img_to_use = img_silence
                            
                        f.write(f"file '{img_to_use}'\n")
                        f.write(f"duration {chunk:.3f}\n")
                        dur_remaining -= chunk
                        cycle_idx = (cycle_idx + 1) % 3
                        
                    t = word_obj.start
                    
                clean_word = "".join(c.lower() for c in word_obj.word if c.isalpha())
                if clean_word:
                    # Dividimos la duración de la palabra entre sus letras
                    char_dur = (word_obj.end - word_obj.start) / len(clean_word)
                    
                    for char in clean_word:
                        img_name = VISEME_MAP.get(char, '1_sin_fondo.png')
                        img_path = (avatar_dir / img_name).as_posix()
                        f.write(f"file '{img_path}'\n")
                        f.write(f"duration {char_dur:.3f}\n")
                        
                t = word_obj.end
        
        f.write(f"file '{img_silence}'\n")
    return True

print("\nCargando modelo de IA para subtítulos (Word-Level Lip-Sync activo)...")
model = WhisperModel("small", device="cpu", compute_type="int8")

def process_single_video(video_path, output_dir, base_dir, logo1, logo2, logo3, use_avatar, use_branding, use_left):
    audio_path = output_dir / f"{video_path.stem}.wav"
    srt_path = output_dir / f"{video_path.stem}.srt"
    avatar_txt_path = output_dir / f"{video_path.stem}_avatar.txt"
    final_video_path = output_dir / f"{video_path.stem}_final.mp4"
    
    print(f"[{video_path.name}] Iniciando procesamiento...")
    
    # 1. Extraer audio
    if not audio_path.exists() and (not srt_path.exists() or srt_path.stat().st_size == 0 or use_avatar):
        subprocess.run([
            "ffmpeg", "-y", "-i", str(video_path), 
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", 
            str(audio_path)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
    # 2. Generar Transcripción con timestamps por palabra
    segments_gen, info = model.transcribe(str(audio_path), language="es", beam_size=5, word_timestamps=True)
    segments = list(segments_gen)
    
    if not srt_path.exists() or srt_path.stat().st_size == 0:
        generate_srt(segments, str(srt_path))
        
    # 3. Generar secuencia de Avatar basada en IA de Texto
    has_avatar = False
    if use_avatar:
        has_avatar = generate_text_based_avatar(segments, avatar_txt_path, base_dir)
    
    # Limpiar audio
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
            f"[{avatar_idx}:v]scale=400:-1[av];"
            f"{curr_stream}[av]overlay={av_x}:H-h[curr_av];"
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
        str(final_video_path)
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
            futures.append(executor.submit(process_single_video, video_path, output_dir, base_dir, logo1, logo2, logo3, args.avatar, args.branding, args.left))
            
        for future in concurrent.futures.as_completed(futures):
            try:
                name = future.result()
                print(f"¡Completado con éxito! -> {name}")
            except Exception as e:
                print(f"Error procesando un video: {e}")

    print("\n--- TODOS LOS VIDEOS HAN SIDO PROCESADOS ---")

if __name__ == "__main__":
    main()
