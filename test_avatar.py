import wave
import struct
import math
import random
from pathlib import Path

def generate_avatar_sequence(audio_wav_path, output_txt_path, base_dir, fps=12):
    with wave.open(str(audio_wav_path), 'rb') as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        audio_data = wf.readframes(n_frames)
        
    if sample_width != 2:
        raise ValueError("Audio must be 16-bit PCM")
        
    fmt = f"<{n_frames * channels}h"
    samples = struct.unpack(fmt, audio_data)
    
    if channels == 2:
        samples = [ (samples[i] + samples[i+1])/2 for i in range(0, len(samples), 2) ]
        
    samples_per_frame = framerate // fps
    total_video_frames = len(samples) // samples_per_frame
    
    img_closed = (base_dir / "avatar" / "boca_cerrada_sinf.png").as_posix()
    img_open = (base_dir / "avatar" / "boca_abierta-sinfondo.png").as_posix()
    vowels = [
        (base_dir / "avatar" / "a_.png").as_posix(),
        (base_dir / "avatar" / "e_.png").as_posix(),
        (base_dir / "avatar" / "i_.png").as_posix(),
        (base_dir / "avatar" / "o_.png").as_posix(),
        (base_dir / "avatar" / "u_.png").as_posix()
    ]
    
    rms_values = []
    for i in range(total_video_frames):
        chunk = samples[i * samples_per_frame : (i+1) * samples_per_frame]
        if not chunk: break
        rms = math.sqrt(sum(s**2 for s in chunk) / len(chunk))
        rms_values.append(rms)
        
    max_rms = max(rms_values) if rms_values else 1
    threshold = max_rms * 0.1
    if threshold < 150: threshold = 150
    
    frame_duration = 1.0 / fps
    
    with open(output_txt_path, 'w', encoding='utf-8') as f:
        for rms in rms_values:
            if rms < threshold:
                image = img_closed
            else:
                if rms > max_rms * 0.5:
                    image = random.choice([img_open, vowels[0], vowels[3]]) # abierta, a, o
                else:
                    image = random.choice([vowels[1], vowels[2], vowels[4]]) # e, i, u
                    
            f.write(f"file '{image}'\n")
            f.write(f"duration {frame_duration:.3f}\n")
            
        f.write(f"file '{img_closed}'\n")

if __name__ == '__main__':
    base_dir = Path(r"C:\Users\jose-\.gemini\antigravity\scratch\video_processor")
    generate_avatar_sequence(base_dir / "output" / "p5.wav", base_dir / "output" / "p5_avatar.txt", base_dir)
    print("Done")
