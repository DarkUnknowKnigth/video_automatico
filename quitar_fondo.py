from PIL import Image
import os
from pathlib import Path

def remove_white_background(img_path, out_path, tolerance=220):
    img = Image.open(img_path).convert("RGBA")
    data = img.getdata()
    
    new_data = []
    for item in data:
        # Check if the pixel is near white
        if item[0] > tolerance and item[1] > tolerance and item[2] > tolerance:
            # Change to transparent
            new_data.append((255, 255, 255, 0))
        else:
            new_data.append(item)
            
    img.putdata(new_data)
    img.save(out_path, "PNG")
    print(f"Procesado: {out_path.name}")

if __name__ == '__main__':
    avatar_dir = Path(__file__).parent.absolute() / "avatar"
    
    for i in range(1, 8):
        img_path = avatar_dir / f"{i}.png"
        out_path = avatar_dir / f"{i}_sin_fondo.png"
        if img_path.exists():
            remove_white_background(img_path, out_path, tolerance=240)
