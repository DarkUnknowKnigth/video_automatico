# Procesador Automático de Videos para Clases

Este script en Python automatiza la post-producción de tus videos o clases. Procesa videos por lotes utilizando todos los núcleos de tu procesador y añade subtítulos, marcas de agua corporativas y, opcionalmente, un **Avatar animado (PNG-Tuber)** que gesticula el texto gracias a la IA de Whisper.

## Características Principales
- 🚀 **Procesamiento Paralelo:** Renderiza hasta 4 videos simultáneamente (configurable).
- 📝 **Subtítulos con IA:** Usa `faster-whisper` para extraer subtítulos súper precisos y "quemarlos" en el video.
- 🎨 **Logos Efecto Cristal:** Añade hasta 3 logos en las esquinas. Aplica automáticamente escalado, sombreado y opacidad tipo cristal oscuro para que destaquen sobre cualquier fondo.
- 🦊 **IA Text-Based Lip-Sync:** Si activas el avatar, el script lee cada letra pronunciada en el milisegundo exacto y anima al avatar (PNG-Tuber) usando un mapa de 7 posturas vocales distintas.

## Requisitos Previos
1. **Python 3.8+** instalado en tu sistema.
2. **FFmpeg** instalado y agregado a las variables de entorno (PATH). Es el motor principal para renderizar videos.

## Instalación
Para instalar este proyecto en una PC nueva, abre una terminal en esta carpeta y ejecuta:
```bash
pip install -r requirements.txt
```

## Estructura de Carpetas
Antes de ejecutar el script, asegúrate de tener la siguiente estructura:
- `/input/` -> Coloca aquí todos los videos que quieras procesar (`.mp4`, `.mkv`, `.mov`).
- `/output/` -> Aquí aparecerán los videos procesados (terminación `_final.mp4`) y los archivos sueltos `.srt`.
- `/logos/` -> Debes tener exactamente **3 imágenes** (ej. `logo1.png`, `logo2.png`, `logo3.png`).
- `/avatar/` -> Contiene el set de imágenes de tu avatar para el lip-sync (boca abierta, cerrada, etc.).

## Uso
Abre tu consola en la carpeta del proyecto. Tienes dos formas de ejecutarlo:

**1. Modo Estándar (Solo subtítulos)**
Por defecto, el script genera la transcripción por IA y quema los subtítulos en el video.
```bash
python procesar_videos.py
```

**2. Añadir Logos (Branding)**
Para agregar los 3 logos con efecto de cristal en las esquinas, debes incluir la bandera `--branding`.
```bash
python procesar_videos.py --branding
```

**3. Modo Avatar Animado**
Para insertar el avatar animado en la esquina inferior derecha hablando al ritmo de tu voz:
```bash
python procesar_videos.py --avatar
```

**4. Invertir posiciones (Avatar a la izquierda)**
Si usas la bandera `--left`, el avatar se colocará en la esquina inferior izquierda, y el tercer logo pasará a la esquina inferior derecha (intercambiando sus posiciones por defecto):
```bash
python procesar_videos.py --branding --avatar --left
```

*Nota: Puedes combinar las opciones libremente.*

## Cómo funciona el Lip-Sync
El lip-sync no se basa en el volumen del audio (que suele fallar), sino **en el texto**.
La Inteligencia Artificial determina cuándo empieza y termina de pronunciarse cada palabra. El código separa las letras e invoca las imágenes de `/avatar/` (ej. si dices "Hola", manda a llamar rápidamente las formas de la boca para H-O-L-A en fracción de segundos). Cuando hay silencio o pausas largas, el avatar entra en un modo "Idle" en donde simula respirar (alterna suavemente las imágenes `inhala` y `exala`).
