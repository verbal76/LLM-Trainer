"""Derive the placeholder launcher icon from the canonical studio logo (fit-inside, no distortion).
A dedicated LLM Trainer icon has not been supplied; replace app/src/main/res/mipmap-nodpi/ic_launcher.png when it is."""
from PIL import Image
import pathlib
root = pathlib.Path(__file__).resolve().parent.parent
logo = Image.open(root / "branding/hot-attic-games/studio-logo.png").convert("RGBA")
size = 432
canvas = Image.new("RGBA", (size, size), (14, 14, 18, 255))
scale = (size * 0.9) / max(logo.size)
resized = logo.resize((round(logo.width * scale), round(logo.height * scale)), Image.LANCZOS)
canvas.alpha_composite(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
out = root / "app/src/main/res/mipmap-nodpi/ic_launcher.png"
out.parent.mkdir(parents=True, exist_ok=True)
canvas.save(out)
print("wrote", out)
