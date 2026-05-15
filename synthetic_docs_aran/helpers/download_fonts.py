from pathlib import Path
import requests

TYPEWRITER_DIR = Path("typewriter_fonts")
HANDWRITING_DIR = Path("fonts")

TYPEWRITER_DIR.mkdir(exist_ok=True)
HANDWRITING_DIR.mkdir(exist_ok=True)

fonts = {
    # Typewriter / mecanografiat
    "typewriter_fonts/SpecialElite-Regular.ttf":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/specialelite/SpecialElite-Regular.ttf",

    "typewriter_fonts/CourierPrime-Regular.ttf":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/courierprime/CourierPrime-Regular.ttf",

    "typewriter_fonts/CourierPrime-Italic.ttf":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/courierprime/CourierPrime-Italic.ttf",

    # Manuscrit / anotacions
    "fonts/Caveat-Regular.ttf":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/caveat/Caveat%5Bwght%5D.ttf",

    "fonts/PatrickHand-Regular.ttf":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/patrickhand/PatrickHand-Regular.ttf",

    "fonts/ReenieBeanie-Regular.ttf":
        "https://raw.githubusercontent.com/google/fonts/main/apache/reeniebeanie/ReenieBeanie-Regular.ttf",

    "fonts/Schoolbell-Regular.ttf":
        "https://raw.githubusercontent.com/google/fonts/main/apache/schoolbell/Schoolbell-Regular.ttf",

    "fonts/HomemadeApple-Regular.ttf":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/homemadeapple/HomemadeApple-Regular.ttf",

    "fonts/ShadowsIntoLight-Regular.ttf":
        "https://raw.githubusercontent.com/google/fonts/main/ofl/shadowsintolight/ShadowsIntoLight-Regular.ttf",
}

for path, url in fonts.items():
    path = Path(path)

    if path.exists():
        print(f"Ja existeix: {path}")
        continue

    print(f"Descarregant {path.name}...")
    r = requests.get(url, timeout=30)

    if r.status_code != 200:
        print(f"ERROR {r.status_code}: {url}")
        continue

    path.write_bytes(r.content)
    print(f"OK: {path}")

print("\nFet.")