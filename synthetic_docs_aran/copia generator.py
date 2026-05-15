import os
import random
import uuid
import glob
import zipfile
import io
import math
import time
import asyncio
from typing import Tuple, List, Dict, Any
import json
from google import genai
from dotenv import load_dotenv
from google.genai import types
from tqdm.asyncio import tqdm

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps

OPENAI_AVAILABLE = False

class GeminiTextGenerator:
    def __init__(self):
        load_dotenv()

        api_key = os.environ.get("GEMINI_API_KEY")
        
        if not api_key:
            print("Yooow, soc l'Aran, si veus aixo es que no has posat la API key de Gemini, o no has canviat el hardcodeo de text")
            self.client = None
        else:
            try:
                self.client = genai.Client()
            except Exception as e:
                print(f"⚠️ Error initializing client: {e}")
                self.client = None
                
        self.model_id = "gemini-2.5-flash"

        self.epocas = [
            {"anyo": "1925", "contexto": "Inicios de la radio. Tono pionero, cultural, mucha música clásica, ópera y boletines de bolsa. Lenguaje muy formal."},
            {"anyo": "1932", "contexto": "Segunda República. Tono más abierto, catalanismo cultural (sardanas, cursos de gramática catalana), jazz, conferencias."},
            {"anyo": "1941", "contexto": "Posguerra temprana. Tono estricto, propagandístico, mucha censura, misas, rosarios, marchas militares y exaltación patriótica."},
            {"anyo": "1953", "contexto": "Aperturismo y radio-espectáculo. Radionovelas, consultorios femeninos, publicidad de marcas, concursos y música ligera."}
        ]

        self.tipos_documento = [
            "Guía-índice de programación (tabla con horas y secciones)",
            "Parrilla de programación (tabla detallada con columnas de horas, intérprete y programa)",
            "Guion literal de locutorio (con pies para control de sonido y efectos)",
            "Guion de radioteatro (personajes y acotaciones escénicas)",
            "Informe interno de revisión de contenidos (acta de aprobaciones y rechazos)",
            "Carta oficial / Oficio (con membrete formal, sello y firma del censor)",
            "Inventario de discos (lista tabulada de canciones y autores)"
        ]

        self.niveles_censura = [
            "Baja: Solo correcciones técnicas, tiempos o cambios de canciones.",
            "Media: Alguna palabra tachada por inapropiada o discos que no llegaron.",
            "Alta: Intervención dura del censor, párrafos enteros tachados, sellos de NULO o REVISADO."
        ]

    # CHANGED TO ASYNC
    async def generate_random_script_async(self) -> dict:
        epoca = random.choice(self.epocas)
        tipo_doc = random.choice(self.tipos_documento)
        censura = random.choice(self.niveles_censura)

        target_annotations = random.randint(4, 40)

        prompt = f"""
        Eres un experto archivero histórico de Radio Barcelona (E.A.J.-1).
        Genera el contenido para un documento radiofónico histórico con las siguientes características:

        - AÑO/ÉPOCA: {epoca['anyo']} ({epoca['contexto']})
        - TIPO DE DOCUMENTO: {tipo_doc}
        - NIVEL DE CENSURA/EDICIÓN manuscrita: {censura}

        El documento base fue mecanografiado, pero tiene anotaciones a mano (correcciones, sellos, tachones).

        Devuelve ÚNICAMENTE un JSON válido con esta estructura exacta (sin formato markdown):
        {{
          "documento_base": "Texto mecanografiado completo. Al menos 10 líneas. Usa saltos de línea \\n para el formato y múltiples espacios consecutivos para alinear tabulaciones en tablas.",
          "anotaciones": [
            {{ "tipo": "insercion", "texto": "Texto breve al margen escrito a mano", "texto_a_tachar": null }},
            {{ "tipo": "correccion", "texto": "Reemplazo", "texto_a_tachar": "FRASE_EXACTA_A_REEMPLAZAR" }},
            {{ "tipo": "tachon", "texto": "", "texto_a_tachar": "FRASE_EXACTA_A_CENSURAR" }},
            {{ "tipo": "censura_bloque", "texto": "", "texto_a_tachar": "PALABRA_A_CENSURAR" }},
            {{ "tipo": "subrayado", "texto": "", "texto_a_tachar": "FRASE_A_SUBRAYAR" }},
            {{ "tipo": "circulo", "texto": "", "texto_a_tachar": "PALABRA_A_CIRCULAR" }},
            {{ "tipo": "sello", "texto": "EJ: ORIGINAL, REVISADO, APROBADO, CENSURADO", "texto_a_tachar": null }},
            {{ "tipo": "nota_larga", "texto": "Texto largo, de entre 2 líneas y 2 párrafos completos. Son comentarios extensos del censor, advertencias legales o notas de producción.", "texto_a_tachar": null }}
          ]
        }}
        INSTRUCCIÓN IMPORTANTE: Debes generar ENTRE {target_annotations} y {target_annotations + 4} anotaciones de diferentes tipos de la lista.
        """

        if not self.client:
            return {
                "documento_base": f"PROGRAMA - {epoca['anyo']}\n10:00 Sintonía\n10:15 Fallo técnico en la transmisión.",
                "anotaciones": []
            }

        try:
            # Using client.aio for asynchronous SDK requests
            response = await self.client.aio.models.generate_content(
                model=self.model_id,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.7
                )
            )
            return json.loads(response.text)

        except Exception as e:
            # Fallback in case of rate limit hit or parse error
            return {
                "documento_base": f"PROGRAMA - {epoca['anyo']}\n10:00 Sintonía\n10:15 Fallo técnico en la transmisión.",
                "anotaciones": []
            }

# class SemanticLogicEngine:
#     def __init__(self, use_llm: bool = False, llm_base_url: str = ""):
#         self.use_llm = False

#     def censor_script(self, text: str) -> Dict[str, str]:
#         # Split by both periods and newlines to handle tables and scripts better
#         segments = text.replace('\n', ' . ').split('.')
#         lines = [line.strip() for line in segments if len(line.strip()) > 15 and "---" not in line]

#         if not lines:
#             return {"sensitive_sentence": "TEXTO", "substitute_text": "[CENSURADO]"}

#         replacements = [
#             "Atenta contra la moral.",
#             "Revisado.",
#             "¡Suprimir inmediatamente!",
#             "Inadecuado.",
#             "Peligroso. Borrar.",
#             "NULO."
#         ]
#         return {
#             "sensitive_sentence": random.choice(lines),
#             "substitute_text": random.choice(replacements)
#         }

class TextGenerator:
    def __init__(self):
        self.paragraphs = [
            "Las gloriosas fuerzas del orden han desfilado hoy por la avenida principal, demostrando una vez más la inquebrantable voluntad de nuestro pueblo.",
            "Las multitudes aclamaban con fervor patriótico mientras los estandartes ondeaban al viento cálido de la tarde.",
            "En el ámbito internacional, las potencias extranjeras continúan observando nuestro desarrollo con envidia.",
            "Nuestra autarquía económica nos blinda contra las perniciosas influencias del exterior.",
            "El pan en nuestra mesa es fruto del sudor nacional y de la tierra que trabajamos con orgullo.",
            "La inauguración de la nueva red de ferrocarriles supone un hito sin precedentes en la historia civil.",
            "Queremos recordar a las madres de la patria la importancia de la educación en el hogar y la moral."
        ]

    def generate_random_script(self) -> str:
        layout = random.choice(['standard', 'script', 'sparse', 'table'])

        date = f"Fecha: {random.randint(1, 28)}/{random.randint(1,12)}/194{random.randint(0,9)}\n"

        if layout == 'standard':
            header = "MINISTERIO DE INFORMACIÓN\nDIRECCIÓN GENERAL DE RADIODIFUSIÓN\n\n"
            body = "\n\n".join(random.sample(self.paragraphs, random.randint(3, 5)))
            return header + date + "\n" + body

        elif layout == 'script':
            header = "GUION DE EMISIÓN - ESTUDIO 1\n" + date + "\n"
            lines = []
            time_m = 14
            for para in random.sample(self.paragraphs, random.randint(3, 4)):
                lines.append(f"[{time_m:02d}:00] LOCUTOR: {para}")
                time_m += random.randint(1, 3)
                lines.append(f"[{time_m:02d}:30] EFECTOS: (Música de fondo / Transición)")
                time_m += random.randint(1, 2)
            return header + "\n\n".join(lines)

        elif layout == 'sparse':
            header = "INFORME CONFIDENCIAL\n" + date + "\n"
            # Huge, irregular gaps between text blocks
            gaps = ["\n\n\n\n", "\n\n\n\n\n\n", "\n\n\n\n\n\n\n\n"]
            body = ""
            for para in random.sample(self.paragraphs, random.randint(2, 4)):
                body += para + random.choice(gaps)
            return header + body

        elif layout == 'table':
            header = "REGISTRO DE CONTENIDOS APROBADOS\n" + date + "\n"
            lines = ["SECCIÓN       DURACIÓN       CONTENIDO", "--------------------------------------------------"]
            for para in random.sample(self.paragraphs, random.randint(3, 4)):
                short_para = para[:40] + "..." if len(para) > 40 else para
                lines.append(f"BLOQUE {random.randint(1,9)}    0{random.randint(2,9)} MINUTOS    {short_para}")
                # Sometimes add the full paragraph below as notes
                if random.random() > 0.5:
                    lines.append(f"  NOTAS: {para}\n")
            return header + "\n".join(lines)

class LayerRenderer:
    def __init__(self, width=1654, height=2339):
        self.width = width
        self.height = height

        # Load Handwriting Fonts
        self.handwritten_fonts = []
        for ext in ('*.ttf', '*.otf'):
            for f in glob.glob(f"fonts/{ext}"):
                self.handwritten_fonts.append({"type": "raw", "path": f})
        for zip_path in glob.glob("fonts/*.zip"):
            try:
                with zipfile.ZipFile(zip_path, 'r') as z:
                    for f_info in z.infolist():
                        if not f_info.filename.startswith('__MACOSX') and f_info.filename.lower().endswith(('.ttf', '.otf')):
                            self.handwritten_fonts.append({"type": "zip", "zip_path": zip_path, "internal_path": f_info.filename})
            except zipfile.BadZipFile:
                pass

        # Load Typewriter Fonts (Need 4 different ones in the folder!)
        self.tw_fonts_paths = glob.glob("typewriter_fonts/*.ttf") + glob.glob("typewriter_fonts/*.otf")
        if not self.tw_fonts_paths:
            print("⚠️ WARNING: No fonts found in 'typewriter_fonts/'. Using default system font.")

        self.iam_images = glob.glob("iam_samples/*.png") + glob.glob("iam_samples/*.tif")

    def get_random_cursive_font(self, min_size=60, max_size=120):
        if self.handwritten_fonts:
            font_choice = random.choice(self.handwritten_fonts)
            size = random.randint(min_size, max_size)
            try:
                if font_choice["type"] == "raw":
                    return ImageFont.truetype(font_choice["path"], size)
                elif font_choice["type"] == "zip":
                    with zipfile.ZipFile(font_choice["zip_path"], 'r') as z:
                        font_bytes = z.read(font_choice["internal_path"])
                        return ImageFont.truetype(io.BytesIO(font_bytes), size)
            except Exception:
                pass
        return ImageFont.load_default()

    def get_document_typewriter_font(self):
        # Pick one specific machine font and size for the entire document
        size = random.randint(34, 42)
        if self.tw_fonts_paths:
            try:
                return ImageFont.truetype(random.choice(self.tw_fonts_paths), size)
            except Exception:
                pass
        return ImageFont.load_default()

    def extract_iam_ink(self, img_path: str, target_color: Tuple[int, int, int, int], target_height: int) -> Image.Image:
        try:
            raw_img = Image.open(img_path).convert('L')
            raw_img = ImageOps.autocontrast(raw_img)
            arr = np.array(raw_img)
            alpha = 255 - arr
            alpha[alpha < 80] = 0
            color_img = Image.new('RGBA', raw_img.size, target_color)
            color_img.putalpha(Image.fromarray(alpha))
            scale = target_height / float(color_img.height)
            new_width = int(color_img.width * scale)
            return color_img.resize((new_width, target_height), Image.Resampling.LANCZOS)
        except Exception:
            return None

    def render_layer0(self, text: str) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        img = Image.new('RGBA', (self.width, self.height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)

        tw_font = self.get_document_typewriter_font()

        margin_x, margin_y = random.randint(120, 200), random.randint(150, 250)
        current_y = margin_y
        line_height = random.randint(55, 75)
        word_boxes = []
        base_ink = (random.randint(10, 60), random.randint(10, 60), random.randint(10, 80))

        for paragraph in text.split('\n'):
            if not paragraph.strip():
                current_y += line_height
                continue

            current_x = margin_x

            # Preserve spaces so tables and multiple tabs don't collapse into a single space
            words = []
            curr_word = ""
            for char in paragraph:
                if char == " ":
                    if curr_word: words.append(curr_word)
                    words.append(" ")
                    curr_word = ""
                else:
                    curr_word += char
            if curr_word:
                words.append(curr_word)

            for word in words:
                if not word: continue
                # Handling pure spaces for alignment
                if word == " ":
                    try:
                        space_w = int(tw_font.getlength(" "))
                    except AttributeError:
                        space_w = int(tw_font.getbbox(" ")[2]) if tw_font.getbbox(" ") else 20
                    current_x += space_w
                    continue

                # ESTIMATE WIDTH - FORCED TO INTEGER
                try:
                    word_w = int(tw_font.getlength(word))
                except AttributeError:
                    word_w = int(tw_font.getbbox(word)[2]) if tw_font.getbbox(word) else len(word)*20

                if current_x + word_w > self.width - margin_x:
                    current_x = margin_x
                    current_y += line_height

                char_x = current_x
                for char in word:
                    y_jitter = random.choice([0, 0, 0, 0, 0, -1, 1, -2, 2, -3])
                    opacity = random.randint(120, 255)
                    ink_val = base_ink + (opacity,)

                    draw.text((char_x, current_y + y_jitter), char, font=tw_font, fill=ink_val)

                    try:
                        char_w = int(tw_font.getlength(char))
                    except AttributeError:
                        char_w = int(tw_font.getbbox(char)[2]) if tw_font.getbbox(char) else 20
                    char_x += char_w

                word_boxes.append({"word": word, "box": [current_x, current_y, current_x + word_w, current_y + line_height]})

                try:
                    space_w = int(tw_font.getlength(" "))
                except AttributeError:
                    space_w = int(tw_font.getbbox(" ")[2]) if tw_font.getbbox(" ") else 20

                current_x += word_w + space_w
            current_y += line_height

        img = img.rotate(random.uniform(-1.5, 1.5), resample=Image.BICUBIC, expand=0)
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.7)))
        return img, word_boxes

    def render_layer1(self, word_boxes: List[Dict[str, Any]], anotaciones: List[Dict[str, Any]]) -> Image.Image:
        import textwrap
        img = Image.new('RGBA', (self.width, self.height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)
        ink_colors = [
            (10, 30, 150, 240), (20, 20, 20, 250), (180, 20, 20, 230),
            (20, 100, 30, 220), (90, 40, 140, 200), (80, 90, 100, 210)
        ]

        for ano in anotaciones:
            if not isinstance(ano, dict):
                continue

            tipo = ano.get("tipo", "insercion")
            texto = ano.get("texto", "")
            texto_tachar = ano.get("texto_a_tachar", "")
            main_ink = random.choice(ink_colors)

            # Sanitización para las fuentes TTF
            if len(texto) > 120:
                texto = texto[:117] + "..."
            texto_seguro = "\n".join(textwrap.wrap(texto, width=30))

            # -----------------------------------------------------------------
            # 1. CORRECCIONES, TACHONES Y CENSURAS
            # -----------------------------------------------------------------
            if tipo in ["correccion", "tachon", "censura_bloque", "subrayado", "circulo"] and texto_tachar:
                words_to_find = texto_tachar.split()
                target_boxes = []

                if words_to_find and word_boxes:
                    for i, wb in enumerate(word_boxes):
                        if words_to_find[0] in wb["word"]:
                            match = True
                            temp_boxes = [wb]
                            for j in range(1, len(words_to_find)):
                                if i + j < len(word_boxes) and words_to_find[j] in word_boxes[i+j]["word"]:
                                    temp_boxes.append(word_boxes[i+j])
                                else:
                                    match = False
                                    break
                            if match:
                                target_boxes = temp_boxes
                                break

                if target_boxes:
                    x0 = min(int(b["box"][0]) for b in target_boxes)
                    y0 = min(int(b["box"][1]) for b in target_boxes)
                    x1 = max(int(b["box"][2]) for b in target_boxes)
                    y1 = max(int(b["box"][3]) for b in target_boxes)

                    mid_y = y0 + (y1 - y0) // 2 + random.randint(-2, 2)

                    if tipo in ["correccion", "tachon"]:
                        # Multi-line scribble or wavy line instead of single straight line
                        num_lines = random.randint(2, 4)
                        for _ in range(num_lines):
                            y_offset = random.randint(-4, 4)
                            draw.line([(x0, mid_y + y_offset), (x1, mid_y + y_offset)], fill=main_ink, width=random.randint(2, 4))

                    elif tipo == "censura_bloque":
                        # Hacer el bloque más estrecho (estilo rotulador negro) en lugar de un bloque gigante
                        altura_texto = y1 - y0
                        margen_vertical = altura_texto // 4  # Deja un poco de aire por arriba y por abajo

                        # También añadimos una pequeña variación aleatoria para que no sea un rectángulo perfecto
                        draw.rectangle([
                            x0 - random.randint(1, 4),
                            y0 + margen_vertical,
                            x1 + random.randint(1, 4),
                            y1 - margen_vertical + random.randint(0, 2)
                        ], fill=(20, 20, 20, 245))

                    elif tipo == "subrayado":
                        # Hand-drawn wavy/crooked underline
                        curr_x = x0
                        under_y = y1 + random.randint(2, 6)
                        while curr_x < x1:
                            next_x = min(curr_x + random.randint(10, 30), x1)
                            draw.line([(curr_x, under_y), (next_x, under_y + random.randint(-2, 2))], fill=main_ink, width=random.randint(2, 4))
                            curr_x = next_x
                            under_y += random.randint(-2, 2)

                    elif tipo == "circulo":
                        # Hand-drawn ellipse
                        draw.ellipse([x0 - random.randint(5,15), y0 - random.randint(5,10), x1 + random.randint(5,15), y1 + random.randint(5,10)], outline=main_ink, width=random.randint(2, 4))

                    # If it's a correction, draw the replacement text above
                    if tipo == "correccion":
                        sub_x = x0
                        sub_y = y0 - random.randint(40, 70)
                        font = self.get_random_cursive_font(60, 90)
                        try:
                            draw.text((sub_x, sub_y), texto_seguro, font=font, fill=main_ink)
                        except OSError:
                            draw.text((sub_x, sub_y), texto_seguro, font=ImageFont.load_default(), fill=main_ink)
                else:
                    tipo = "insercion" # Si no encuentra qué tachar, lo vuelve anotación al margen

            # -----------------------------------------------------------------
            # 2. INSERCIONES (Usa dataset IAM para alto realismo si es posible)
            # -----------------------------------------------------------------
            if tipo == "insercion":
                use_iam = self.iam_images and (random.random() > 0.1) # 70% de probabilidad de usar IAM real
                if use_iam:
                    iam_img_path = random.choice(self.iam_images)
                    decal = self.extract_iam_ink(iam_img_path, main_ink, target_height=random.randint(60, 120))
                    if decal:
                        x = random.randint(50, max(51, self.width - decal.width - 50))
                        y = random.randint(100, max(101, self.height - decal.height - 100))
                        decal = decal.rotate(random.uniform(-15, 15), expand=True, resample=Image.BICUBIC)
                        img.paste(decal, (x, y), decal)
                else:
                    x = random.randint(100, self.width - 400)
                    y = random.randint(100, self.height - 200)
                    font = self.get_random_cursive_font(60, 100)
                    try:
                        draw.text((x, y), texto_seguro, font=font, fill=main_ink)
                    except OSError:
                        draw.text((x, y), texto_seguro, font=ImageFont.load_default(), fill=main_ink)

            # -----------------------------------------------------------------
            # 2.5 NOTAS LARGAS (Buscando espacios en blanco)
            # -----------------------------------------------------------------
            elif tipo == "nota_larga":
                # Usamos una fuente un poco más pequeña para que quepan textos largos
                font = self.get_random_cursive_font(40, 70)

                best_box = None
                min_overlap = float('inf')

                # Intentamos 20 posiciones aleatorias y nos quedamos con la que pise menos texto original
                for _ in range(20):
                    # Definimos un ancho aleatorio para el bloque de texto
                    box_w = random.randint(400, 800)

                    # Aproximamos cuántos caracteres caben en esa línea
                    # El ancho promedio de un carácter suele ser la mitad del tamaño de la fuente
                    approx_char_w = max(10, font.size * 0.5)
                    chars_per_line = max(15, int(box_w / approx_char_w))

                    wrapped_text = "\n".join(textwrap.wrap(texto, width=chars_per_line))

                    # Calculamos el alto aproximado del bloque
                    lines = wrapped_text.count('\n') + 1
                    line_height = int(font.size * 1.2)
                    box_h = lines * line_height

                    # Escogemos coordenadas (x, y) aleatorias que no se salgan del papel
                    x = random.randint(50, max(51, self.width - box_w - 50))
                    y = random.randint(50, max(51, self.height - box_h - 50))

                    # Calculamos el área de solapamiento con todas las palabras mecanografiadas
                    overlap_area = 0
                    for wb in word_boxes:
                        wx0, wy0, wx1, wy1 = wb["box"]
                        # Intersección de rectángulos
                        ix0 = max(x, wx0)
                        iy0 = max(y, wy0)
                        ix1 = min(x + box_w, wx1)
                        iy1 = min(y + box_h, wy1)

                        if ix0 < ix1 and iy0 < iy1: # Hay solapamiento
                            overlap_area += (ix1 - ix0) * (iy1 - iy0)

                    # Guardamos la mejor posición
                    if overlap_area < min_overlap:
                        min_overlap = overlap_area
                        best_box = (x, y, wrapped_text, box_w, box_h, line_height)
                        if min_overlap == 0:
                            break # Encontramos un hueco perfecto, dejamos de buscar

                # Dibujamos el texto en la mejor posición encontrada
                if best_box:
                    x, y, wrapped_text, box_w, box_h, line_height = best_box

                    # Lo dibujamos en una imagen temporal para poder rotarlo ligeramente
                    txt_img = Image.new('RGBA', (box_w + 100, box_h + 100), (255, 255, 255, 0))
                    txt_draw = ImageDraw.Draw(txt_img)

                    current_y = 50
                    for line in wrapped_text.split('\n'):
                        try:
                            txt_draw.text((50, current_y), line, font=font, fill=main_ink)
                        except OSError:
                            txt_draw.text((50, current_y), line, font=ImageFont.load_default(), fill=main_ink)
                        current_y += line_height

                    # Rotación muy leve (es un párrafo, si se rota mucho se ve irreal)
                    txt_img = txt_img.rotate(random.uniform(-3, 3), expand=True, resample=Image.BICUBIC)

                    # Lo pegamos en la imagen final ajustando el offset de la imagen temporal
                    img.paste(txt_img, (x - 50, y - 50), txt_img)

            # -----------------------------------------------------------------
            # 3. SELLOS / FIRMAS GIGANTES
            # -----------------------------------------------------------------
            elif tipo == "sello":
                use_iam_for_stamp = self.iam_images and (random.random() > 0.3)

                if use_iam_for_stamp and random.random() > 0.5: # 50% of time use IAM for signature "stamps"
                    iam_img_path = random.choice(self.iam_images)
                    decal = self.extract_iam_ink(iam_img_path, main_ink, target_height=random.randint(150, 400))
                    if decal:
                        x = random.randint(50, max(51, self.width - decal.width - 50))
                        y = random.randint(100, max(101, self.height - decal.height - 100))
                        decal = decal.rotate(random.uniform(-60, 60), expand=True, resample=Image.BICUBIC)
                        img.paste(decal, (x, y), decal)
                else:
                    txt_img = Image.new('RGBA', (800, 400), (255, 255, 255, 0))
                    txt_draw = ImageDraw.Draw(txt_img)
                    texto_sello = texto[:25].upper()

                    # 50% chance of circular stamp vs rectangular
                    if random.random() > 0.5:
                        font = self.get_random_cursive_font(50, 80)
                        txt_draw.ellipse([50, 50, 350, 350], outline=main_ink, width=random.randint(4, 10))
                        txt_draw.ellipse([65, 65, 335, 335], outline=main_ink, width=random.randint(2, 5))
                        try:
                            # Try to center the text reasonably
                            text_w = font.getbbox(texto_sello)[2] if font.getbbox(texto_sello) else 150
                            txt_draw.text((200 - text_w//2, 160), texto_sello, font=font, fill=main_ink)
                        except OSError:
                            pass
                    else:
                        font = self.get_random_cursive_font(80, 140)
                        txt_draw.rectangle([20, 20, 780, 200], outline=main_ink, width=random.randint(6, 12))
                        try:
                            txt_draw.text((50, 40), texto_sello, font=font, fill=main_ink)
                        except OSError:
                            pass

                    # Add distress/noise to stamp
                    arr = np.array(txt_img)
                    noise = np.random.rand(*arr.shape[:2])
                    # Ensure alpha channel is preserved where not distressed
                    alpha_channel = arr[:, :, 3]
                    distress_mask = (noise > 0.85)
                    alpha_channel[distress_mask] = 0 # Poke holes in the stamp for realism
                    arr[:, :, 3] = alpha_channel
                    txt_img = Image.fromarray(arr)

                    txt_img = txt_img.rotate(random.uniform(-45, 45), expand=1)
                    x, y = int(random.randint(50, self.width - 600)), int(random.randint(100, self.height - 400))
                    img.paste(txt_img, (x, y), txt_img)

        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))
        return img

class AugmentationPipeline:
    def __init__(self, width: int, height: int):
        self.width, self.height = width, height

    def generate_paper_background(self):
        colors = [(240, 230, 200), (210, 200, 190), (225, 205, 175), (245, 245, 240)]
        base_color = random.choice(colors)
        bg_arr = np.ones((self.height, self.width, 3), dtype=np.float32) * base_color

        stain_colors = [(140, 110, 80), (180, 160, 130), (100, 120, 100), (200, 190, 170)]
        for _ in range(random.randint(0, 4)):
            cx, cy = random.randint(0, self.width), random.randint(0, self.height)
            base_radius = random.randint(150, 600)
            num_points = random.randint(6, 12)
            angles = sorted([random.uniform(0, 2 * math.pi) for _ in range(num_points)])
            points = []
            for angle in angles:
                r = base_radius * random.uniform(0.5, 1.5)
                points.append([int(cx + r * math.cos(angle)), int(cy + r * math.sin(angle))])
            pts = np.array(points, np.int32).reshape((-1, 1, 2))
            mask = np.zeros((self.height, self.width), dtype=np.float32)
            cv2.fillPoly(mask, [pts], 1.0)
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=random.randint(80, 200))
            mask *= random.uniform(0.1, 0.4)
            stain_c = random.choice(stain_colors)
            for c in range(3):
                bg_arr[:, :, c] = bg_arr[:, :, c] * (1 - mask) + stain_c[c] * mask

        noise = np.random.normal(0, random.randint(10, 25), (self.height, self.width, 3))
        bg_arr = bg_arr + noise
        bg_arr = cv2.GaussianBlur(bg_arr, (5, 5), 0)
        speckles = np.random.rand(self.height, self.width)
        bg_arr[speckles > 0.995] = [50, 50, 50]

        return Image.fromarray(np.clip(bg_arr, 0, 255).astype(np.uint8)).convert('RGBA')

    def apply_lighting_shadows(self, img_array: np.ndarray) -> np.ndarray:
        x = np.linspace(-1, 1, self.width)
        y = np.linspace(-1, 1, self.height)
        X, Y = np.meshgrid(x, y)
        grad_x = X * random.uniform(-0.3, 0.3)
        grad_y = Y * random.uniform(-0.3, 0.3)
        gradient = np.clip(1.0 - (grad_x + grad_y), 0.5, 1.0)
        for i in range(3):
            img_array[:, :, i] = img_array[:, :, i] * gradient
        return np.clip(img_array, 0, 255).astype(np.uint8)

    def composite(self, l0, l1):
        bg = self.generate_paper_background()
        comp = Image.alpha_composite(bg, l0)
        comp = Image.alpha_composite(comp, l1)
        comp_arr = np.array(comp.convert('RGB'))
        comp_arr = self.apply_lighting_shadows(comp_arr)
        final_img = Image.fromarray(comp_arr)
        if random.random() > 0.4:
            final_img = final_img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
        return final_img


class DatasetOrchestrator:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.finals_dir = os.path.join(output_dir, "all_final_images")

        os.makedirs(self.finals_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        self.text_gen = GeminiTextGenerator()
        self.renderer = LayerRenderer()
        self.aug_pipe = AugmentationPipeline(self.renderer.width, self.renderer.height)

    # NEW: Synchronous method dedicated purely to CPU-bound rendering
    def render_and_save_sync(self, data: dict, i: int):
        texto_base = data.get("documento_base", "Documento vacío o error.")
        anotaciones = data.get("anotaciones", [])

        # CPU BOUND WORK
        l0_img, word_boxes = self.renderer.render_layer0(texto_base)
        l1_img = self.renderer.render_layer1(word_boxes, anotaciones)
        final_img = self.aug_pipe.composite(l0_img, l1_img)

        # IO BOUND (Disk)
        sample_dir = os.path.join(self.output_dir, f"sample_{i:04d}")
        os.makedirs(sample_dir, exist_ok=True)

        with open(os.path.join(sample_dir, "text_original.txt"), "w", encoding="utf-8") as f:
            f.write(texto_base)

        with open(os.path.join(sample_dir, "text_annotations.json"), "w", encoding="utf-8") as f:
            json.dump(anotaciones, f, indent=4, ensure_ascii=False)

        final_filename = f"final_merged_{i:04d}.jpg"
        final_img.save(os.path.join(self.finals_dir, final_filename), quality=random.randint(70, 95))

        white_bg = Image.new("RGBA", l0_img.size, (255, 255, 255, 255))
        l0_white = Image.alpha_composite(white_bg, l0_img)
        l1_white = Image.alpha_composite(white_bg, l1_img)

        l0_white.save(os.path.join(sample_dir, "image_layer0_clean.png"))
        l1_white.save(os.path.join(sample_dir, "image_layer1_clean.png"))

    # NEW: Async orchestrator function for a single item
    async def process_item(self, i: int, pbar: tqdm):
        # 1. Await the IO-bound API request
        data = await self.text_gen.generate_random_script_async()

        # 2. Push the CPU-bound rendering to a separate thread so it doesn't block the async event loop
        await asyncio.to_thread(self.render_and_save_sync, data, i)

        # Update progress bar
        pbar.update(1)


async def main():
    TOTAL_SAMPLES = 5

    print(f"Generating {TOTAL_SAMPLES} highly advanced samples asynchronously...")
    os.makedirs("fonts", exist_ok=True)
    os.makedirs("iam_samples", exist_ok=True)
    os.makedirs("typewriter_fonts", exist_ok=True)

    TOTAL_SAMPLES = 1
    # Target ~850 RPM to stay safely under the 1000 RPM limit.
    # 60 seconds / 850 requests = ~0.07 seconds between starting each request.
    REQUEST_DELAY = 0.07
    MAX_CONCURRENT_TASKS = 60 # Prevent memory explosion from too many heavy image generations at once

    orch = DatasetOrchestrator("output_dataset_pro_try")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

    async def sem_task(i, pbar):
        async with semaphore:
            await orch.process_item(i, pbar)

    # Initialize async progress bar
    pbar = tqdm(total=TOTAL_SAMPLES, desc="Generating Samples")

    tasks = []
    for i in range(TOTAL_SAMPLES):
        tasks.append(asyncio.create_task(sem_task(i, pbar)))

        # This small sleep acts as our API Rate Limiter
        # It staggers the creation of tasks so we don't dump 5000 requests into the queue at second 0
        await asyncio.sleep(REQUEST_DELAY)

    # Wait for the remaining tasks to finish rendering
    await asyncio.gather(*tasks)
    pbar.close()

    print("Done! Check 'output_dataset_pro'.")

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
