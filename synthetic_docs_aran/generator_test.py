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
            return {
                "documento_base": f"PROGRAMA - {epoca['anyo']}\n10:00 Sintonía\n10:15 Fallo técnico en la transmisión.",
                "anotaciones": []
            }

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

        # Load Typewriter Fonts
        self.tw_fonts_paths = glob.glob("typewriter_fonts/*.ttf") + glob.glob("typewriter_fonts/*.otf")
        if not self.tw_fonts_paths:
            print("⚠️ WARNING: No fonts found in 'typewriter_fonts/'. Using default system font.")

        self.iam_images = glob.glob("iam_samples/*.png") + glob.glob("iam_samples/*.tif")

        # Prefer elegant old fonts
        preferred = ["Italianno", "Tangerine", "AlexBrush", "PinyonScript", "Allura"]
        preferred_entries = []
        for entry in self.handwritten_fonts:
            if entry["type"] == "raw":
                base = os.path.basename(entry["path"])
                if any(p.lower() in base.lower() for p in preferred):
                    preferred_entries.append(entry)
        # Fallback if no preferred found
        if not preferred_entries and self.handwritten_fonts:
            preferred_entries = self.handwritten_fonts[:]

        # Pick one writer for the document
        if preferred_entries:
            choice = random.choice(preferred_entries)
            if choice["type"] == "raw":
                self.document_hand_font = choice["path"]
            else:
                self.document_hand_font = choice  # zip entry
        else:
            self.document_hand_font = None

        self.print_fonts = {
            'title': self._load_system_font("DejaVuSerif-Bold.ttf", 50),
            'subtitle': self._load_system_font("DejaVuSerif.ttf", 19),
            'header': self._load_system_font("DejaVuSans-Bold.ttf", 15),
            'label': self._load_system_font("DejaVuSans.ttf", 13),
            'small': self._load_system_font("DejaVuSans.ttf", 11),
        }

    def _load_system_font(self, font_name, size):
        paths = [
            f"/usr/share/fonts/truetype/dejavu/{font_name}",
            f"/usr/share/fonts/truetype/liberation/{font_name}",
        ]
        for p in paths:
            if os.path.exists(p):
                try:
                    return ImageFont.truetype(p, size)
                except:
                    pass
        return ImageFont.load_default()

    def get_cursive_font(self, size):
        if self.document_hand_font is None:
            return ImageFont.load_default()
        try:
            if isinstance(self.document_hand_font, str):
                return ImageFont.truetype(self.document_hand_font, size)
            elif isinstance(self.document_hand_font, dict):
                with zipfile.ZipFile(self.document_hand_font["zip_path"], 'r') as z:
                    font_bytes = z.read(self.document_hand_font["internal_path"])
                    return ImageFont.truetype(io.BytesIO(font_bytes), size)
        except Exception:
            pass
        return ImageFont.load_default()

    def get_random_cursive_font(self, min_size=60, max_size=120):
        size = random.randint(min_size, max_size)
        return self.get_cursive_font(size)

    def get_document_typewriter_font(self):
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

    def _draw_text_with_fallback(self, draw, pos, text, font, fill):
        try:
            draw.text(pos, text, font=font, fill=fill)
        except (OSError, UnicodeEncodeError):
            draw.text(pos, text, font=ImageFont.load_default(), fill=fill)

    def render_layer0(self, text: str) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        img = Image.new('RGBA', (self.width, self.height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)

        # Base paper
        base_bg = Image.new('RGBA', (self.width, self.height), (212, 198, 168, 255))
        img = Image.alpha_composite(base_bg, img)
        draw = ImageDraw.Draw(img)

        blue_ink = (38, 48, 100, 255)
        dark_ink = (18, 18, 22, 255)

        margin_left = 50
        margin_right = self.width - 50
        margin_top = 45

        # --- TITLE ---
        title_y = margin_top + 5
        title_font = self.print_fonts['title']
        title_letters = list("PROGRAMA")
        title_x = margin_left + 5
        letter_spacing = 10
        for letter in title_letters:
            self._draw_text_with_fallback(draw, (title_x, title_y), letter, title_font, dark_ink)
            try:
                lw = title_font.getlength(letter)
            except:
                lw = title_font.getbbox(letter)[2] if title_font.getbbox(letter) else 30
            title_x += int(lw) + letter_spacing

        # Subtitle
        sub_y = title_y + 62
        self._draw_text_with_fallback(draw, (margin_left + 180, sub_y), "para el día", self.print_fonts['subtitle'], dark_ink)
        self._draw_text_with_fallback(draw, (margin_left + 350, sub_y), "de", self.print_fonts['subtitle'], dark_ink)
        self._draw_text_with_fallback(draw, (margin_left + 510, sub_y), "de 1925 según ordenación que sigue:", self.print_fonts['subtitle'], dark_ink)

        # Top right "Núm."
        self._draw_text_with_fallback(draw, (margin_right - 140, margin_top + 5), "Núm.", self.print_fonts['small'], dark_ink)

        # --- TABLE ---
        col_widths = [55, 80, 65, 80, 120, 130, 320, 110, 220, 135, 110, 130]
        col_xs = [margin_left]
        for w in col_widths[:-1]:
            col_xs.append(col_xs[-1] + w)
        col_widths[-1] = margin_right - col_xs[-1]

        header_labels = [
            ["Núm. de", "orden"],
            ["Hora", "principio"],
            ["Duración"],
            ["Hora", "termina"],
            ["Autor", "música"],
            ["Autor", "letra"],
            ["A S U N T O"],
            ["Anunciado"],
            ["E J E C U T A N T E S"],
            ["Transmisión", "hecha", "desde"],
            ["Encargado", "transmisión"],
            ["OBSERVACIONES"]
        ]

        row_height = 32
        header_height = 46
        table_y_start = sub_y + 48

        draw.line([(margin_left, table_y_start), (margin_right, table_y_start)], fill=blue_ink, width=2)
        draw.line([(margin_left, table_y_start + header_height), (margin_right, table_y_start + header_height)], fill=blue_ink, width=2)

        table_total_rows = 24
        table_bottom_y = table_y_start + header_height + table_total_rows * row_height

        for cx in col_xs:
            draw.line([(cx, table_y_start), (cx, table_bottom_y)], fill=blue_ink, width=1)
        draw.line([(margin_right, table_y_start), (margin_right, table_bottom_y)], fill=blue_ink, width=1)

        for cx, w, lines in zip(col_xs, col_widths, header_labels):
            hy = table_y_start + 4
            for line in lines:
                self._draw_text_with_fallback(draw, (cx + 4, hy), line, self.print_fonts['label'], dark_ink)
                hy += 14

        for r in range(table_total_rows + 1):
            y = table_y_start + header_height + r * row_height
            line_ink = blue_ink if random.random() > 0.15 else (blue_ink[0], blue_ink[1], blue_ink[2], 180)
            draw.line([(margin_left, y), (margin_right, y)], fill=line_ink, width=1)

        for r in range(1, 16):
            y = table_y_start + header_height + (r-1) * row_height
            self._draw_text_with_fallback(draw, (col_xs[0] + 20, y + 9), str(r), self.print_fonts['label'], dark_ink)

        # --- MODIFICACIONES ---
        mod_y = table_bottom_y + 18
        mod_text = "M O D I F I C I A C I O N E S"
        self._draw_text_with_fallback(draw, (margin_left + 180, mod_y + 2), mod_text, self.print_fonts['header'], dark_ink)
        mod_header_height = 24
        draw.line([(margin_left, mod_y + mod_header_height), (margin_right, mod_y + mod_header_height)], fill=blue_ink, width=1)

        mod_rows = 6
        mod_bottom_y = mod_y + mod_header_height + mod_rows * row_height
        for cx in col_xs:
            draw.line([(cx, mod_y + mod_header_height), (cx, mod_bottom_y)], fill=blue_ink, width=1)
        draw.line([(margin_right, mod_y + mod_header_height), (margin_right, mod_bottom_y)], fill=blue_ink, width=1)

        for r in range(mod_rows + 1):
            y = mod_y + mod_header_height + r * row_height
            draw.line([(margin_left, y), (margin_right, y)], fill=blue_ink, width=1)

        # --- OBSERVACIONES ---
        obs_y = mod_bottom_y + 15
        self._draw_text_with_fallback(draw, (margin_left, obs_y), "OBSERVACIONES:", self.print_fonts['header'], dark_ink)
        obs_line_y = obs_y + 20
        draw.line([(margin_left, obs_line_y), (margin_right, obs_line_y)], fill=blue_ink, width=1)
        for r in range(1, 5):
            y = obs_line_y + r * 30
            draw.line([(margin_left, y), (margin_right, y)], fill=blue_ink, width=1)

        # --- FOOTER ---
        footer_y = self.height - 130
        self._draw_text_with_fallback(draw, (margin_left, footer_y), "Conforme:", self.print_fonts['small'], dark_ink)
        self._draw_text_with_fallback(draw, (margin_left, footer_y + 14), "Ing. - Director,", self.print_fonts['small'], dark_ink)
        self._draw_text_with_fallback(draw, (640, footer_y + 32), "Barcelona,", self.print_fonts['small'], dark_ink)
        self._draw_text_with_fallback(draw, (1130, footer_y), "Certifico que se ha ejecutado el programa anterior", self.print_fonts['small'], dark_ink)
        self._draw_text_with_fallback(draw, (1130, footer_y + 12), "con las modificaciones y observaciones que se citan.", self.print_fonts['small'], dark_ink)
        self._draw_text_with_fallback(draw, (1380, footer_y + 32), "El Anunciador,", self.print_fonts['small'], dark_ink)

        img = img.rotate(random.uniform(-0.2, 0.2), resample=Image.BICUBIC, expand=0)
        word_boxes = []
        return img, word_boxes

    def render_layer1(self, word_boxes: List[Dict[str, Any]], anotaciones: List[Dict[str, Any]]) -> Image.Image:
        img = Image.new('RGBA', (self.width, self.height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)

        main_ink = (85, 25, 115, 215)
        light_ink = (100, 35, 130, 195)

        margin_left = 50
        col_widths = [55, 80, 65, 80, 120, 130, 320, 110, 220, 135, 110, 130]
        col_xs = [margin_left]
        for w in col_widths[:-1]:
            col_xs.append(col_xs[-1] + w)

        table_y_start = 45 + 5 + 62 + 48
        header_height = 46
        row_height = 32

        def draw_hand_text(x, y, text, size, ink=main_ink):
            font = self.get_cursive_font(size)
            jx = x + random.randint(-2, 2)
            jy = y + random.randint(-1, 2)
            try:
                draw.text((jx, jy), text, font=font, fill=ink)
            except (OSError, UnicodeEncodeError):
                draw.text((jx, jy), text, font=ImageFont.load_default(), fill=ink)

        # Hardcoded entries to match target radio program form
        main_entries = [
            (1, 8, "Septimino Radio", 34),
            (1, 9, "Auditorium", 30),
            (2, 4, "J. Rueda", 26),
            (2, 6, "La Ciudad Condal", 28),
            (2, 8, "Jaime Abant Batlle", 26),
            (2, 9, "(recitado)", 20),
            (3, 4, "Bullones", 26),
            (3, 6, "el 24 de setiembre", 28),
            (4, 4, "Villaseca", 26),
            (4, 6, "el poeta recuerda", 28),
            (5, 4, "Tchaikovsky", 26),
            (5, 6, "Somos new (vals)", 28),
            (5, 8, "Septimino Radio", 32),
            (6, 4, "Liszt", 26),
            (6, 6, "Leyenda (opereta)", 28),
            (7, 4, "Brahms", 26),
            (7, 6, "Danza hungara nº 5", 28),
            (8, 4, "Bellini", 26),
            (8, 6, "La Norma (fantasia)", 28),
            (9, 4, "Donizetti", 26),
            (9, 6, "La Favorita (una página)", 26),
            (9, 8, "celestino López", 26),
            (9, 9, "(tenor)", 20),
            (10, 4, "Verdi", 26),
            (10, 6, "Rigoletto (la donna è mobile)", 26),
            (11, 4, "Gounod", 26),
            (11, 6, "Faust (salve dimora)", 28),
            (11, 8, "Septimino Radio", 32),
            (13, 4, "Mozart", 26),
            (13, 6, "Flauta mágica (aria)", 28),
            (13, 8, "Sta. María del Carmen", 26),
            (13, 9, "Naval", 22),
            (14, 4, "Brook", 26),
            (14, 6, "Variaciones", 28),
            (15, 4, "J. Strauss", 26),
            (15, 6, "Vals", 28),
            (16, 4, "Helen Bella", 26),
            (16, 6, "Violets austriaca", 28),
            (16, 8, "Duetto Aragay", 30),
            (17, 4, "Aragay", 26),
            (17, 6, "Schubert", 26),
            (18, 4, "Gretchaninov", 26),
            (18, 6, "Andante", 26),
            (19, 4, "Aragay", 26),
            (19, 6, "Himno de España (aveglos)", 26),
            (20, 4, "Aragay", 26),
            (20, 6, "Gipsy", 26),
            (20, 8, "S. Bocasens", 28),
        ]

        for row, col_idx, text, size in main_entries:
            y = table_y_start + header_height + row * row_height + 3
            x = col_xs[col_idx] + 3
            draw_hand_text(x, y, text, size)

        # Modificaciones
        mod_y = table_y_start + header_height + 24 * row_height + 18
        mod_header_height = 24
        mod_entries = [
            (1, 4, "Bocasens", 26),
            (1, 6, "Fiat (canción)", 28),
            (2, 4, "Booger", 26),
            (2, 6, "Idili", 26),
            (3, 4, "Wagner", 26),
            (3, 6, "La Rosa", 28),
            (4, 4, "Verdi", 26),
            (4, 6, "La Escarlata (fantasia)", 28),
            (4, 8, "Septimino Radio", 32),
            (5, 4, "Zorn", 26),
            (5, 6, "barytone", 26),
        ]

        for row, col_idx, text, size in mod_entries:
            y = mod_y + mod_header_height + row * row_height + 3
            x = col_xs[col_idx] + 3
            draw_hand_text(x, y, text, size)

        # Top right "A Interventor Núm. 166"
        font_tr = self.get_cursive_font(22)
        try:
            draw.text((1320, 42), "A Interventor", font=font_tr, fill=light_ink)
            draw.text((1350, 62), "Núm. 166", font=font_tr, fill=light_ink)
        except:
            pass

        # Date blanks
        font_date = self.get_cursive_font(24)
        try:
            draw.text((320, 114), "1", font=font_date, fill=light_ink)
            draw.text((455, 111), "Junio", font=font_date, fill=light_ink)
        except:
            pass

        # Footer signature and date
        footer_y = self.height - 130
        sig_ink = (65, 20, 100, 220)
        font_sig = self.get_cursive_font(62)
        font_date2 = self.get_cursive_font(28)
        try:
            draw.text((margin_left + 10, footer_y + 28), "Guillen-Garcia", font=font_sig, fill=sig_ink)
            draw.text((730, footer_y + 28), "31", font=font_date2, fill=sig_ink)
            draw.text((795, footer_y + 28), "mayo", font=font_date2, fill=sig_ink)
            draw.text((885, footer_y + 28), "1925", font=font_date2, fill=sig_ink)
        except:
            pass

        img = img.filter(ImageFilter.GaussianBlur(radius=0.2))
        return img


class AugmentationPipeline:
    def __init__(self, width: int, height: int):
        self.width, self.height = width, height

    def generate_paper_background(self):
        base_color = (210, 194, 168)
        bg_arr = np.ones((self.height, self.width, 3), dtype=np.float32) * base_color

        stain_colors = [(195, 180, 150), (185, 170, 140), (205, 190, 160), (175, 160, 130)]
        for _ in range(random.randint(6, 12)):
            cx, cy = random.randint(0, self.width), random.randint(0, self.height)
            base_radius = random.randint(500, 1500)
            num_points = random.randint(6, 12)
            angles = sorted([random.uniform(0, 2 * math.pi) for _ in range(num_points)])
            points = []
            for angle in angles:
                r = base_radius * random.uniform(0.5, 1.5)
                points.append([int(cx + r * math.cos(angle)), int(cy + r * math.sin(angle))])
            pts = np.array(points, np.int32).reshape((-1, 1, 2))
            mask = np.zeros((self.height, self.width), dtype=np.float32)
            cv2.fillPoly(mask, [pts], 1.0)
            mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=random.randint(150, 350))
            mask *= random.uniform(0.15, 0.4)
            stain_c = random.choice(stain_colors)
            for c in range(3):
                bg_arr[:, :, c] = bg_arr[:, :, c] * (1 - mask) + stain_c[c] * mask

        noise = np.random.normal(0, random.randint(5, 12), (self.height, self.width, 3))
        bg_arr = bg_arr + noise
        bg_arr = cv2.GaussianBlur(bg_arr, (3, 3), 0)

        speckles = np.random.rand(self.height, self.width)
        bg_arr[speckles > 0.997] = [45, 40, 30]

        # Horizontal fold creases
        for fy_pct in [0.25, 0.5, 0.75]:
            fy = int(self.height * fy_pct) + random.randint(-30, 30)
            for dx in range(self.width):
                width = random.randint(1, 4)
                for dy in range(-width, width+1):
                    if 0 <= fy+dy < self.height:
                        darken = random.uniform(0.78, 0.88)
                        bg_arr[fy+dy, dx] = bg_arr[fy+dy, dx] * darken

        for _ in range(random.randint(1, 3)):
            fy = random.randint(200, self.height - 200)
            for dx in range(self.width):
                w = random.randint(1, 2)
                for dy in range(-w, w+1):
                    if 0 <= fy+dy < self.height:
                        darken = random.uniform(0.85, 0.92)
                        bg_arr[fy+dy, dx] = bg_arr[fy+dy, dx] * darken

        # Edge darkening
        edge_mask = np.ones((self.height, self.width), dtype=np.float32)
        edge_w = 25
        for y in range(self.height):
            for x in range(edge_w):
                edge_mask[y, x] = 0.88 + 0.12 * x / edge_w
            for x in range(max(0, self.width - edge_w), self.width):
                edge_mask[y, x] = 0.88 + 0.12 * (self.width - x) / edge_w
        for c in range(3):
            bg_arr[:, :, c] = bg_arr[:, :, c] * edge_mask

        # Bottom edge
        for y in range(max(0, self.height - 40), self.height):
            factor = 0.9 + 0.1 * (self.height - y) / 40
            bg_arr[y, :] = bg_arr[y, :] * factor

        # Left edge tears
        for y in range(100, self.height - 100):
            if random.random() < 0.02:
                tear_w = random.randint(3, 12)
                bg_arr[y, :tear_w] = bg_arr[y, :tear_w] * 0.7
            if random.random() < 0.01:
                tear_w = random.randint(2, 8)
                bg_arr[y, :tear_w] = [160, 150, 130]

        return Image.fromarray(np.clip(bg_arr, 0, 255).astype(np.uint8)).convert('RGBA')

    def apply_lighting_shadows(self, img_array: np.ndarray) -> np.ndarray:
        x = np.linspace(-1, 1, self.width)
        y = np.linspace(-1, 1, self.height)
        X, Y = np.meshgrid(x, y)
        grad_x = X * random.uniform(-0.1, 0.1)
        grad_y = Y * random.uniform(-0.1, 0.1)
        gradient = np.clip(1.0 - (grad_x + grad_y), 0.82, 1.0)
        for i in range(3):
            img_array[:, :, i] = img_array[:, :, i] * gradient
        return np.clip(img_array, 0, 255).astype(np.uint8)

    def apply_perspective(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        margin = random.randint(10, 25)
        top_shift = random.randint(-margin, margin)
        bottom_shift = random.randint(-margin, margin)

        src_pts = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst_pts = np.float32([[top_shift, 0], [w + top_shift, 0], [w + bottom_shift, h], [bottom_shift, h]])

        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        arr = np.array(img)
        warped = cv2.warpPerspective(arr, matrix, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(212, 198, 168, 255))
        return Image.fromarray(warped)

    def add_scan_border(self, img: Image.Image) -> Image.Image:
        arr = np.array(img)
        h, w = arr.shape[:2]
        border_w = 15
        border_color = [25, 25, 45]
        for y in range(h):
            for x in range(border_w):
                factor = x / border_w
                arr[y, x, :3] = arr[y, x, :3] * factor + np.array(border_color) * (1 - factor)
            for x in range(w - border_w, w):
                factor = (w - x) / border_w
                arr[y, x, :3] = arr[y, x, :3] * factor + np.array(border_color) * (1 - factor)
        for x in range(w):
            for y in range(border_w):
                factor = y / border_w
                arr[y, x, :3] = arr[y, x, :3] * factor + np.array(border_color) * (1 - factor)
            for y in range(h - border_w, h):
                factor = (h - y) / border_w
                arr[y, x, :3] = arr[y, x, :3] * factor + np.array(border_color) * (1 - factor)
        return Image.fromarray(arr)

    def add_scan_noise(self, img: Image.Image) -> Image.Image:
        arr = np.array(img).astype(np.float32)
        noise = np.random.normal(0, 3.0, arr.shape)
        arr = arr + noise
        for _ in range(random.randint(5, 12)):
            sy = random.randint(0, arr.shape[0] - 1)
            arr[sy, :] += random.uniform(-10, 10)
            if sy + 1 < arr.shape[0]:
                arr[sy + 1, :] += random.uniform(-5, 5)
        return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))

    def composite(self, l0, l1):
        bg = self.generate_paper_background()
        comp = Image.alpha_composite(bg, l0)
        comp = Image.alpha_composite(comp, l1)
        comp = self.apply_perspective(comp)
        comp_arr = np.array(comp.convert('RGB'))
        comp_arr = self.apply_lighting_shadows(comp_arr)
        final_img = Image.fromarray(comp_arr)
        final_img = final_img.filter(ImageFilter.GaussianBlur(radius=0.35))
        final_img = self.add_scan_border(final_img)
        final_img = self.add_scan_noise(final_img)
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

    def render_and_save_sync(self, data: dict, i: int):
        texto_base = data.get("documento_base", "Documento vacío o error.")
        anotaciones = data.get("anotaciones", [])

        l0_img, word_boxes = self.renderer.render_layer0(texto_base)
        l1_img = self.renderer.render_layer1(word_boxes, anotaciones)
        final_img = self.aug_pipe.composite(l0_img, l1_img)

        sample_dir = os.path.join(self.output_dir, f"sample_{i:04d}")
        os.makedirs(sample_dir, exist_ok=True)

        with open(os.path.join(sample_dir, "text_original.txt"), "w", encoding="utf-8") as f:
            f.write(texto_base)

        with open(os.path.join(sample_dir, "text_annotations.json"), "w", encoding="utf-8") as f:
            json.dump(anotaciones, f, indent=4, ensure_ascii=False)

        final_filename = f"final_merged_{i:04d}.jpg"
        final_img.save(os.path.join(self.finals_dir, final_filename), quality=random.randint(85, 95))

        white_bg = Image.new("RGBA", l0_img.size, (255, 255, 255, 255))
        l0_white = Image.alpha_composite(white_bg, l0_img)
        l1_white = Image.alpha_composite(white_bg, l1_img)

        l0_white.save(os.path.join(sample_dir, "image_layer0_clean.png"))
        l1_white.save(os.path.join(sample_dir, "image_layer1_clean.png"))

    async def process_item(self, i: int, pbar: tqdm):
        data = await self.text_gen.generate_random_script_async()
        await asyncio.to_thread(self.render_and_save_sync, data, i)
        pbar.update(1)


async def main():
    TOTAL_SAMPLES = 1

    print(f"Generating {TOTAL_SAMPLES} sample...")
    os.makedirs("fonts", exist_ok=True)
    os.makedirs("iam_samples", exist_ok=True)
    os.makedirs("typewriter_fonts", exist_ok=True)

    REQUEST_DELAY = 0.07
    MAX_CONCURRENT_TASKS = 60

    orch = DatasetOrchestrator("output_dataset_pro_try")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)

    async def sem_task(i, pbar):
        async with semaphore:
            await orch.process_item(i, pbar)

    pbar = tqdm(total=TOTAL_SAMPLES, desc="Generating Samples")

    tasks = []
    for i in range(TOTAL_SAMPLES):
        tasks.append(asyncio.create_task(sem_task(i, pbar)))
        await asyncio.sleep(REQUEST_DELAY)

    await asyncio.gather(*tasks)
    pbar.close()

    print("Done! Check 'output_dataset_pro_try'.")

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())