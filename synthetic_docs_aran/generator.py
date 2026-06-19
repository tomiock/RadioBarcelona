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
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageOps, ImageChops


try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    AsyncOpenAI = None
    OPENAI_AVAILABLE = False


# ============================================================
# CONFIGURACIÓ GLOBAL DEL GENERADOR
# ============================================================

CONFIG = {
    # Generació
    "TOTAL_SAMPLES": int(os.environ.get("TOTAL_SAMPLES", 3)),

    "USE_GEMINI": os.environ.get("USE_GEMINI", "false").lower() == "true",
    
    "USE_OPENAI": os.environ.get("USE_OPENAI", "false").lower() == "true",
    "OPENAI_MODEL": os.environ.get("OPENAI_MODEL", "gpt-5-mini"),

    # Text base mecanografiat
    "MIN_PROGRAM_LINES": 3, # Número de líneas del programa mecanografiado. Si el prompt de Gemini/OpenAI genera un número diferente, se ignora este rango y se usan las líneas generadas.     
    "MAX_PROGRAM_LINES": 40, # 
    "TYPEWRITER_FONT_MIN_SIZE": 12, # Tamaño mínimo de la fuente mecanografiada. Si el prompt de Gemini/OpenAI especifica un tamaño, se ignora este rango y se usa el tamaño generado.
    "TYPEWRITER_FONT_MAX_SIZE": 50,
    "LINE_HEIGHT_MIN": 10, # Altura mínima entre líneas del texto mecanografiado. Si el prompt de Gemini/OpenAI especifica una altura, se ignora este rango y se usa la altura generada.
    "LINE_HEIGHT_MAX": 150,

    # Anotacions generades localment
    "MIN_ANNOTATIONS": 0,
    "MAX_ANNOTATIONS": 25,

    # Segells i censura forçats
    "MIN_EXTRA_STAMPS": 0,
    "MAX_EXTRA_STAMPS": 4,
    "MIN_EXTRA_CENSORSHIP": 0,
    "MAX_EXTRA_CENSORSHIP": 2,

    # Probabilitats d’assets reals
    "REAL_STAMP_PROB": 0.95, # Probabilitat de posar un segell real en comptes de dibuixar-lo. Si no hi ha assets, ignora aquesta probabilitat i no posa segells.
    "EXTRA_STAMP_PROB": 0.55, # Probabilitat de posar segells extra addicionals a les anotacions, a part dels que surten al text. Aquests segells extra no tenen perquè estar associats a cap frase concreta, poden estar repartits per la pàgina.

    "EXTRA_CENSORSHIP_PROB": 0.30,

    "PATCH_PROB": 0.10,

    "ERASURE_PROB": 0.40,
    "MIN_EXTRA_ERASURES": 0,
    "MAX_EXTRA_ERASURES": 30,

    "TABLE_PROB": 0.25,

    "STAIN_PROB": 0.20,

    # Concurrència
    "MAX_CONCURRENT_TASKS": 3,
    "REQUEST_DELAY": 1.5,
}


# ============================================================
# ASSET DISCOVERY HELPERS
# ============================================================

ASSET_IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.tif", "*.tiff")


def collect_assets_from_dirs(*dirs):
    """
    Carrega assets d'imatge des de diverses carpetes.

    Això permet combinar:
        assets/stamps/
        assets_real_reviewed/stamps/

    Si una carpeta no existeix o és buida, simplement s'ignora.
    """
    paths = []

    for folder in dirs:
        for pattern in ASSET_IMAGE_EXTENSIONS:
            paths.extend(glob.glob(os.path.join(folder, pattern)))

    # Eliminem duplicats i mantenim ordre estable.
    return sorted(set(paths))




def remove_white_background(img, threshold=235, alpha_strength=0.85):
    """
    Converteix el fons blanc/gris clar en transparència.
    Ideal per segells, signatures, tatxadures i manuscrit escanejat.
    """
    img = img.convert("RGBA")
    data = np.array(img)

    rgb = data[..., :3].astype(np.int16)
    brightness = rgb.mean(axis=2)

    alpha = 255 - brightness
    alpha = np.clip(alpha * alpha_strength, 0, 255).astype(np.uint8)

    white_mask = (
        (data[..., 0] > threshold) &
        (data[..., 1] > threshold) &
        (data[..., 2] > threshold)
    )

    data[..., 3] = alpha
    data[white_mask, 3] = 0

    return Image.fromarray(data)



def bbox_list_to_dict(box):
    x1, y1, x2, y2 = box
    return {
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2),
    }


def make_layout_object(
    obj_id,
    obj_type,
    bbox,
    text=None,
    layer=None,
    source="synthetic_generator",
    confidence=1.0,
    extra=None,
):
    obj = {
        "id": obj_id,
        "type": obj_type,
        "bbox": bbox if isinstance(bbox, dict) else bbox_list_to_dict(bbox),
        "text": text,
        "layer": layer,
        "source": source,
        "confidence": confidence,
    }

    if extra:
        obj.update(extra)

    return obj


def union_bbox(boxes):
    x1 = min(int(b[0]) for b in boxes)
    y1 = min(int(b[1]) for b in boxes)
    x2 = max(int(b[2]) for b in boxes)
    y2 = max(int(b[3]) for b in boxes)
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}





def build_typewritten_objects(word_boxes, sample_id, line_y_tolerance=12):
    objects = []

    # 1. Paraules individuals
    for idx, wb in enumerate(word_boxes, start=1):
        box = wb.get("box")
        word = wb.get("word", "")

        if not box or not word.strip():
            continue

        objects.append(
            make_layout_object(
                obj_id=f"typewritten_word_{sample_id:04d}_{idx:06d}",
                obj_type="typewritten_word",
                bbox=box,
                text=word,
                layer="layer0_clean",
            )
        )

    # 2. Agrupació simple en línies segons coordenada Y
    valid_words = [
        wb for wb in word_boxes
        if wb.get("box") and wb.get("word", "").strip()
    ]

    valid_words.sort(key=lambda wb: (wb["box"][1], wb["box"][0]))

    lines = []

    for wb in valid_words:
        box = wb["box"]
        y = int(box[1])

        placed = False
        for line in lines:
            if abs(line["y"] - y) <= line_y_tolerance:
                line["words"].append(wb)
                line["y"] = int((line["y"] + y) / 2)
                placed = True
                break

        if not placed:
            lines.append({"y": y, "words": [wb]})

    for line_idx, line in enumerate(lines, start=1):
        words = sorted(line["words"], key=lambda wb: wb["box"][0])
        boxes = [wb["box"] for wb in words]
        text = " ".join(wb["word"] for wb in words)

        objects.append(
            make_layout_object(
                obj_id=f"typewritten_text_{sample_id:04d}_{line_idx:04d}",
                obj_type="typewritten_text",
                bbox=union_bbox(boxes),
                text=text,
                layer="layer0_clean",
            )
        )

    return objects

class GeminiTextGenerator:
    def __init__(self):
        load_dotenv()

        use_gemini = os.environ.get("USE_GEMINI", "false").lower() == "true"
        api_key = os.environ.get("GEMINI_API_KEY") if use_gemini else None
        
        if not api_key:
            print("Yooow, soc l'Aran, si veus aixo es que no has posat la API key de Gemini, o no has canviat el hardcodeo de text")
            self.client = None
        else:
            try:
                self.client = genai.Client(api_key=api_key)
            except Exception as e:
                print(f"⚠️ Error initializing client: {e}")
                self.client = None                                            
    
                
        self.model_id = "gemini-2.5-flash"


        self.openai_client = None

        if CONFIG["USE_OPENAI"] and OPENAI_AVAILABLE:
            openai_key = os.environ.get("OPENAI_API_KEY")
            if openai_key:
                self.openai_client = AsyncOpenAI(api_key=openai_key)
                print("✅ OpenAI client ready")
            else:
                print("⚠️ USE_OPENAI=true però falta OPENAI_API_KEY")
        elif CONFIG["USE_OPENAI"] and not OPENAI_AVAILABLE:
            print("⚠️ USE_OPENAI=true però falta instal·lar openai")



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

        # 1) Primer intentem Gemini, si està activat i disponible
        if self.client:
            try:
                response = await self.client.aio.models.generate_content(
                    model=self.model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.7
                    )
                )
                print("🟢 Using Gemini")
                return json.loads(response.text)

            except Exception as e:
                print(f"⚠️ Gemini generation failed: {type(e).__name__}: {e}")

        # 2) Si Gemini no està activat o falla, intentem OpenAI
        if self.openai_client:
            try:
                response = await self.openai_client.responses.create(
                    model=CONFIG["OPENAI_MODEL"],
                    input=prompt
                )
                print("🟣 Using OpenAI")
                return json.loads(response.output_text)

            except Exception as oe:
                print(f"⚠️ OpenAI generation failed: {type(oe).__name__}: {oe}")

        # 3) Si tot falla, generador local
        local_gen = LocalRadioTextGenerator()
        return local_gen.generate()

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


class LocalRadioTextGenerator:
    def __init__(self):
        self.headers = [
            "RADIO BARCELONA E.A.J.-1",
            "GUIÓN DE EMISIÓN",
            "PROGRAMA DEL DÍA",
            "COMISARÍA DE RADIODIFUSIÓN",
            "DIRECCIÓN GENERAL DE RADIODIFUSIÓN",
        ]

        self.times = [
            "08:00", "09:15", "10:30", "13:00", "14:15",
            "16:00", "17:30", "19:00", "21:00", "22:15"
        ]

        self.programs = [
            "Hora exacta",
            "Lectura de la prensa de Barcelona",
            "Emisión de sobremesa",
            "Sesión radiobenéfica",
            "Comunicat de guerra",
            "Radioteatro de EAJ-1",
            "Música de cámara",
            "Boletín meteorológico",
            "Cotizaciones de bolsa",
            "Concierto del Trío Iberia",
            "Avisos oficiales",
            "Programa infantil",
            "Conferencia cultural",
        ]

        self.notes = [
            "Demanat permís",
            "Ver observaciones",
            "Revisado",
            "Pot radiarse",
            "Suprimir",
            "Censurado",
            "Pendiente de aprobación",
            "Cambiar orden",
            "No radiar",
            "Demanat permís abans de radiar",
            "Ver observaciones del censor",
            "Cambiar el orden de emisión",
            "No radiar hasta nueva revisión",
            "Aprobado con modificaciones",
            "Revisar tono político del comentario",
            "Suprimir referencia indicada",
            "Pendiente de autorización oficial",
        ]

    def generate(self):
        year = random.choice([1934, 1936, 1937, 1939, 1940, 1943, 1945, 1952])
        day = random.randint(1, 28)
        month = random.randint(1, 12)

        header = random.choice(self.headers)
        title = f"{header}\nFecha: {day:02d}/{month:02d}/{year}\n"

        lines = []
        chosen_times = [
            f"{random.randint(6, 23):02d}:{random.choice(['00', '15', '30', '45'])}"
            for _ in range(random.randint(CONFIG["MIN_PROGRAM_LINES"], CONFIG["MAX_PROGRAM_LINES"]))
        ]

        for idx, t in enumerate(chosen_times, start=1):
            program = random.choice(self.programs)
            duration = random.choice(["15 minutos", "30 minutos", "45 minutos", "1 hora"])
            lines.append(
                f"{idx:02d}. {t}    {program}    Duración: {duration}    "
                f"Observaciones: emisión prevista para Radio Barcelona E.A.J.-1."
            )

        documento_base = title + "\n" + "\n".join(lines)

        anotaciones = []
        for _ in range(random.randint(CONFIG["MIN_ANNOTATIONS"], CONFIG["MAX_ANNOTATIONS"])): # CONFIGURAR: número de anotaciones por documento
            tipo = random.choices(
                [
                    "insercion",
                    "tachon",
                    "censura_bloque",
                    "subrayado",
                    "circulo",
                    "sello",
                    "nota_larga",
                ],
                weights=[
                    2,   # insercion
                    5,   # tachon
                    6,   # censura_bloque
                    2,   # subrayado
                    1,   # circulo
                    5,   # sello
                    2,   # nota_larga
                ],
                k=1
            )[0]

            target_line = random.choice(lines)
            target_words = target_line.split()
            texto_a_tachar = " ".join(random.sample(target_words, min(2, len(target_words))))

            if tipo == "sello":
                anotaciones.append({
                    "tipo": "sello",
                    "texto": random.choice(["REVISADO", "CENSURADO", "APROBADO", "ORIGINAL"]),
                    "texto_a_tachar": None
                })
            elif tipo == "insercion":
                anotaciones.append({
                    "tipo": "insercion",
                    "texto": random.choice(self.notes),
                    "texto_a_tachar": None
                })
            elif tipo == "nota_larga":
                anotaciones.append({
                    "tipo": "nota_larga",
                    "texto": random.choice([
                        "Se autoriza la emisión después de revisar el contenido completo del programa.",
                        "Debe modificarse el fragmento indicado antes de la emisión pública.",
                        "Contenido pendiente de revisión por la autoridad competente antes de su radiodifusión.",
                        "Se recomienda cambiar el orden de los bloques y eliminar las referencias señaladas.",
                        "El programa puede radiarse únicamente si se suprimen las frases marcadas en rojo."
                    ]),
                    "texto_a_tachar": None
                })
            else:
                anotaciones.append({
                    "tipo": tipo,
                    "texto": "",
                    "texto_a_tachar": texto_a_tachar
                })

        for _ in range(random.randint(CONFIG["MIN_EXTRA_STAMPS"], CONFIG["MAX_EXTRA_STAMPS"])):
            anotaciones.append({
                "tipo": "sello",
                "texto": random.choice(["REVISADO", "CENSURADO", "APROBADO", "ORIGINAL", "NULO"]),
                "texto_a_tachar": None
            })

        for _ in range(random.randint(CONFIG["MIN_EXTRA_CENSORSHIP"], CONFIG["MAX_EXTRA_CENSORSHIP"])):
            target_line = random.choice(lines)
            words = target_line.split()
            target = " ".join(words[:random.randint(2, min(6, len(words)))])

            anotaciones.append({
                "tipo": random.choice(["tachon", "censura_bloque"]),
                "texto": "",
                "texto_a_tachar": target
            })


        return {
            "documento_base": documento_base,
            "anotaciones": anotaciones
        }



'''
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
'''

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

        # ------------------------------------------------------------------
        # ASSETS REALS DEL PROJECTE
        # ------------------------------------------------------------------
        # IMPORTANT:
        # - iam_samples/ continua existint i es fa servir per manuscrit genèric.
        # - assets/ és per retalls reals de Radio Barcelona: segells, censura,
        #   pegats, textures, etc.
        #
        # El codi funciona encara que aquestes carpetes estiguin buides.
        # Manuscrit genèric:
        # - iam_samples/ són mostres originals.
        # - assets_real_reviewed/handwriting/ són crops reals revisats manualment.
        self.iam_images = collect_assets_from_dirs(
            "iam_samples",
            "assets_real_reviewed/handwriting",
        )

        # Segells:
        # - assets/stamps/ són assets inicials.
        # - assets_real_reviewed/stamps/ són segells detectats en documents reals i validats.
        self.stamp_paths = collect_assets_from_dirs(
            "assets/stamps",
            "assets_real_reviewed/stamps",
        )

        # Censura:
        # - assets/censorship/ són assets inicials.
        # - assets_real_reviewed/censorship/ són blocs de censura validats.
        self.censorship_paths = collect_assets_from_dirs(
            "assets/censorship",
            "assets_real_reviewed/censorship",
        )

        # Pegats:
        # De moment només usem assets originals. Si més endavant revisem patches,
        # podem afegir assets_real_reviewed/patches/.
        self.patch_paths = collect_assets_from_dirs(
            "assets/patches",
        )

        # Esborrats / tatxadures:
        # Al generator, erasures són visualment crossouts/tatxadures.
        # Per això barregem assets/erasures amb assets_real_reviewed/crossouts.
        self.erasure_paths = collect_assets_from_dirs(
            "assets/erasures",
            "assets_real_reviewed/crossouts",
        )

        # Taules:
        # - assets/tables/ són assets inicials.
        # - assets_real_reviewed/tables/ són fragments de taules reals revisats.
        self.table_paths = collect_assets_from_dirs(
            "assets/tables",
            "assets_real_reviewed/tables",
        )

        print(
            "Assets loaded:",
            f"iam/handwriting={len(self.iam_images)}",
            f"stamps={len(self.stamp_paths)}",
            f"censorship={len(self.censorship_paths)}",
            f"patches={len(self.patch_paths)}",
            f"erasures/crossouts={len(self.erasure_paths)}",
            f"tables={len(self.table_paths)}",
        )


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
        size = random.randint(
            CONFIG["TYPEWRITER_FONT_MIN_SIZE"],
            CONFIG["TYPEWRITER_FONT_MAX_SIZE"]
        )
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

    # ------------------------------------------------------------------
    # HELPERS PER ENGANXAR ASSETS REALS
    # ------------------------------------------------------------------
    def _load_asset_as_ink(self, asset_path: str) -> Image.Image:
        """
        Carrega un asset real i elimina el fons blanc/gris clar.
        Això és millor que Multiply per segells, tatxadures i signatures.
        """
        asset = Image.open(asset_path)

        # Per segells, tatxadures i retalls amb fons blanc:
        asset = remove_white_background(
            asset,
            threshold=235
        )

        return asset

    def _paste_asset_random(self, canvas: Image.Image, asset_paths: List[str],
                            min_scale: float = 0.4, max_scale: float = 0.9,
                            rotation: float = 45.0, margin: int = 100, asset_class: str = "asset") -> dict | None:
        """
        Enganxa un asset en una posició aleatòria de la capa.

        Retorna True si ha pogut enganxar-lo, False si no hi havia assets
        o si l'asset era massa gran.
        """
        if not asset_paths:
            return False

        try:
            asset = self._load_asset_as_ink(random.choice(asset_paths))
        except Exception:
            return False

        # Redimensionem mantenint proporció.
        scale = random.uniform(min_scale, max_scale)
        new_w = max(1, int(asset.width * scale))
        new_h = max(1, int(asset.height * scale))
        asset = asset.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Rotació per evitar que tots els retalls semblin enganxats igual.
        asset = asset.rotate(
            random.uniform(-rotation, rotation),
            expand=True,
            resample=Image.BICUBIC
        )

        # Si l'asset és massa gran, no l'enganxem.
        if asset.width >= self.width - 2 * margin or asset.height >= self.height - 2 * margin:
            return False

        x = random.randint(margin, self.width - asset.width - margin)
        y = random.randint(margin, self.height - asset.height - margin)

        canvas.paste(asset, (x, y), asset)
        return {
        "class": asset_class,
        "bbox": {"x1": x, "y1": y, "x2": x + asset.width, "y2": y + asset.height},
        "confidence": 1.0,
        "source": "synthetic_generator"
        }
    

    def _paste_asset_over_box(self, canvas: Image.Image, asset_paths: List[str],
                              box: Tuple[int, int, int, int],
                              rotation: float = 6.0) -> bool:
        """
        Enganxa un asset real sobre una caixa concreta del text.

        Útil per a censura/tatxadures: detectem una frase i hi posem
        una línia real o un bloc real damunt.
        """
        if not asset_paths:
            return False

        x0, y0, x1, y1 = box
        target_w = max(10, x1 - x0)
        target_h = max(10, y1 - y0)

        try:
            asset = self._load_asset_as_ink(random.choice(asset_paths))
        except Exception:
            return False

        # Ajustem l'asset a la mida de la frase objectiu, amb una mica de marge.
        scale_w = target_w / max(1, asset.width)
        scale_h = (target_h * random.uniform(0.6, 1.5)) / max(1, asset.height)
        scale = max(scale_w, scale_h)

        new_w = max(1, int(asset.width * scale * random.uniform(0.9, 1.15)))
        new_h = max(1, int(asset.height * scale * random.uniform(0.8, 1.2)))
        asset = asset.resize((new_w, new_h), Image.Resampling.LANCZOS)

        asset = asset.rotate(
            random.uniform(-rotation, rotation),
            expand=True,
            resample=Image.BICUBIC
        )

        # Centrem aproximadament l'asset sobre la frase.
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        paste_x = int(cx - asset.width // 2 + random.randint(-5, 5))
        paste_y = int(cy - asset.height // 2 + random.randint(-5, 5))

        paste_x = max(0, min(paste_x, self.width - asset.width))
        paste_y = max(0, min(paste_y, self.height - asset.height))

        canvas.paste(asset, (paste_x, paste_y), asset)
        return True

    def render_layer0(self, text: str) -> Tuple[Image.Image, List[Dict[str, Any]]]:
        # Capa 0: text mecanografiat sobre fons transparent.
        # La convertirem a fons blanc només al moment de fer Multiply.
        img = Image.new('RGBA', (self.width, self.height), (255, 255, 255, 0))
        draw = ImageDraw.Draw(img)

        tw_font = self.get_document_typewriter_font()

        margin_x, margin_y = random.randint(120, 200), random.randint(150, 250)
        current_y = margin_y
        line_height = random.randint(
            CONFIG["LINE_HEIGHT_MIN"],
            CONFIG["LINE_HEIGHT_MAX"]
        )
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

    # def render_layer1(self, word_boxes: List[Dict[str, Any]], anotaciones: List[Dict[str, Any]]) -> Image.Image:
    def render_layer1(self, word_boxes: List[Dict[str, Any]], anotaciones: List[Dict[str, Any]]):
        import textwrap

        # Capa 1: anotacions / segells / censura.
        # Ara té fons blanc perquè després farem ImageChops.multiply().
        # Amb Multiply, el blanc actua com a element neutre i la tinta fosca
        # s'integra millor sobre textures reals de paper.
        img = Image.new('RGBA', (self.width, self.height), (255, 255, 255, 255))
        draw = ImageDraw.Draw(img)
        ink_colors = [
            (10, 30, 150, 240), (20, 20, 20, 250), (180, 20, 20, 230),
            (20, 100, 30, 220), (90, 40, 140, 200), (80, 90, 100, 210)
        ]

        stamp_detections = []
        layout_detections = []

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
                        # Si tenim retalls reals de censura, els usem primer.
                        # Això fa que les tatxadures siguin molt menys artificials.
                        used_real_censor_asset = self._paste_asset_over_box(
                            img,
                            self.censorship_paths,
                            (x0, y0, x1, y1),
                            rotation=6.0
                        )

                        if not used_real_censor_asset:
                            # Fallback antic: bloc negre procedural.
                            # Es manté perquè el codi funcioni encara que assets/censorship sigui buit.
                            altura_texto = y1 - y0
                            margen_vertical = altura_texto // 4

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


                    # guardar anotació posicional de tachons / censura
                    if tipo in ["tachon", "censura_bloque"]:
                        layout_detections.append({
                            "class": "crossout" if tipo == "tachon" else "censorship_block",
                            "bbox": {"x1": x0, "y1": y0, "x2": x1, "y2": y1},
                            "confidence": 1.0,
                            "source": "synthetic_generator",
                            "target_text": texto_tachar,
                        })


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
                        layout_detections.append({
                            "class": "handwritten_text",
                            "bbox": {"x1": x, "y1": y, "x2": x + decal.width, "y2": y + decal.height},
                            "confidence": 1.0,
                            "source": "synthetic_generator",
                            "target_text": texto,
                            "subtype": "iam_sample",
                        })
                else:
                    x = random.randint(100, self.width - 400)
                    y = random.randint(100, self.height - 200)
                    font = self.get_random_cursive_font(60, 100)
                    try:
                        bbox_text = draw.textbbox((x, y), texto_seguro, font=font)
                        draw.text((x, y), texto_seguro, font=font, fill=main_ink)
                    except OSError:
                        font = ImageFont.load_default()
                        bbox_text = draw.textbbox((x, y), texto_seguro, font=font)
                        draw.text((x, y), texto_seguro, font=font, fill=main_ink)

                    layout_detections.append({
                        "class": "handwritten_text",
                        "bbox": {
                            "x1": bbox_text[0],
                            "y1": bbox_text[1],
                            "x2": bbox_text[2],
                            "y2": bbox_text[3],
                        },
                        "confidence": 1.0,
                        "source": "synthetic_generator",
                        "target_text": texto,
                        "subtype": "procedural_font",
                    })

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

                    layout_detections.append({
                        "class": "handwritten_text",
                        "bbox": {
                            "x1": x - 50,
                            "y1": y - 50,
                            "x2": x - 50 + txt_img.width,
                            "y2": y - 50 + txt_img.height,
                        },
                        "confidence": 1.0,
                        "source": "synthetic_generator",
                        "target_text": texto,
                        "subtype": "long_note",
                    })

            # -----------------------------------------------------------------
            # 3. SELLOS / FIRMAS GIGANTES
            # -----------------------------------------------------------------
            elif tipo == "sello":
                # Primer intentem fer servir un segell REAL de assets/stamps/.
                # Si no n'hi ha, o aleatòriament no toca, fem servir el comportament antic.
                if self.stamp_paths and random.random() < CONFIG["REAL_STAMP_PROB"]: # 90% de probabilitat de usar un segell real si hay disponibles
                    used_stamp = self._paste_asset_random(
                        img,
                        self.stamp_paths,
                        min_scale=0.35,
                        max_scale=0.85,
                        rotation=45.0,
                        margin=80,
                        asset_class="official_stamp"
                    )
                    if used_stamp:
                        stamp_detections.append(used_stamp)
                        continue

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
                    stamp_detections.append({
                        "class": "synthetic_stamp",
                        "bbox": {"x1": x, "y1": y, "x2": x + txt_img.width, "y2": y + txt_img.height},
                        "confidence": 1.0,
                        "source": "synthetic_generator"
                    })

        # Pegats reals opcionals: són independents de les anotacions textuals.
        # Exemple: paper enganxat, cinta, fragments físics. Es posa molt poc sovint
        # perquè si abusem d'això el document queda massa artificial.
        # CONFIGURAR 
        if self.patch_paths and random.random() < CONFIG["PATCH_PROB"]:
            for _ in range(random.randint(1, 2)):
                self._paste_asset_random(
                    img,
                    self.patch_paths,
                    min_scale=0.3,
                    max_scale=1.0,
                    rotation=20.0,
                    margin=60
                )

        # Esborrats reals opcionals
        if self.erasure_paths and random.random() < CONFIG["ERASURE_PROB"]:
            for _ in range(random.randint(CONFIG["MIN_EXTRA_ERASURES"], CONFIG["MAX_EXTRA_ERASURES"])):
                used_erasure = self._paste_asset_random(
                    img,
                    self.erasure_paths,
                    min_scale=0.3,
                    max_scale=0.9,
                    rotation=10.0,
                    margin=80,
                    asset_class="crossout"
                )

                if used_erasure:
                    layout_detections.append({
                        "class": "crossout",
                        "bbox": used_erasure["bbox"],
                        "confidence": 1.0,
                        "source": "synthetic_generator",
                        "target_text": None,
                        "subtype": "erasure_asset",
                    })

        # Fragments de taules reals opcionals
        if self.table_paths and random.random() < CONFIG["TABLE_PROB"]:
            used_table = self._paste_asset_random(
                img,
                self.table_paths,
                min_scale=0.4,
                max_scale=1.0,
                rotation=4.0,
                margin=120,
                asset_class="table_fragment"
            )

            if used_table:
                layout_detections.append({
                    "class": "table_fragment",
                    "bbox": used_table["bbox"],
                    "confidence": 1.0,
                    "source": "synthetic_generator",
                    "target_text": None,
                    "subtype": "table_asset",
                })



        # Segells reals extra independents
        if self.stamp_paths and random.random() < CONFIG["EXTRA_STAMP_PROB"]:
            for _ in range(random.randint(1, 3)):
                used_stamp = self._paste_asset_random(
                    img,
                    self.stamp_paths,
                    min_scale=0.25,
                    max_scale=0.65,
                    rotation=35.0,
                    margin=80,
                    asset_class="official_stamp",
                )

                if used_stamp:
                    stamp_detections.append(used_stamp)

                    

        # Tatxadures/censures extra independents
        if self.censorship_paths and random.random() < CONFIG["EXTRA_CENSORSHIP_PROB"]:
            for _ in range(random.randint(2, 5)):
                used_censor = self._paste_asset_random(
                    img,
                    self.censorship_paths,
                    min_scale=0.25,
                    max_scale=0.8,
                    rotation=10.0,
                    margin=80,
                    asset_class="crossout"
                )

                if used_censor:
                    layout_detections.append({
                        "class": "crossout",
                        "bbox": used_censor["bbox"],
                        "confidence": 1.0,
                        "source": "synthetic_generator",
                        "target_text": None,
                        "subtype": "censorship_asset",
                    })


        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.3, 1.0)))
        
        return img, stamp_detections, layout_detections

class AugmentationPipeline:
    def __init__(self, width: int, height: int):
        self.width, self.height = width, height

        # Textures reals opcionals. Si les carpetes estan buides, el codi
        # continuarà generant paper vell procedural com abans.
        self.paper_paths = (
            glob.glob("assets/paper_textures/*.jpg") +
            glob.glob("assets/paper_textures/*.jpeg") +
            glob.glob("assets/paper_textures/*.png")
        )
        self.stain_paths = (
            glob.glob("assets/stains/*.png") +
            glob.glob("assets/stains/*.jpg") +
            glob.glob("assets/stains/*.jpeg")
        )

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

    def get_real_paper_bg(self) -> Image.Image:
        """
        Retorna una textura de paper real si n'hi ha a assets/paper_textures/.
        Si no, usa el generador procedural antic.
        """
        if self.paper_paths:
            try:
                bg = Image.open(random.choice(self.paper_paths)).convert("RGBA")
                return bg.resize((self.width, self.height), Image.Resampling.LANCZOS)
            except Exception:
                pass

        # Fallback: paper sintètic procedural original.
        return self.generate_paper_background()

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
        """
        Composició final amb mode Multiply.

        Abans: alpha_composite(paper, layer0, layer1)
        Ara:
            1. paper real/procedural
            2. layer0 i layer1 sobre blanc
            3. Multiply(paper, tinta_combinada)

        Per què Multiply?
        - Les imatges escanejades de segells/tatxadures sovint tenen fons blanc.
        - Amb Multiply, el blanc desapareix visualment i només queda la tinta.
        - Això integra molt millor els retalls sobre el paper vell.
        """
        bg = self.get_real_paper_bg()

        # layer0 original és transparent. El convertim a fons blanc perquè
        # Multiply necessita una imatge tipus "paper blanc + tinta".
        white_bg = Image.new("RGBA", l0.size, (255, 255, 255, 255))
        l0_white = Image.alpha_composite(white_bg, l0)

        # layer1 ja ve amb fons blanc des de render_layer1().
        # Multipliquem les dues capes de tinta perquè el blanc es mantingui blanc
        # i les tintes de totes dues capes quedin combinades.
        combined_ink = ImageChops.multiply(l0_white, l1)

        # Aplicació principal de Multiply sobre el paper.
        final_comp = ImageChops.multiply(bg, combined_ink)

        # Taques reals opcionals. Si són imatges amb fons blanc, Multiply també
        # funciona bé: només s'hi integren les parts fosques o acolorides.
        if self.stain_paths and random.random() < CONFIG["STAIN_PROB"]:
            try:
                stain = Image.open(random.choice(self.stain_paths)).convert("RGBA")
                stain = stain.resize((self.width, self.height), Image.Resampling.LANCZOS)
                final_comp = ImageChops.multiply(final_comp, stain)
            except Exception:
                pass

        comp_arr = np.array(final_comp.convert('RGB'))
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
        l1_img, stamp_detections, layer1_detections = self.renderer.render_layer1(word_boxes, anotaciones)
        final_img = self.aug_pipe.composite(l0_img, l1_img)

        layout_objects = []

        # 1. Text mecanografiat
        layout_objects.extend(build_typewritten_objects(word_boxes, i))

        # 2. Stamps, mantenint compatibilitat amb stamp_detections
        for j, det in enumerate(stamp_detections, start=1):
            layout_objects.append(
                make_layout_object(
                    obj_id=f"stamp_{i:04d}_{j:03d}",
                    obj_type="stamp",
                    bbox=det["bbox"],
                    text=None,
                    layer="layer1_annotations",
                    confidence=det.get("confidence", 1.0),
                    extra={
                        "subtype": det.get("class", "official_stamp"),
                        "legacy_class": det.get("class", "official_stamp"),
                        "notes": "synthetic ground truth",
                    },
                )
            )
        
        # 3. Altres marques de layer1: tachons, censura, etc.
        for j, det in enumerate(layer1_detections, start=1):
            layout_objects.append(
                make_layout_object(
                    obj_id=f"{det.get('class', 'layer1_object')}_{i:04d}_{j:03d}",
                    obj_type=det.get("class", "layer1_object"),
                    bbox=det["bbox"],
                    text=det.get("target_text"),
                    layer="layer1_annotations",
                    confidence=det.get("confidence", 1.0),
                    extra={
                        "subtype": det.get("subtype"),
                        "notes": "synthetic ground truth",
                    },
                )
            )

        # IO BOUND (Disk)
        sample_dir = os.path.join(self.output_dir, f"sample_{i:04d}")
        os.makedirs(sample_dir, exist_ok=True)

        with open(os.path.join(sample_dir, "text_original.txt"), "w", encoding="utf-8") as f:
            f.write(texto_base)

        with open(os.path.join(sample_dir, "text_annotations.json"), "w", encoding="utf-8") as f:
            json.dump(anotaciones, f, indent=4, ensure_ascii=False)
        
        with open(os.path.join(sample_dir, "stamp_detections.json"), "w", encoding="utf-8") as f:
            json.dump({
                "document_id": f"synthetic_{i:04d}",
                "page": 1,
                "image": f"final_merged_{i:04d}.jpg",
                "stamp_detections": [
                    {
                        "id": f"stamp_{i:04d}_{j:03d}",
                        "class": det.get("class", "official_stamp"),
                        "bbox": det["bbox"],
                        "confidence": det.get("confidence", 1.0),
                        "crop_path": None,
                        "mask_path": None,
                        "ocr_text": None,
                        "notes": "synthetic ground truth"
                    }
                    for j, det in enumerate(stamp_detections, start=1)
                ]
            }, f, indent=2, ensure_ascii=False)

        with open(os.path.join(sample_dir, "layout_annotations.json"), "w", encoding="utf-8") as f:
            json.dump({
                "document_id": f"synthetic_{i:04d}",
                "page": 1,
                "image": f"final_merged_{i:04d}.jpg",
                "image_width": self.renderer.width,
                "image_height": self.renderer.height,
                "objects": layout_objects,
            }, f, indent=2, ensure_ascii=False)

        final_filename = f"final_merged_{i:04d}.jpg"
        final_img.save(os.path.join(self.finals_dir, final_filename), quality=random.randint(70, 95))

        white_bg = Image.new("RGBA", l0_img.size, (255, 255, 255, 255))
        l0_white = Image.alpha_composite(white_bg, l0_img)
        l1_white = Image.alpha_composite(white_bg, l1_img)

        l0_white.save(os.path.join(sample_dir, "image_layer0_clean.png"))
        l1_white.save(os.path.join(sample_dir, "image_layer1_clean.png"))

        #print("TEXT PREVIEW:", texto_base[:200])
        #print("N ANOTACIONS:", len(anotaciones))

    # NEW: Async orchestrator function for a single item
    async def process_item(self, i: int, pbar: tqdm):
        # 1. Await the IO-bound API request
        data = await self.text_gen.generate_random_script_async()

        # 2. Push the CPU-bound rendering to a separate thread so it doesn't block the async event loop
        await asyncio.to_thread(self.render_and_save_sync, data, i)

        # Update progress bar
        pbar.update(1)


async def main():
    # Nombre de mostres a generar.
    # Pots canviar-ho amb una variable d'entorn sense editar el fitxer:
    #   TOTAL_SAMPLES=100 python generator.py
    TOTAL_SAMPLES = CONFIG["TOTAL_SAMPLES"]

    print(f"Generating {TOTAL_SAMPLES} highly advanced samples asynchronously...")

    # Creem totes les carpetes necessàries si no existeixen.
    os.makedirs("fonts", exist_ok=True)
    os.makedirs("iam_samples", exist_ok=True)
    os.makedirs("typewriter_fonts", exist_ok=True)
    os.makedirs("assets/stamps", exist_ok=True)
    os.makedirs("assets/censorship", exist_ok=True)
    os.makedirs("assets/patches", exist_ok=True)
    os.makedirs("assets/paper_textures", exist_ok=True)
    os.makedirs("assets/stains", exist_ok=True)
    os.makedirs("assets/erasures", exist_ok=True)
    os.makedirs("assets/tables", exist_ok=True)

    os.makedirs("assets_real_reviewed/stamps", exist_ok=True)
    os.makedirs("assets_real_reviewed/handwriting", exist_ok=True)
    os.makedirs("assets_real_reviewed/crossouts", exist_ok=True)
    os.makedirs("assets_real_reviewed/censorship", exist_ok=True)
    os.makedirs("assets_real_reviewed/tables", exist_ok=True)
    # Target ~850 RPM to stay safely under the 1000 RPM limit.
    # 60 seconds / 850 requests = ~0.07 seconds between starting each request.
    #REQUEST_DELAY = 0.07
    #MAX_CONCURRENT_TASKS = 60 # Prevent memory explosion from too many heavy image generations at once
    MAX_CONCURRENT_TASKS = CONFIG["MAX_CONCURRENT_TASKS"]
    REQUEST_DELAY = CONFIG["REQUEST_DELAY"]

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

    print("Done! Check 'output_dataset_pro_try'.")

if __name__ == "__main__":
    if os.name == 'nt':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(main())
