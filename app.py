"""
Servidor web de CIPIE.
Sirve el sitio promocional estático (páginas HTML, CSS e imágenes) y expone
un endpoint seguro de chat con IA (Gemini) que mantiene la clave en el
servidor, nunca en el navegador.

Uso:
    # Define la clave (nunca la subas al repositorio):
    #   PowerShell:  $env:GEMINI_API_KEY = "tu_clave"
    #   Bash:        export GEMINI_API_KEY="tu_clave"
    python app.py
Luego abre http://localhost:8000 en el navegador.
"""

import json
import os
import urllib.error
import urllib.request

from flask import Flask, jsonify, request, send_from_directory, abort

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

# Clave de Gemini leída del entorno. Si no está definida, el endpoint /api/chat
# responde 503 y el frontend usa su respaldo local automáticamente.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash:generateContent"
)

# Límites defensivos para el endpoint público.
MAX_MESSAGE_CHARS = 1000
MAX_HISTORY_TURNS = 20

SYSTEM_PROMPT = """Eres el asistente virtual oficial del CIPIE (Colegio Interdisciplinario de Profesionistas en Innovación Educativa).
Responde SIEMPRE en español, de forma amable, concisa y clara. No uses listas con asteriscos, usa guiones o puntos.
No inventes información. Si no sabes algo, deriva al WhatsApp +52 55 2901 8664 o al correo controlescolar@cipie.edu.mx.

INFORMACIÓN OFICIAL DE CIPIE:

INSTITUCIÓN:
- Nombre completo: Colegio Interdisciplinario de Profesionistas en Innovación Educativa (CIPIE)
- Sitio web: cipie.edu.mx
- Campus virtual: cipiecampusvirtual.moodlecloud.com (plataforma Moodle)
- Fundación: 2022
- Incorporado a DGAIR-SEP con programas de RVOE vigente
- Enfoque: neurodidáctica aplicada, tecnología educativa, formación interdisciplinaria

CONTACTO:
- WhatsApp y teléfono: +52 55 2901 8664
- Correo: controlescolar@cipie.edu.mx
- Facebook: facebook.com/cipie.cipie.2025
- Instagram: instagram.com/colecipie

LICENCIATURAS (9 cuatrimestres cada una):
1. Licenciatura en Pedagogía – RVOE 20221456 (20 octubre 2024) – Modalidad mixta
2. Licenciatura en Derecho – RVOE 20240616 (19 marzo 2024) – Modalidad escolar
3. Licenciatura en Administración de Empresas – RVOE 20251876 (16 julio 2025) – Modalidad no escolarizada (100% en línea)

POSGRADOS (4 cuatrimestres cada uno):
1. Maestría en Ciencias de la Educación – RVOE 20221170 (23 septiembre 2022) – Modalidad escolar
2. Maestría en Juicios Orales – RVOE 20221457 (19 octubre 2022) – Modalidad escolar
3. Doctorado en Educación – RVOE 20221169 (23 septiembre 2022) – Modalidad escolar
4. Doctorado en Derecho – RVOE 20251877 (16 julio 2025) – Modalidad no escolarizada

MODALIDADES:
- Escolarizada: clases presenciales con horarios fijos
- Mixta/Híbrida: combina presencial con actividades en línea
- No escolarizada: 100% en línea vía Moodle con neurodidáctica y recursos de vanguardia

ADMISIONES (requisitos generales):
- Certificado de bachillerato (para licenciaturas) o título de licenciatura (para posgrados)
- Identificación oficial vigente
- CURP
- Para más detalles contactar por WhatsApp

VALORES INSTITUCIONALES:
Excelencia, Innovación, Integridad, Responsabilidad Social, Disciplina, Pensamiento Crítico, Liderazgo, Trabajo Colaborativo."""

# static_folder=None: gestionamos manualmente el envío de archivos para poder
# resolver rutas sin extensión (p. ej. /pedagogia -> pedagogia.html).
app = Flask(__name__, static_folder=None)


@app.route("/api/chat", methods=["POST"])
def chat():
    """Proxy seguro hacia Gemini. La clave nunca se expone al navegador."""
    if not GEMINI_API_KEY:
        # Sin clave configurada: el frontend usará su respaldo local.
        return jsonify(error="IA no configurada"), 503

    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    if not message:
        return jsonify(error="Mensaje vacío"), 400
    if len(message) > MAX_MESSAGE_CHARS:
        return jsonify(error="Mensaje demasiado largo"), 400

    # Reconstruye el historial recibido del cliente de forma controlada.
    contents = []
    for turn in (data.get("history") or [])[-MAX_HISTORY_TURNS:]:
        role = turn.get("role")
        text = (turn.get("text") or "").strip()
        if role in ("user", "model") and text:
            contents.append({"role": role, "parts": [{"text": text[:MAX_MESSAGE_CHARS]}]})
    contents.append({"role": "user", "parts": [{"text": message}]})

    payload = json.dumps(
        {
            "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "contents": contents,
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 400,
                "topP": 0.9,
            },
        }
    ).encode("utf-8")

    req = urllib.request.Request(
        GEMINI_URL,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": GEMINI_API_KEY,
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, ValueError):
        return jsonify(error="No se pudo contactar la IA"), 502

    reply = ""
    try:
        reply = result["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError, TypeError):
        reply = ""

    if not reply:
        return jsonify(error="Sin respuesta"), 502

    return jsonify(reply=reply)


@app.route("/")
def home():
    """Página de inicio."""
    return send_from_directory(BASE_DIR, "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    """Sirve cualquier archivo del sitio (HTML, CSS, imágenes, etc.)."""
    full_path = os.path.join(BASE_DIR, filename)

    # Si piden una ruta sin extensión, intenta servir el .html correspondiente.
    if not os.path.isfile(full_path) and not os.path.splitext(filename)[1]:
        html_candidate = f"{filename}.html"
        if os.path.isfile(os.path.join(BASE_DIR, html_candidate)):
            return send_from_directory(BASE_DIR, html_candidate)

    if not os.path.isfile(full_path):
        abort(404)

    return send_from_directory(BASE_DIR, filename)


if __name__ == "__main__":
    # En Windows, bind a "0.0.0.0" dispara un getfqdn() lento que parece colgar.
    # Por defecto usamos 127.0.0.1 en local; en despliegue define HOST=0.0.0.0.
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", 8000))
    app.run(host=host, port=port, debug=True)
