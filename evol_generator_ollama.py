"""
ollama_evolutivo_generator_delimited.py

Generates Spanish hospital-style clinical progress notes (evolutivos) from source clinical cases using Ollama.

Strategy: no stop tokens. The model generates freely up to num_predict tokens.
The note is extracted via regex between ---INICIO--- and ---FIN--- delimiters.
If delimiters are missing, the full content (minus think blocks) is used as fallback.
think=True is passed at request root level so Ollama separates thinking from content.
Resume support: already-generated output files are skipped on restart.
"""

import json
import hashlib
import os
import re
import time
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import requests



INPUT_DIR = ""
OUT_DIR = ""
LOG_CSV_PATH = ""
DEFAULT_LOG_CSV_NAME = "log.csv"

OLLAMA_BASE_URL = "http://127.0.0.1:11434"
OLLAMA_MODEL = ("qwen3.5:35b-a3b")
#OLLAMA_MODEL = ("glm-5:cloud")


REQUEST_TIMEOUT_SECONDS = 600

NUM_PREDICT = 3000
TEMPERATURE = 0.2
TOP_P = 0.9
TOP_K = 40
REPEAT_PENALTY = 1.05
SEED = 12345

MIN_OUTPUT_CHARS = 80

SYSTEM_PROMPT = (
    "Actúa como un médico especialista en Urología u Oncología que redacta notas evolutivas clínicas "
    "en una historia clínica electrónica hospitalaria en español. "
    "Devuelve únicamente la nota clínica solicitada, sin explicaciones ni metatexto."
)

USER_PROMPT_TEMPLATE = """
Se te proporcionará un caso clínico completamente anotado mediante etiquetas XML que identifican entidades clínicas (por ejemplo: `<ENFERMEDAD>`, `<Date>`, `<GLEASON>`, etc.). A partir de este caso clínico, genera un texto equivalente con el estilo y estructura propios de la narrativa de historia clínica de un paciente de cáncer de próstata, tal y como podría aparecer en un sistema de historia clínica real de un centro sanitario.

Las posibles etiquetas XML que identifican entidades clínicas son las siguientes: [`<SINTOMA>`, `<PROCEDIMIENTO>`, `<ENFERMEDAD>`, `<Age>`, `<Date>`, `<Dose>`, `<Duration>`, `<Frequency>`, `<Neg_cue>`, `<Negated>`, `<Spec_cue>`, `<Speculated>`, `<Time>`, `<PROTEINAS>`, `<UNCLEAR>`, `<SPECIES>`, `<HUMAN>`, `<NORMALIZABLES>` `<GLEASON>`, `<PSA>`].

El texto generado debe:

El texto generado debe:

1. Mantener estrictamente la información y los datos clínicamente relevantes desde el punto de vista del manejo médico urológico.
2.Prescindir única y exclusivamente de datos que no tengan relevancia ni significación clínica desde el punto de vista del manejo (diagnóstico, tratamiento, plan de actuación, estadificación, etc.) de la patología de cáncer de próstata.
3. No añadir datos nuevos que no estén explícitamente presentes.
4. No omitir información clínicamente relevante para la patología de cáncer de próstata.
5. No inferir resultados analíticos, radiológicos, de estadiaje, de PSA, de Gleason, ni datos anatomopatológicos si no aparecen en el caso clínico original.
6. Reformular completamente el texto (no copiar frases literalmente) con el estilo y estructura de narrativa de historia clínica.
7. Utilizar estilo clínico habitual en historias clínicas de pacientes reales de cáncer de próstata:
    - Utiliza las frases más breves posibles, directas y clínicamente informativas.
    - Minimiza y condensa al máximo el texto generado, omitiendo todo contenido prescindible, evitando redundancias y priorizando frases breves, compactas y de alta densidad informativa.
    - Evita el lenguaje narrativo o académico del caso clínico original, propio de artículos científicos pero no de historia clínica.
    - No incluyas comentarios explicativos, conectores entre frases ni texto de relleno prescindible.
    - Evita el uso de verbos en las frases siempre que sea posible (p. ej., "Evolución favorable con PSA 0.04 a las 4 semanas" mejor que "Presenta evolución favorable con PSA 0.04 a las 4 semanas").
    - Emplea las abreviaturas clínicas habituales cuando sea apropiado (p. ej., ADC, dx, tto, pte, RT, UCI, ADT, etc.).
    - Utiliza terminología médica habitual en la práctica clínica hospitalaria española.
8. Adapta, en la medida de lo posible, la prosa narrativa extensa del caso clínico original en un formato parecido a este estilo (sin que las secciones se muestre de forma explícita):
    - Información paciente.
    - Diagnóstico inicial + fecha + tratamiento inicial.
    - Progresión + momento temporal + manejo.
    - Complicación/evoluciones + hallazgos + actuación.
    - Situación final + evolución clínica.
9. Mantener coherencia temporal, tal y como está reflejada en el caso clínico original (fechas, intervalos temporales, etc.).
10. Conservar las dosis (sin necesidad específica de mantener las unidades), frecuencias, procedimientos y diagnósticos del caso clínico original.
11. NO Contener las etiquetas marcadas en ninguna parte del texto que generes.

No expliques lo que haces. Devuelve únicamente la nota clínica generada.

EJEMPLO 1:
- Caso clínico original:
    "Presentamos el caso de un <HUMAN>varón</HUMAN> de <Age>71 años</Age> diagnosticado en <Date>1995</Date> de <ENFERMEDAD>adenocarcinoma de próstata</ENFERMEDAD> estadio <SINTOMA>pT3</SINTOMA>c <GLEASON>Gleason 7</GLEASON>, con <SINTOMA><PROCEDIMIENTO>tomografía computarizada</PROCEDIMIENTO> y <PROCEDIMIENTO>gammagrafía ósea</PROCEDIMIENTO> normales</SINTOMA> en el momento del diagnóstico. Fue tratado con <PROCEDIMIENTO>hormonoterapia</PROCEDIMIENTO> neoadyuvante y <PROCEDIMIENTO>prostatectomía radical</PROCEDIMIENTO>, precisando posteriormente la <PROCEDIMIENTO>colocación de un esfínter urinario artificial</PROCEDIMIENTO>.
        <Duration>Dos años</Duration> después de la <PROCEDIMIENTO>cirugía</PROCEDIMIENTO> presentó progresión bioquímica de la enfermedad, confirmándose <PROCEDIMIENTO>histológicamente</PROCEDIMIENTO> <ENFERMEDAD>recidiva a nivel de la unión uretrovesical</ENFERMEDAD>. Se decidió instaurar <PROCEDIMIENTO>bloqueo androgénico total</PROCEDIMIENTO>, que fue mal tolerado por <ENFERMEDAD>hepatotoxicidad</ENFERMEDAD>, por lo que desde <Date>2002</Date> continuó tratamiento con <PROCEDIMIENTO>análogo <Frequency>trimestral</Frequency></PROCEDIMIENTO> y <PROCEDIMIENTO>antiandrógeno</PROCEDIMIENTO>.
        En <Date>2006</Date> se objetivó mediante <PROCEDIMIENTO>tomografía computarizada</PROCEDIMIENTO> <ENFERMEDAD>ureterohidronefrosis derecha</ENFERMEDAD> con <ENFERMEDAD>atrofia renal</ENFERMEDAD> secundaria, así como una <ENFERMEDAD>metástasis blástica en la pala ilíaca</ENFERMEDAD> derecha. El <PROCEDIMIENTO>estudio anatomopatológico</PROCEDIMIENTO> confirmó <ENFERMEDAD>adenocarcinoma de próstata</ENFERMEDAD> <GLEASON>Gleason 4+5</GLEASON>, con un <PROTEINAS>PSA</PROTEINAS> de <PSA>0,5 ng/mL</PSA>.
        En <Date>2007</Date> ingresó nuevamente para la <PROCEDIMIENTO>retirada del <Negated>manguito del esfínter artificial</Negated></PROCEDIMIENTO> por extrusión del mismo, siendo necesaria <Duration>meses</Duration> después la <PROCEDIMIENTO>retirada del <Negated>reservorio</Negated></PROCEDIMIENTO> por <SINTOMA>molestias locales</SINTOMA> y <SINTOMA>signos inflamatorios cutáneos</SINTOMA>. La <PROCEDIMIENTO>biopsia del tejido perirreservorio</PROCEDIMIENTO> fue <Spec_cue>compatible con</Spec_cue> <Speculated><ENFERMEDAD>adenocarcinoma de próstata</Speculated></ENFERMEDAD>, presentando en ese momento cifras de <PROTEINAS>PSA</PROTEINAS> de <PSA>0,14 ng/mL</PSA>.
        <Duration>Dos meses</Duration> más tarde reingresó por <ENFERMEDAD>insuficiencia renal aguda obstructiva</ENFERMEDAD>, con <NORMALIZABLES>creatinina</NORMALIZABLES> de 6,3 mg/dL y <NORMALIZABLES>urea</NORMALIZABLES> de 133 mg/dL. Se <PROCEDIMIENTO>colocó un catéter de nefrostomía derecha</PROCEDIMIENTO>, confirmándose además la progresión de las <ENFERMEDAD>metástasis óseas</ENFERMEDAD>. <Duration>Cuatro meses</Duration> después del diagnóstico de la <ENFERMEDAD>metástasis subcutánea</ENFERMEDAD>, el <HUMAN>paciente</HUMAN> presentaba un <SINTOMA>deterioro progresivo de su estado general</SINTOMA>.""

- Texto generado (estilo historia clínica de paciente de cáncer de próstata):
            "Varón de 71 años.
            - ADC próstata estadio pT3c (Gleason 7) en 1995. Sin enfermedad a distancia. ADT neoadyuvante + Prostatectomía Radical. Posteriormente colocación esfínter artificial.
            - Recidiva en unión uretrovesical 2 años post PR. ADT mal tolerada por hepatotoxicidad. Análogo trimestral + antiandrógeno desde 2002.
            - UPO derecha con atrofia renal + progresión metastásica en pala ilíaca derecha en TAC de revisión en 2006 (Gleason 9 (4+5), PSA 0,5).
            - Extrusión manguito esfínter artificial, retirado en 2007. Posteriormente retirada de reservorio por molestias e inflamación cutánea (AP perirreservorio: ADCp, PSA 0,14).
            - Ingreso debido a FRA por UPO derecha a los 2 meses, con Cr 6,3 y Urea 133, necesitando catéter de NPC derecha. Se objetivó progresión de metástasis óseas.
            - Deterioro progresivo de estado general, 4 meses tras diagnóstico de metástasis subcutáneas."

EJEMPLO 2:
- Caso clínico original:
    Se trata de un <HUMAN>hombre</HUMAN> de <Age>67 años</Age> que inició su padecimiento en <Date>marzo de 2016</Date> con <SINTOMA>disuria</SINTOMA>, <SINTOMA>nicturia</SINTOMA>, <SINTOMA>tenesmo vesical</SINTOMA>, <SINTOMA>polaquiuria</SINTOMA> y <SINTOMA>goteo terminal</SINTOMA>. A la <PROCEDIMIENTO>exploración física</PROCEDIMIENTO>, se encontró al <PROCEDIMIENTO>tacto rectal</PROCEDIMIENTO> la <SINTOMA>próstata aumentada de tamaño</SINTOMA> (grado II), de consistencia normal, <Neg_cue>sin</Neg_cue> datos sospechosos de malignidad. Se solicitó <PROTEINAS>antígeno prostático específico</PROTEINAS>, cuyo resultado fue de <PSA>7,9 ng/ml</PSA> con 14,6% de <PROTEINAS>antígeno libre</PROTEINAS>, por lo que se le tomaron <PROCEDIMIENTO>biopsias de próstata guiadas por ultrasonido</PROCEDIMIENTO> en <Date>abril de 2017.</Date> El <PROCEDIMIENTO>diagnóstico anatomopatológico</PROCEDIMIENTO> de las <PROCEDIMIENTO>biopsias de próstata</PROCEDIMIENTO> fue de <ENFERMEDAD>adenocarcinoma acinar</ENFERMEDAD> con suma <GLEASON>Gleason 5 + 5 = 10</GLEASON>, bilateral.
    La <PROCEDIMIENTO>gammagrafía</PROCEDIMIENTO> ósea <Neg_cue>no</Neg_cue> mostró <Negated><ENFERMEDAD>metástasis óseas</Negated></ENFERMEDAD>, por lo que se decidió <PROCEDIMIENTO>manejo quirúrgico</PROCEDIMIENTO> con <PROCEDIMIENTO>prostatectomía radical retropúbica</PROCEDIMIENTO> con <PROCEDIMIENTO>linfadenectomía pélvica bilateral extendida</PROCEDIMIENTO>. La glándula prostática pesó 99 g y midió 6,8 × 5,0 × 4,9 cm, era piriforme, de superficie anfractuosa y de color pardo. Al corte, era de consistencia firme, sólida y de color café.
    Los <PROCEDIMIENTO>cortes histológicos</PROCEDIMIENTO> demostraron que el 90% de la <ENFERMEDAD>lesión</ENFERMEDAD> era <ENFERMEDAD>malacoplaquia</ENFERMEDAD> con un foco de <ENFERMEDAD>adenocarcinoma</ENFERMEDAD> de 5 mm. A mayor aumento, las glándulas tumorales muestran patrón de <GLEASON>Gleason 3 + 4</GLEASON>, similares a las vistas en las <PROCEDIMIENTO>biopsias transrectales</PROCEDIMIENTO>. Asimismo, el estroma mostró acumulaciones de histiocitos fusiformes, acompañados de infiltrado inflamatorio mixto, con linfocitos y leucocitos polimorfonucleares. Focalmente y a mayor aumento, el citoplasma de algunos macrófagos es amplio y eosinofílico (células de von Hansemann) con calcificaciones pequeñas y redondas, que corresponden a cuerpos de Michaelis-Gutmann y que resaltan con la <PROCEDIMIENTO>tinción de von</PROCEDIMIENTO> Kossa. Con <PROCEDIMIENTO>estudio de inmunohistoquímica</PROCEDIMIENTO>, los histiocitos que conforman la malacoplaquia son positivos a <PROTEINAS>CD68</PROTEINAS> y a <PROTEINAS>CD163</PROTEINAS>.
    El <PROTEINAS>antígeno prostático específico</PROTEINAS>, <Duration>2 meses</Duration> después de la <PROCEDIMIENTO>cirugía</PROCEDIMIENTO>, resultó en <PSA>0,0 ng/ml</PSA> y así se ha mantenido hasta su última visita en <Date>febrero del 2019.</Date>
- Texto generado (estilo historia clínica de paciente de cáncer de próstata):
    Varón de 67 años. 
    - Consulta en marzo de 2016 por síndrome miccional, con TR volumen II sin sospecha de malignidad. PSA de 7,9 (PSA libre 14,6%).  
    - BTRP en abril de 2017 con AP de ADC acinar de próstata Gleason 10 (5+5) bilateral. GGO sin metástasis óseas. Se decide prostatectomía radical retropúbica + LFD pélvica bilateral extendida. 
    - Resultados AP pieza quirúrgica: 90% de la lesión compatible con malacoplaquia con un foco de adenocarcinoma de 5 mm Gleason 7(3 + 4) ISUP 2. 
    - En remisión desde el segundo mes post qx y hasta su última visita en 02/2019, con PSA 0.  
Devuelve la salida EXACTAMENTE entre estos delimitadores, sin añadir nada fuera:
 ---INICIO--- 
 (aquí la nota) 
 ---FIN---
CASO CLÍNICO ORIGINAL:
{CASE_TEXT}

"""


@dataclass
class RunResult:
    input_path: str
    output_path: str
    status: str
    error: str
    latency_s: float
    input_chars: int
    output_chars: int
    model: str
    prompt_hash: str
    response_hash: str


def ensure_dir(path: str) -> None:
    """
    Creates a directory if missing.

    Arguments: path (str): directory path.
    Return: None
    """
    os.makedirs(path, exist_ok=True)


def iter_txt_files(root_dir: str) -> List[str]:
    """
    Recursively finds .txt files under root_dir.

    Arguments: root_dir (str): root directory.
    Return: List[str]: list of .txt file paths.
    """
    out: List[str] = []
    for base, _, files in os.walk(root_dir):
        for fn in files:
            if fn.lower().endswith(".txt"):
                out.append(os.path.join(base, fn))
    return sorted(out)


def sha1_text(s: str) -> str:
    """
    Computes SHA1 of a string.

    Arguments: s (str): input string.
    Return: str: sha1 hex digest.
    """
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def safe_relpath(path: str, start: str) -> str:
    """
    Computes a stable relative path; falls back to basename if relpath fails.

    Arguments:
        path (str): absolute path.
        start (str): base directory.
    Return:
        str: relative path.
    """
    try:
        rel = os.path.relpath(path, start)
        if rel.startswith(".."):
            return os.path.basename(path)
        return rel
    except Exception:
        return os.path.basename(path)


def write_text_atomic(path: str, text: str) -> None:
    """
    Writes text to path atomically.

    Arguments:
        path (str): output file path.
        text (str): content.
    Return: None
    """
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)


def load_already_generated(out_dir: str, input_dir: str, all_input_paths: List[str]) -> Set[str]:
    """
    Returns the set of input paths whose corresponding output file already exists
    and is non-empty, so they can be skipped on resume.

    Arguments:
        out_dir (str): root output directory.
        input_dir (str): root input directory.
        all_input_paths (List[str]): all discovered input file paths.
    Return:
        Set[str]: set of input paths to skip.
    """
    done: Set[str] = set()
    for in_path in all_input_paths:
        rel = safe_relpath(in_path, input_dir)
        out_path = os.path.join(out_dir, os.path.splitext(rel)[0] + ".txt")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            done.add(in_path)
    return done


def call_ollama_chat_raw(
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout_s: int,
) -> Dict:
    """
    Calls Ollama /api/chat (non-stream) with think=True at request root level
    and no stop tokens, allowing the model to generate freely up to num_predict.

    Arguments:
        base_url (str): Ollama base URL.
        model (str): model name.
        system_prompt (str): system role prompt.
        user_prompt (str): user prompt.
        timeout_s (int): request timeout seconds.
    Return:
        Dict: JSON response from Ollama.
    """
    url = base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {
            "num_predict": NUM_PREDICT,
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "repeat_penalty": REPEAT_PENALTY,
            "seed": SEED,
        },
    }
    r = requests.post(url, json=payload, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        raise ValueError("Ollama returned non-dict JSON.")
    return data


def extract_chat_content(raw_json: Dict) -> Tuple[str, str]:
    """
    Extracts assistant message content and thinking from an Ollama /api/chat response.

    Arguments:
        raw_json (Dict): Ollama JSON response.
    Return:
        Tuple[str, str]: (content, thinking), both trimmed, may be empty string.
    """
    msg = raw_json.get("message", {})
    if not isinstance(msg, dict):
        return "", ""
    content = msg.get("content", "") or ""
    thinking = msg.get("thinking", "") or ""
    return content.strip(), thinking.strip()


def strip_think_blocks(text: str) -> str:
    """
    Removes <think>...</think> blocks from model output.

    Arguments:
        text (str): raw model output.
    Return:
        str: output with think blocks removed.
    """
    return re.sub(r"(?is)<think>.*?</think>", "", text).strip()


def extract_delimited_note(text: str) -> str:
    """
    Extracts the note between ---INICIO--- and ---FIN--- delimiters.

    Arguments:
        text (str): model output text.
    Return:
        str: extracted note or empty string if delimiters are missing.
    """
    m = re.search(r"(?s)---INICIO---\s*(.*?)\s*---FIN---", text)
    if not m:
        return ""
    return m.group(1).strip()


def resolve_note(content: str, thinking: str) -> Tuple[str, str]:
    """
    Resolves the final note from content and thinking fields using a cascade strategy.

    Arguments:
        content (str): message.content from Ollama response.
        thinking (str): message.thinking from Ollama response.
    Return:
        Tuple[str, str]: (note text, resolution strategy label).
    """
    if content:
        note = extract_delimited_note(content)
        if note:
            return note, "delimited_content"

        content_stripped = strip_think_blocks(content)
        note = extract_delimited_note(content_stripped)
        if note:
            return note, "delimited_content_stripped"

        if len(content_stripped) >= MIN_OUTPUT_CHARS:
            return content_stripped, "full_content_stripped"

    if thinking:
        note = extract_delimited_note(thinking)
        if note:
            return note, "delimited_thinking"

    return "", "no_output"


def generate_note(case_text: str) -> Tuple[str, Dict, str]:
    """
    Generates an evolutivo note for a given case.

    Arguments:
        case_text (str): source case text.
    Return:
        Tuple[str, Dict, str]:
            - note text (may be empty),
            - raw JSON response,
            - status string ("ok", "empty_output").
    """
    user_prompt = USER_PROMPT_TEMPLATE.format(CASE_TEXT=case_text)
    raw = call_ollama_chat_raw(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        timeout_s=REQUEST_TIMEOUT_SECONDS,
    )

    content, thinking = extract_chat_content(raw)
    print(f"    content_chars={len(content)} thinking_chars={len(thinking)} done_reason={raw.get('done_reason')} eval_count={raw.get('eval_count')}")

    note, strategy = resolve_note(content, thinking)
    print(f"    resolution_strategy={strategy}")

    if note:
        return note, raw, "ok"
    return "", raw, "empty_output"


def write_log_csv(rows: List[RunResult], csv_path: str) -> None:
    """
    Writes a CSV log of runs.

    Arguments:
        rows (List[RunResult]): run results.
        csv_path (str): output CSV path.
    Return: None
    """
    header = [
        "input_path",
        "output_path",
        "status",
        "error",
        "latency_s",
        "input_chars",
        "output_chars",
        "model",
        "prompt_hash",
        "response_hash",
    ]
    ensure_dir(os.path.dirname(csv_path) or ".")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(",".join(header) + "\n")
        for rr in rows:
            vals = [
                rr.input_path.replace('"', '""'),
                rr.output_path.replace('"', '""'),
                rr.status.replace('"', '""'),
                rr.error.replace('"', '""'),
                f"{rr.latency_s:.3f}",
                str(rr.input_chars),
                str(rr.output_chars),
                rr.model,
                rr.prompt_hash,
                rr.response_hash,
            ]
            f.write(",".join(f'"{v}"' for v in vals) + "\n")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Arguments: None
    Return: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="Generate Spanish clinical progress notes from .txt cases using Ollama."
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing input .txt files.")
    parser.add_argument("--out-dir", required=True, help="Directory where generated notes will be written.")
    parser.add_argument(
        "--log-csv",
        default="",
        help="Optional CSV log path. Defaults to <out-dir>/log.csv.",
    )
    parser.add_argument("--model", default=OLLAMA_MODEL, help="Ollama model name.")
    parser.add_argument(
        "--base-url",
        default=OLLAMA_BASE_URL,
        help="Ollama server base URL.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=int,
        default=REQUEST_TIMEOUT_SECONDS,
        help="HTTP timeout for Ollama requests.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Processes all .txt cases under INPUT_DIR and writes generated evolutivos to OUT_DIR.
    Already-generated outputs (non-empty file at expected output path) are skipped.

    Arguments: None
    Return: None
    """
    global INPUT_DIR, OUT_DIR, LOG_CSV_PATH, OLLAMA_MODEL, OLLAMA_BASE_URL, REQUEST_TIMEOUT_SECONDS

    args = parse_args()
    INPUT_DIR = args.input_dir
    OUT_DIR = args.out_dir
    LOG_CSV_PATH = args.log_csv or os.path.join(OUT_DIR, DEFAULT_LOG_CSV_NAME)
    OLLAMA_MODEL = args.model
    OLLAMA_BASE_URL = args.base_url
    REQUEST_TIMEOUT_SECONDS = args.request_timeout_seconds

    ensure_dir(OUT_DIR)
    err_dir = os.path.join(OUT_DIR, "_errors")
    ensure_dir(err_dir)

    files = iter_txt_files(INPUT_DIR)
    already_done = load_already_generated(OUT_DIR, INPUT_DIR, files)
    pending = [p for p in files if p not in already_done]

    print(f"INPUT_DIR:       {INPUT_DIR}")
    print(f"OUT_DIR:         {OUT_DIR}")
    print(f"MODEL:           {OLLAMA_MODEL}")
    print(f"Total found:     {len(files)}")
    print(f"Already done:    {len(already_done)}")
    print(f"Pending:         {len(pending)}")

    results: List[RunResult] = []

    for i, in_path in enumerate(pending, 1):
        rel = safe_relpath(in_path, INPUT_DIR)
        out_path = os.path.join(OUT_DIR, rel)
        out_path = os.path.splitext(out_path)[0] + ".txt"
        ensure_dir(os.path.dirname(out_path))

        with open(in_path, "r", encoding="utf-8", errors="replace") as f:
            case_text = f.read().strip()

        print(f"\n[{i}/{len(pending)}] IN:  {in_path}")
        print(f"[{i}/{len(pending)}] OUT: {out_path}")
        print(f"[{i}/{len(pending)}] in_chars={len(case_text)}")

        if not case_text:
            results.append(
                RunResult(
                    input_path=in_path,
                    output_path=out_path,
                    status="empty_input",
                    error="Input .txt is empty after strip().",
                    latency_s=0.0,
                    input_chars=0,
                    output_chars=0,
                    model=OLLAMA_MODEL,
                    prompt_hash="",
                    response_hash="",
                )
            )
            print(f"[{i}/{len(pending)}] SKIP: empty input")
            continue

        prompt_hash = sha1_text(case_text + SYSTEM_PROMPT)

        t0 = time.time()
        status = "ok"
        err = ""
        output = ""
        raw_json: Optional[Dict] = None

        try:
            output, raw_json, status = generate_note(case_text)
        except Exception as e:
            status = "error"
            err = str(e)

        latency = time.time() - t0

        if status == "ok":
            write_text_atomic(out_path, output.strip() + "\n")
            print(f"[{i}/{len(pending)}] WROTE: out_chars={len(output)} latency={latency:.2f}s")
        else:
            base = os.path.splitext(os.path.basename(in_path))[0]
            err_path = os.path.join(err_dir, f"{base}.{status}.error.json")
            dump = {
                "input_path": in_path,
                "output_path": out_path,
                "status": status,
                "error": err,
                "model": OLLAMA_MODEL,
                "latency_s": latency,
                "input_chars": len(case_text),
                "raw_json": raw_json,
            }
            write_text_atomic(err_path, json.dumps(dump, ensure_ascii=False, indent=2) + "\n")
            print(f"[{i}/{len(pending)}] {status.upper()}: {err if err else 'no_output'}")
            print(f"[{i}/{len(pending)}] Saved error: {err_path}")

        results.append(
            RunResult(
                input_path=in_path,
                output_path=out_path,
                status=status,
                error=err,
                latency_s=latency,
                input_chars=len(case_text),
                output_chars=len(output),
                model=OLLAMA_MODEL,
                prompt_hash=prompt_hash,
                response_hash=sha1_text(output) if output else "",
            )
        )

    write_log_csv(results, LOG_CSV_PATH)
    print(f"\nWrote log: {LOG_CSV_PATH}")


if __name__ == "__main__":
    main()
