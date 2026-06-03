"""
This script evaluates generated Spanish clinical progress notes against their
original clinical cases using the Anthropic Messages API.

The script is configured for prompt caching with Claude Sonnet 4.6:
- The system prompt is marked with an explicit cache_control breakpoint.
- Only the system prompt is cached; user messages (clinical cases) are not.
- Stores compact JSONL output: {"file": "1.txt", "evaluation": {...}}
- Stores parse and API errors separately.
- Prints cache usage fields per request:
    input_tokens, cache_creation_input_tokens, cache_read_input_tokens, output_tokens
- Supports resuming: already-evaluated files are skipped on restart.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
import requests


ORIGINAL_CASES_DIR = ""
GENERATED_NOTES_ROOT_DIR = ""
OUTPUT_DIR = ""

RESULTS_JSONL_FILENAME = ("results_CSonnet46_GSAMan_JVM.jsonl")
RESULTS_CSV_FILENAME = "results.csv"
PARSE_ERRORS_JSONL_FILENAME = "parse_errors.jsonl"
DEBUG_JSONL_FILENAME = "debug.jsonl"

RESULTS_JSONL_PATH = os.path.join(OUTPUT_DIR, RESULTS_JSONL_FILENAME)
RESULTS_CSV_PATH = os.path.join(OUTPUT_DIR, RESULTS_CSV_FILENAME)
PARSE_ERRORS_JSONL_PATH = os.path.join(OUTPUT_DIR, PARSE_ERRORS_JSONL_FILENAME)
DEBUG_JSONL_PATH = os.path.join(OUTPUT_DIR, DEBUG_JSONL_FILENAME)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_COUNT_TOKENS_URL = "https://api.anthropic.com/v1/messages/count_tokens"
ANTHROPIC_VERSION = "2023-06-01"
ANTHROPIC_BETA = "prompt-caching-2024-07-31"

CLAUDE_MODEL = "claude-sonnet-4-6"
CLAUDE_MAX_TOKENS = 1200

REQUEST_TIMEOUT_SECONDS = 600
REQUEST_SLEEP_SECONDS = 0.5
MAX_RETRIES = 3

MAX_ORIGINAL_CHARS = 12000
MAX_GENERATED_CHARS = 8000
WRITE_DEBUG_JSONL = True

SYSTEM_PROMPT = """
Actúa como un médico especialista en Urología u Oncología que evalúa notas evolutivas clínicas sintéticas con amplia experiencia en documentación clínica hospitalaria.

En tu tarea, debes comparar:
- CASO CLÍNICO ORIGINAL (CC)
- NOTA GENERADA (NG)

Evalúa la calidad clínica de la NG respecto al CC y como documento clínico independiente.

CRITERIOS DE EVALUACIÓN

1. Actualizada (temporalmente correcta)
Evalúa si la NG sigue el mismo orden lógico y temporal del CC en cuanto a evolución clínica, eventos, tratamientos y procedimientos.

2. Precisa (datos NG → datos CC)
Evalúa si todos los datos presentes en la NG aparecen en el CC y son clínicamente correctos. Penaliza alucinaciones o invención de datos.

3. Exhaustiva (datos CC → datos NG)
Evalúa estrictamente la información y los datos clínicamente relevantes desde el punto de vista del manejo médico urológico de la patología de cáncer de próstata: diagnóstico, tratamiento, plan de actuación, estadificación, comorbilidades, etc. de la patología de cáncer de próstata.
4. Útil
Independientemente del CC, evalúa si la NG contiene información clínicamente relevante para la toma de decisiones.

5. Organizada
Evalúa la estructura lógica y la organización del documento.

6. Comprensible
Evalúa la claridad del lenguaje y el uso adecuado de terminología médica.

7. Concisa
Evalúa si la NG evita redundancias innecesarias.

8. Sintetizada
Evalúa si la información clínica está integrada de forma coherente.

9. Consistencia interna
Evalúa si existen contradicciones clínicas dentro de la NG.

PREGUNTAS ADICIONALES

Existe algún error de seguridad grave:
errores clínicos potencialmente peligrosos (dosis incorrectas, errores de estadiaje, procedimientos inseguros, etc.).
Respuesta: SI/NO

Existe alguna contradicción clínica insalvable:
contradicciones que hacen la nota clínicamente incoherente.
Respuesta: SI/NO

Esta nota podría haber sido generada por un humano:
Respuesta: SI/NO

REGLAS IMPORTANTES
- No penalices cambios de estilo o reordenación si la información clínica es fiel.
- Penaliza fuertemente alucinaciones clínicas.
- Penaliza omisiones relevantes.
- Si detectas alucinaciones importantes, "precisa" <= 2.
- Si detectas contradicciones internas importantes, "consistencia_interna" <= 2.
- Cuando la puntuación de un criterio sea 5 (Excelente), el campo "justification" debe ser una cadena vacía "". No escribas ningún texto en ese campo.

ESCALA DE EVALUACIÓN

Las puntuaciones deben asignarse utilizando estrictamente la siguiente escala:

5 (Excelente)
Cumplimiento total del criterio evaluado. No se detectan errores clínicos, inconsistencias ni problemas de interpretación.
→ justification: ""

4 (Adecuado)
Existen errores menores, estilísticos o de redacción que no afectan a la seguridad clínica ni a la correcta interpretación del caso.

3 (Regular)
Existen errores moderados o limitaciones que deberían corregirse, pero el significado clínico general se mantiene y la nota sigue siendo interpretable.

2 (Deficiente)
Existen errores significativos, omisiones relevantes o problemas de coherencia que requerirían una reescritura parcial de la nota para que sea clínicamente correcta.

1 (Inaceptable)
Existe un error crítico de seguridad, información clínica falsa peligrosa, contradicciones graves o incoherencia que invalidan la nota.

FORMATO DE RESPUESTA

Devuelve exclusivamente un JSON válido con esta estructura exacta:

{
  "actualizada_temporalmente_correcta": {"score": int, "justification": string},
  "precisa": {"score": int, "justification": string},
  "exhaustiva": {"score": int, "justification": string},
  "util": {"score": int, "justification": string},
  "organizada": {"score": int, "justification": string},
  "comprensible": {"score": int, "justification": string},
  "concisa": {"score": int, "justification": string},
  "sintetizada": {"score": int, "justification": string},
  "consistencia_interna": {"score": int, "justification": string},
  "error_seguridad_grave": boolean,
  "contradicciones_clinicas_insalvables": boolean,
  "human_generated": boolean
}

No incluyas texto fuera del JSON.
Cuando score sea 5, justification debe ser "".
""".strip()

USER_PROMPT_TEMPLATE = """
Evalúa la siguiente nota clínica.

CASO CLÍNICO ORIGINAL:
{ORIGINAL_CASE}

NOTA GENERADA:
{GENERATED_NOTE}
""".strip()

EXPECTED_KEYS = [
    "actualizada_temporalmente_correcta",
    "precisa",
    "exhaustiva",
    "util",
    "organizada",
    "comprensible",
    "concisa",
    "sintetizada",
    "consistencia_interna",
    "error_seguridad_grave",
    "contradicciones_clinicas_insalvables",
    "human_generated",
]


def rename_files(directory):
    """
    Rename files from 'xt.txt' to 'x.txt'.

    Arguments:
        directory (str): Path to the directory containing files.
    Return:
        None
    """
    for filename in os.listdir(directory):
        if filename.endswith("t.txt"):
            new_name = filename.replace("t.txt", ".txt")
            old_path = os.path.join(directory, filename)
            new_path = os.path.join(directory, new_name)

            print(f"Renaming: {filename} -> {new_name}")
            os.rename(old_path, new_path)


def ensure_dir(path: str) -> None:
    """
    Create the target directory if it does not already exist.
    Arguments: path.
    Return: None.
    """
    os.makedirs(path, exist_ok=True)


def reset_transient_files(paths: List[str]) -> None:
    """
    Remove only transient output files (errors, debug) so the current run starts
    clean without discarding previously accumulated results.
    Arguments: paths.
    Return: None.
    """
    for path in paths:
        if os.path.exists(path):
            os.remove(path)


def load_already_evaluated_files(results_jsonl_path: str) -> Set[str]:
    """
    Read the existing results JSONL and return the set of filenames already evaluated.
    Lines that cannot be parsed are silently skipped.
    Arguments: results_jsonl_path.
    Return: Set of evaluated filenames.
    """
    evaluated: Set[str] = set()
    if not os.path.exists(results_jsonl_path):
        return evaluated

    with open(results_jsonl_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if isinstance(record, dict) and isinstance(record.get("file"), str):
                    evaluated.add(record["file"])
            except Exception:
                continue

    return evaluated


def load_existing_flat_rows(results_csv_path: str) -> List[Dict[str, Any]]:
    """
    Load previously saved flat rows from the CSV snapshot so the in-memory
    accumulator starts populated on resume.
    Arguments: results_csv_path.
    Return: List of row dictionaries.
    """
    if not os.path.exists(results_csv_path):
        return []

    try:
        df = pd.read_csv(results_csv_path, encoding="utf-8")
        return df.to_dict(orient="records")
    except Exception:
        return []


def read_text(path: str) -> str:
    """
    Read a text file using UTF-8 and replacement for invalid characters.
    Arguments: path.
    Return: File content as string.
    """
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read().strip()


def truncate_text(text: str, max_chars: int) -> str:
    """
    Truncate text to a maximum number of characters.
    Arguments: text, max_chars.
    Return: Truncated text.
    """
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n[TRUNCATED]"


def get_default_strategy_name(root_dir: str) -> str:
    """
    Derive a default strategy name from the generated notes root directory.
    Arguments: root_dir.
    Return: Strategy name.
    """
    return os.path.basename(os.path.normpath(root_dir))


def is_valid_generated_filename(filename: str) -> bool:
    """
    Filter candidate generated note files.
    Arguments: filename.
    Return: True if the file is a candidate generated note.
    """
    if not filename.lower().endswith(".txt"):
        return False

    lowered = filename.lower()
    banned_terms = ["report", "analysis", "stats", "summary", "metric", "log", "readme"]
    return not any(term in lowered for term in banned_terms)


def resolve_original_filename(generated_filename: str) -> str:
    """
    Resolve the expected original-case filename from a generated-note filename.
    Arguments: generated_filename.
    Return: Expected original-case filename.
    """
    name = generated_filename.strip()
    if name.lower().endswith("t.txt"):
        return name[:-14] + ".txt"
    return name


def iter_generated_note_files(
    root_dir: str,
    original_cases_dir: str,
) -> List[Tuple[str, str, str, str]]:
    """
    Collect generated note files from either a flat directory or strategy subdirectories.
    Arguments: root_dir, original_cases_dir.
    Return: List of tuples with strategy name, generated filename, original filename and full file path.
    """
    rows: List[Tuple[str, str, str, str]] = []

    if not os.path.isdir(root_dir):
        return rows

    original_filenames = {
        fn for fn in os.listdir(original_cases_dir)
        if fn.lower().endswith(".txt")
    }

    default_strategy = get_default_strategy_name(root_dir)

    for fn in sorted(os.listdir(root_dir)):
        full_path = os.path.join(root_dir, fn)
        if os.path.isfile(full_path) and is_valid_generated_filename(fn):
            original_fn = resolve_original_filename(fn)
            if original_fn in original_filenames:
                rows.append((default_strategy, fn, original_fn, full_path))

    subdirs = [
        d for d in sorted(os.listdir(root_dir))
        if os.path.isdir(os.path.join(root_dir, d))
    ]

    for strategy_name in subdirs:
        strategy_path = os.path.join(root_dir, strategy_name)
        for base, _, files in os.walk(strategy_path):
            for fn in sorted(files):
                if not is_valid_generated_filename(fn):
                    continue
                full_path = os.path.join(base, fn)
                original_fn = resolve_original_filename(fn)
                if original_fn in original_filenames:
                    rows.append((strategy_name, fn, original_fn, full_path))

    unique_rows: List[Tuple[str, str, str, str]] = []
    seen = set()
    for row in rows:
        if row not in seen:
            seen.add(row)
            unique_rows.append(row)

    return unique_rows


def build_user_prompt(original_case: str, generated_note: str) -> str:
    """
    Build the dynamic user prompt containing the original case and generated note.
    Arguments: original_case, generated_note.
    Return: Prompt string.
    """
    return USER_PROMPT_TEMPLATE.format(
        ORIGINAL_CASE=truncate_text(original_case, MAX_ORIGINAL_CHARS),
        GENERATED_NOTE=truncate_text(generated_note, MAX_GENERATED_CHARS),
    )


def get_anthropic_headers() -> Dict[str, str]:
    """
    Build request headers for the Anthropic API without the prompt caching beta header.
    Arguments: None.
    Return: Headers dictionary.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise ValueError("ANTHROPIC_API_KEY is not set in the environment.")

    return {
        "x-api-key": api_key,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }


def build_messages_payload(user_prompt: str, max_tokens: int) -> Dict[str, Any]:
    """
    Build a Messages API payload without prompt caching.
    Arguments: user_prompt, max_tokens.
    Return: Payload dictionary.
    """
    return {
        "model": CLAUDE_MODEL,
        "max_tokens": max_tokens,
        "system": SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
        "temperature": 0,
    }


def count_tokens_for_prompt(user_prompt: str) -> Optional[int]:
    """
    Count estimated input tokens for a single request using the count_tokens endpoint.
    Arguments: user_prompt.
    Return: Input token estimate or None on failure.
    """
    payload = build_messages_payload(user_prompt=user_prompt, max_tokens=CLAUDE_MAX_TOKENS)

    try:
        response = requests.post(
            ANTHROPIC_COUNT_TOKENS_URL,
            headers=get_anthropic_headers(),
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        data = response.json()
        if response.ok and isinstance(data, dict) and isinstance(data.get("input_tokens"), int):
            return data["input_tokens"]
    except Exception:
        return None

    return None


def call_claude_messages_api(user_prompt: str) -> Dict[str, Any]:
    """
    Send a single request to the Anthropic Messages API.
    Arguments: user_prompt.
    Return: Parsed response dictionary.
    """
    payload = build_messages_payload(user_prompt=user_prompt, max_tokens=CLAUDE_MAX_TOKENS)

    response = requests.post(
        ANTHROPIC_API_URL,
        headers=get_anthropic_headers(),
        json=payload,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )

    try:
        data = response.json()
    except Exception:
        data = {"raw_text": response.text}

    if not response.ok:
        raise RuntimeError(
            f"Anthropic API error {response.status_code}: "
            f"{json.dumps(data, ensure_ascii=False)}"
        )

    if not isinstance(data, dict):
        raise ValueError("Claude API returned a non-dict response.")

    return data


def extract_claude_message_text(raw_response: Dict[str, Any]) -> str:
    """
    Extract concatenated text content from an Anthropic Messages API response.
    Arguments: raw_response.
    Return: Assistant text content as string.
    """
    content = raw_response.get("content", [])
    if not isinstance(content, list):
        return ""

    parts: List[str] = []
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text", "")
            if isinstance(text, str):
                parts.append(text)

    return "\n".join(parts).strip()


def extract_usage_info(raw_response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract token usage and cache-related fields from an Anthropic response.
    Arguments: raw_response.
    Return: Usage dictionary with input_tokens, cache_creation_input_tokens,
            cache_read_input_tokens and output_tokens.
    """
    usage = raw_response.get("usage", {})
    if not isinstance(usage, dict):
        return {}

    return {
        "input_tokens": usage.get("input_tokens"),
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
        "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
        "output_tokens": usage.get("output_tokens"),
    }


def try_parse_json(text: str) -> Optional[Any]:
    """
    Attempt to parse a string directly as JSON.
    Arguments: text.
    Return: Parsed Python object or None.
    """
    try:
        return json.loads(text)
    except Exception:
        return None


def extract_first_json_block(text: str) -> Optional[Any]:
    """
    Attempt to recover the first valid JSON object or array embedded in free text.
    Arguments: text.
    Return: Parsed Python object or None.
    """
    parsed = try_parse_json(text)
    if parsed is not None:
        return parsed

    for candidate in re.findall(r"\{.*\}", text, flags=re.DOTALL):
        parsed = try_parse_json(candidate)
        if parsed is not None:
            return parsed

    for candidate in re.findall(r"\[.*\]", text, flags=re.DOTALL):
        parsed = try_parse_json(candidate)
        if parsed is not None:
            return parsed

    return None


def parse_model_output(raw_text: str) -> Any:
    """
    Parse the raw model output into a Python structure.
    Arguments: raw_text.
    Return: Parsed Python object.
    """
    parsed = try_parse_json(raw_text)
    if parsed is not None:
        return parsed

    recovered = extract_first_json_block(raw_text)
    if recovered is not None:
        return recovered

    raise ValueError("Could not parse model output as JSON.")


def normalize_bool(value: Any) -> bool:
    """
    Normalize a value into boolean form accepting common string representations.
    Arguments: value.
    Return: Normalized boolean.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "si", "sí"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    if isinstance(value, (int, float)):
        return bool(value)
    return False


def normalize_result(result: Any) -> Dict[str, Any]:
    """
    Normalize parsed model output into a dictionary.
    Arguments: result.
    Return: Normalized result dictionary.
    """
    if isinstance(result, dict):
        return result
    if isinstance(result, list):
        if len(result) == 0:
            return {"parse_error": "empty_list_response"}
        if len(result) == 1 and isinstance(result[0], dict):
            return result[0]
        return {"parse_error": "list_response", "raw_list": result}
    return {
        "parse_error": f"unexpected_type_{type(result).__name__}",
        "raw_value": result,
    }


def validate_result_structure(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and normalize the evaluation structure against the expected schema.
    Arguments: result.
    Return: Validated result dictionary.
    """
    validated: Dict[str, Any] = {}
    boolean_keys = {
        "error_seguridad_grave",
        "contradicciones_clinicas_insalvables",
        "human_generated",
    }

    for key in EXPECTED_KEYS:
        if key in boolean_keys:
            validated[key] = normalize_bool(result.get(key, False))
            continue

        value = result.get(key, {})
        if not isinstance(value, dict):
            value = {}

        score = value.get("score")
        justification = value.get("justification", "")

        if not isinstance(score, int) or score < 1 or score > 5:
            score = None
        if not isinstance(justification, str):
            justification = ""

        validated[key] = {
            "score": score,
            "justification": justification.strip(),
        }

    if "parse_error" in result:
        validated["parse_error"] = result["parse_error"]

    return validated


def append_jsonl(path: str, item: Dict[str, Any]) -> None:
    """
    Append one JSON object as a JSONL line with immediate flush and fsync.
    Arguments: path, item.
    Return: None.
    """
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def write_csv_snapshot(rows: List[Dict[str, Any]], path: str) -> None:
    """
    Write accumulated evaluation rows to CSV.
    Arguments: rows, path.
    Return: None.
    """
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")


def flatten_result(
    file_name: str,
    strategy: str,
    result: Dict[str, Any],
    input_tokens_estimate: Optional[int],
    usage_info: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Flatten the structured evaluation dictionary into a single CSV-compatible row.
    Arguments: file_name, strategy, result, input_tokens_estimate, usage_info.
    Return: Flattened row dictionary.
    """
    row: Dict[str, Any] = {
        "file": file_name,
        "strategy": strategy,
        "input_tokens_estimate": input_tokens_estimate,
        "input_tokens": usage_info.get("input_tokens"),
        "cache_creation_input_tokens": usage_info.get("cache_creation_input_tokens"),
        "cache_read_input_tokens": usage_info.get("cache_read_input_tokens"),
        "output_tokens": usage_info.get("output_tokens"),
    }

    for key, value in result.items():
        if isinstance(value, dict) and "score" in value and "justification" in value:
            row[f"{key}_score"] = value.get("score")
            row[f"{key}_justification"] = value.get("justification")
        else:
            row[key] = value

    return row


def evaluate_pair(
    original_case: str,
    generated_note: str,
) -> Tuple[Dict[str, Any], str, Dict[str, Any], Optional[int], Dict[str, Any]]:
    """
    Evaluate one original/generated pair through the Anthropic Messages API with retries.
    Arguments: original_case, generated_note.
    Return: Tuple of validated result, raw text, raw response, token estimate and usage info.
    """
    user_prompt = build_user_prompt(original_case, generated_note)
    input_tokens_estimate = count_tokens_for_prompt(user_prompt)
    last_error: Optional[Exception] = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw_response = call_claude_messages_api(user_prompt=user_prompt)
            raw_text = extract_claude_message_text(raw_response)
            usage_info = extract_usage_info(raw_response)
            parsed = parse_model_output(raw_text)
            normalized = normalize_result(parsed)
            validated = validate_result_structure(normalized)
            return validated, raw_text, raw_response, input_tokens_estimate, usage_info
        except Exception as e:
            last_error = e
            print(f"Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(REQUEST_SLEEP_SECONDS * attempt)
            else:
                raise RuntimeError(str(last_error)) from last_error

    raise RuntimeError("Unexpected retry flow reached.")


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Arguments: None.
    Return: argparse.Namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate generated Spanish clinical notes against original cases using the Anthropic Messages API."
    )
    parser.add_argument("--original-cases-dir", required=True, help="Directory with original-case .txt files.")
    parser.add_argument("--generated-notes-dir", required=True, help="Directory with generated-note .txt files.")
    parser.add_argument("--output-dir", required=True, help="Directory where evaluation outputs will be written.")
    parser.add_argument("--model", default=CLAUDE_MODEL, help="Anthropic model name.")
    parser.add_argument(
        "--request-timeout-seconds",
        type=int,
        default=REQUEST_TIMEOUT_SECONDS,
        help="HTTP timeout for Anthropic requests.",
    )
    parser.add_argument(
        "--rename-generated-files",
        action="store_true",
        help="Rename generated files ending in 't.txt' to '.txt' before evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    """
    Run the full batch evaluation sequentially with resume support.
    Files already present in the results JSONL are skipped without re-evaluating.
    Arguments: None.
    Return: None.
    """
    global ORIGINAL_CASES_DIR, GENERATED_NOTES_ROOT_DIR, OUTPUT_DIR
    global RESULTS_JSONL_PATH, RESULTS_CSV_PATH, PARSE_ERRORS_JSONL_PATH, DEBUG_JSONL_PATH
    global CLAUDE_MODEL, REQUEST_TIMEOUT_SECONDS

    args = parse_args()
    ORIGINAL_CASES_DIR = args.original_cases_dir
    GENERATED_NOTES_ROOT_DIR = args.generated_notes_dir
    OUTPUT_DIR = args.output_dir
    CLAUDE_MODEL = args.model
    REQUEST_TIMEOUT_SECONDS = args.request_timeout_seconds

    RESULTS_JSONL_PATH = os.path.join(OUTPUT_DIR, RESULTS_JSONL_FILENAME)
    RESULTS_CSV_PATH = os.path.join(OUTPUT_DIR, RESULTS_CSV_FILENAME)
    PARSE_ERRORS_JSONL_PATH = os.path.join(OUTPUT_DIR, PARSE_ERRORS_JSONL_FILENAME)
    DEBUG_JSONL_PATH = os.path.join(OUTPUT_DIR, DEBUG_JSONL_FILENAME)

    if args.rename_generated_files:
        rename_files(GENERATED_NOTES_ROOT_DIR)

    ensure_dir(OUTPUT_DIR)

    already_evaluated = load_already_evaluated_files(RESULTS_JSONL_PATH)
    flat_rows = load_existing_flat_rows(RESULTS_CSV_PATH)

    reset_transient_files([PARSE_ERRORS_JSONL_PATH, DEBUG_JSONL_PATH])

    note_files = iter_generated_note_files(
        root_dir=GENERATED_NOTES_ROOT_DIR,
        original_cases_dir=ORIGINAL_CASES_DIR,
    )

    pending = [
        entry for entry in note_files
        if entry[1] not in already_evaluated
    ]

    print(f"ORIGINAL_CASES_DIR:       {ORIGINAL_CASES_DIR}")
    print(f"GENERATED_NOTES_ROOT_DIR: {GENERATED_NOTES_ROOT_DIR}")
    print(f"OUTPUT_DIR:               {OUTPUT_DIR}")
    print(f"RESULTS_JSONL_PATH:       {RESULTS_JSONL_PATH}")
    print(f"RESULTS_CSV_PATH:         {RESULTS_CSV_PATH}")
    print(f"PARSE_ERRORS_JSONL_PATH:  {PARSE_ERRORS_JSONL_PATH}")
    print(f"DEBUG_JSONL_PATH:         {DEBUG_JSONL_PATH}")
    print(f"CLAUDE_MODEL:             {CLAUDE_MODEL}")
    print(f"Total notes found:        {len(note_files)}")
    print(f"Already evaluated:        {len(already_evaluated)}")
    print(f"Pending evaluation:       {len(pending)}")

    if len(pending) == 0:
        print("All notes already evaluated. Nothing to do.")
        return

    for idx, (strategy, generated_filename, original_filename, generated_path) in enumerate(pending, 1):
        original_path = os.path.join(ORIGINAL_CASES_DIR, original_filename)

        print(f"\n[{idx}/{len(pending)}] Evaluating {generated_filename}")
        print(f"Strategy: {strategy}")

        if not os.path.exists(original_path):
            append_jsonl(
                PARSE_ERRORS_JSONL_PATH,
                {"file": generated_filename, "error": "missing_original_case"},
            )
            flat_rows.append({
                "file": generated_filename,
                "strategy": strategy,
                "parse_error": "missing_original_case",
            })
            write_csv_snapshot(flat_rows, RESULTS_CSV_PATH)
            continue

        original_case = read_text(original_path)
        generated_note = read_text(generated_path)

        print(f"Original chars:  {len(original_case)}")
        print(f"Generated chars: {len(generated_note)}")

        try:
            result, raw_text, raw_response, input_tokens_estimate, usage_info = evaluate_pair(
                original_case=original_case,
                generated_note=generated_note,
            )

            print(f"Usage: {json.dumps(usage_info, ensure_ascii=False)}")

            append_jsonl(
                RESULTS_JSONL_PATH,
                {"file": generated_filename, "evaluation": result},
            )

            if WRITE_DEBUG_JSONL:
                append_jsonl(
                    DEBUG_JSONL_PATH,
                    {
                        "file": generated_filename,
                        "input_tokens_estimate": input_tokens_estimate,
                        "usage": usage_info,
                        "raw_text": raw_text,
                        "raw_response": raw_response,
                    },
                )

            if "parse_error" in result:
                append_jsonl(
                    PARSE_ERRORS_JSONL_PATH,
                    {
                        "file": generated_filename,
                        "parse_error": result["parse_error"],
                        "raw_text": raw_text,
                    },
                )

            flat_rows.append(
                flatten_result(
                    file_name=generated_filename,
                    strategy=strategy,
                    result=result,
                    input_tokens_estimate=input_tokens_estimate,
                    usage_info=usage_info,
                )
            )
            write_csv_snapshot(flat_rows, RESULTS_CSV_PATH)

        except Exception as e:
            append_jsonl(
                PARSE_ERRORS_JSONL_PATH,
                {"file": generated_filename, "error": str(e)},
            )
            flat_rows.append({
                "file": generated_filename,
                "strategy": strategy,
                "parse_error": str(e),
            })
            write_csv_snapshot(flat_rows, RESULTS_CSV_PATH)
            print(f"Saved parse error for: {generated_filename} [{strategy}]")

        time.sleep(REQUEST_SLEEP_SECONDS)

    print(f"\nSaved compact results to: {RESULTS_JSONL_PATH}")
    print(f"Saved CSV results to:     {RESULTS_CSV_PATH}")
    print(f"Saved parse errors to:    {PARSE_ERRORS_JSONL_PATH}")
    print(f"Saved debug results to:   {DEBUG_JSONL_PATH}")


if __name__ == "__main__":
    main()
