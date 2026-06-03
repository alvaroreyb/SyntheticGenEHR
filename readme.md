# Generation and Evaluation of Realistic Synthetic Clinical Progress Notes for Prostate Cancer Using Large Language Models

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Version](https://img.shields.io/badge/Version-v0.1.0-informational)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange)
![Domain](https://img.shields.io/badge/Domain-Spanish%20Clinical%20NLP-purple)
![Clinical Focus](https://img.shields.io/badge/Clinical%20Focus-Prostate%20Cancer-darkred)
![License](https://img.shields.io/badge/License-MIT-green)

This repository provides a research-oriented software framework for the **generation, evaluation, and clinical entity extraction of synthetic Spanish hospital-style progress notes** focused on prostate cancer.

The project integrates **Large Language Model-based clinical note generation**, **LLM-as-a-judge evaluation**, and **biomedical entity extraction** using BSC models, MedSpaNER models, and rule-based clinical pattern detection. It is intended for scientific experimentation, benchmarking, and reproducibility studies in Spanish clinical Natural Language Processing.

---

## Overview

This repository implements a complete experimental pipeline for transforming plain-text prostate cancer clinical cases into realistic synthetic clinical progress notes, evaluating their clinical quality, and extracting relevant biomedical entities for downstream analysis.

The pipeline supports:

- generation of Spanish hospital-style clinical progress notes;
- evaluation of generated notes against original clinical cases;
- extraction of clinically relevant entities from original and generated notes;
- export of structured annotations for statistical analysis and manual review.

The software is designed for research workflows involving clinical text generation, clinical summarisation, information preservation, and evaluation of synthetic clinical documentation.

---

## Main Objectives

The main objectives of this repository are:

1. **Synthetic clinical text generation**  
   Generate realistic Spanish hospital-style progress notes from source prostate cancer clinical cases.

2. **Clinical quality evaluation**  
   Assess generated notes against original clinical cases using a structured LLM-as-a-judge evaluation workflow.

3. **Clinical entity extraction**  
   Extract relevant clinical entities using neural biomedical NLP models and rule-based extraction methods.

4. **Reproducible experimentation**  
   Provide a modular and traceable framework for model comparison, evaluation, and downstream clinical NLP analysis.

---

## Pipeline Overview

The repository follows a modular workflow:

```text
Original clinical cases (.txt)
        |
        v
Synthetic progress-note generation
        |
        |-- OpenAI Responses API
        |-- Ollama-compatible local or remote models
        |
        v
Generated clinical progress notes (.txt)
        |
        v
LLM-as-a-judge clinical evaluation
        |
        v
Evaluation outputs (.jsonl, .csv, debug artifacts)
        |
        v
Clinical entity extraction
        |
        |-- BSC biomedical token-classification models
        |-- MedSpaNER clinical token-classification models
        |-- PSA regular-expression extraction
        |-- Gleason regular-expression extraction
        |
        v
Structured annotations (.csv, .json, .ann)
```

---

## Repository Contents

### `evol_generator_openai.py/evol_generator_ollama.py`

Generates Spanish hospital-style clinical progress notes from source clinical cases using the OpenAI Responses API.

Main functionality:

- recursively reads `.txt` clinical cases from an input directory;
- prompts the selected OpenAI model to generate a hospital-style clinical progress note;
- extracts generated content from `---INICIO---` and `---FIN---` delimiters;
- applies fallback extraction when delimiters are unavailable;
- writes generated notes to an output directory;
- stores failed generations as structured JSON files;
- exports a CSV execution log.

---


### `llm_as_judge_Claude.py`

Evaluates generated clinical progress notes against original clinical cases using the Anthropic Messages API.

Main functionality:

- aligns original clinical cases with generated notes;
- compares generated notes against their corresponding source cases;
- evaluates multiple clinical quality dimensions;
- stores compact JSONL evaluation outputs;
- exports CSV summaries for visual overview analysis;
- stores parsing and debugging artifacts;
- supports resumable execution.

---

### `entityextraction.py`

Runs a clinical entity extraction pipeline over original or generated clinical notes.

Main functionality:

- applies BSC biomedical and clinical token-classification models;
- applies MedSpaNER token-classification models;
- detects Gleason score mentions using regular expressions;
- detects PSA mentions using regular expressions;
- merges neural and rule-based outputs;
- exports per-patient JSON files;
- exports CSV mention tables;
- exports Brat-compatible `.ann` files.

---

## Input Data

All scripts are designed to process plain-text `.txt` files.

- `evol_generator_openai.py`: source clinical cases in `.txt` format.
- `evol_generator_ollama.py`: source clinical cases in `.txt` format.
- `llm_as_judge_Claude.py`: original cases and generated progress notes in `.txt` format.
- `entityextraction.py`: original or generated notes to annotate.

Nested directory structures are supported where applicable.

---

## Output Data

Depending on the executed script, the repository may generate:

- `.txt` files containing synthetic clinical progress notes;
- `.csv` files containing execution logs, evaluation summaries, and extracted mention tables;
- `.json` files containing error files, debug files, and per-patient extraction outputs;
- `.jsonl` files containing compact LLM-as-a-judge evaluation records;
- `.ann` files containing Brat-compatible standoff annotations.

---

## Requirements

### Python Version

Recommended and Supported version:

```text
Python 3.10
```



The codebase was designed primarily for Python 3.10-based research environments.

---

### Core Dependencies

Core Python dependencies include:

```text
openai
anthropic
requests
pandas
numpy
torch
transformers
tqdm
```

Install dependencies with:

```bash
pip install -r requirements.txt
```
---

NER pipeline requires access to the following models:
```python
 "BSC-NLP4BIA/bsc-bio-ehr-es-livingner-species",
 "BSC-NLP4BIA/bsc-bio-ehr-es-livingner-humano",
 "BSC-NLP4BIA/bsc-bio-ehr-es-medprocner",
 "BSC-NLP4BIA/bsc-bio-ehr-es-symptemist",
 "BSC-NLP4BIA/bsc-bio-ehr-es-distemist",
 "PlanTL-GOB-ES/bsc-bio-ehr-es-pharmaconer",
 "medspaner/roberta-es-clinical-trials-cases-medic-attr",
 "medspaner/xlm-roberta-large-spanish-trials-cases-temp-ent",
 "medspaner/roberta-es-clinical-trials-cases-neg-spec"
```
---
Ollama is required to be installed, it can be done with:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Version in order to use Qwen3.5 should be >0.20.0. Please verify your version executing the following command
```bash
ollama --version
```
Code requires Ollama models to be installed, and for clouds models, since april 2026 it is required to have a paid suscription for the usage of GLM-5


## Hardware Requirements

Hardware requirements depend on the selected workflow.

### API-Based Workflows

When using hosted APIs such as OpenAI or Anthropic, the scripts can be executed in a standard CPU environment.

Recommended minimum:

```text
CPU: 4 cores
RAM: 8 GB
GPU: Not required
```

### Local Model Inference and Entity Extraction

When using Ollama-served models or transformer-based NER models locally, a CUDA-compatible GPU is recommended.

Recommended configuration:

```text
CPU: 8 cores or higher
RAM: 24 GB or higher for Qwen3.5:35b
GPU: CUDA-compatible GPU recommended
VRAM: Model-dependent
```

The exact requirements depend on the selected LLM, quantisation level, batch size, and NER model configuration.
For this research, the hardware configuration for local inference in Qwen3.5:35b-a3b
```text
GPU: 2x NVIDIA RTX 3090
VRAM: 48GB
Generated output in an average of 5seconds per patient and 73 tokens/s
```
Please be advised we have observed different outputs depending on the hardware for Qwen3.5:35b and running the model in CPU and RAM might not be able to reproduce our actual results. 

---

## Environment Variables

API credentials must be provided through environment variables. 

```bash
export OPENAI_API_KEY="your_openai_api_key"
export ANTHROPIC_API_KEY="your_anthropic_api_key"
```

---

## Usage

### Generate Synthetic Notes with OpenAI

```bash
python evol_generator_openai.py \
  --input-dir /path/to/cases \
  --out-dir /path/to/generated_notes \
  --model gpt-5.4-nano
```

Optional arguments:

```text
--overwrite
```

Replaces existing generated outputs.

```text
--log-csv /path/to/log.csv
```

Specifies a custom CSV log path.

---

### Generate Synthetic Notes with Ollama

```bash
python evol_generator_ollama.py \
  --input-dir /path/to/cases \
  --out-dir /path/to/generated_notes \
  --model qwen3.5:35b-a3b \
  --base-url http://127.0.0.1:11434
```

This mode is intended for local or self-hosted model inference using an Ollama-compatible endpoint.

---

### Evaluate Generated Notes with Anthropic

```bash
python llm_as_judge_Claude.py \
  --original-cases-dir /path/to/original_cases \
  --generated-notes-dir /path/to/generated_notes \
  --output-dir /path/to/evaluation
```

Optional argument:

```text
--rename-generated-files
```

Applies a helper renaming step before evaluation when generated file names require alignment with the original case identifiers.

---

### Run Clinical Entity Extraction

```bash
python entityextraction.py \
  --input-dir /path/to/notes \
  --out-dir /path/to/entity_outputs
```

This module exports structured entity extraction results in CSV, JSON, and Brat-compatible `.ann` formats.

---

## Example End-to-End Workflow

```bash
python evol_generator_openai.py \
  --input-dir data/cases \
  --out-dir outputs/generated_openai \
  --model gpt-5.4-nano

python llm_as_judge_Claude.py \
  --original-cases-dir data/cases \
  --generated-notes-dir outputs/generated_openai \
  --output-dir outputs/evaluation_openai

python entityextraction.py \
  --input-dir outputs/generated_openai \
  --out-dir outputs/entities_openai
```

This workflow performs:

1. synthetic clinical progress-note generation;
2. LLM-based clinical quality evaluation;
3. clinical entity extraction from generated notes.

---

## Recommended Repository Structure

```text
repository/
  data/
    cases/
  outputs/
    GS1/
        Model/
            entities/
    GS2/
        Model/
            entities/
    evaluation/
        GS1/
        GS2/
  logs/
  errors/
  scripts/
     evol_generator_openai.py
     evol_generator_ollama.py
     llm_as_judge_Claude.py
     entityextraction.py
  requirements.txt
  README.md
```

---

## Reproducibility

Experimental results may vary depending on:

- model version;
- API behaviour;
- local inference backend;
- prompt revisions;
- decoding parameters;
- hardware configuration;
- dependency versions;
- input data versioning.

For reproducible experimentation, it is recommended to record:

- input dataset version;
- model identifiers;
- provider or backend;
- prompt template version;
- decoding configuration;
- Python version;
- dependency versions;
- execution logs;
- output directories;
- generated notes;
- evaluation outputs.

Recommended reproducibility files:

```text
requirements.txt
environment.yml
metadata.json
run_config.json
```

---

## Data Governance and Privacy

This repository is intended for research workflows involving clinical text.

Any use of real or sensitive clinical data must comply with applicable legal, ethical, and institutional requirements. Users are responsible for ensuring that:

- all clinical data are properly anonymised or de-identified and came from a web open source;
- data processing complies with applicable data protection regulations;
- ethics committee or institutional review board requirements are satisfied where applicable;
- access to data and generated outputs is appropriately restricted;
- API-based processing is compatible with the data governance framework of the project.

*Clinical data should not be sent to external APIs unless this is explicitly permitted by the corresponding data governance protocol.*

---

## Clinical Safety Statement

The generated notes and extracted annotations are intended exclusively for research and evaluation purposes.

They must not be used for:

- clinical diagnosis;
- treatment recommendation;
- direct patient management;
- replacement of professional clinical documentation;
- automated clinical decision-making.

All generated clinical content should be reviewed by qualified professionals before any possible downstream clinical interpretation.

---

## Known Limitations

The repository has the following limitations:

- generated outputs may contain omissions, hallucinations, or clinical inconsistencies;
- LLM-as-a-judge evaluation may be sensitive to prompt design and model behaviour;
- entity extraction performance depends on the coverage and limitations of the selected NER models;
- regex-based extraction of PSA and Gleason mentions may not capture all possible linguistic variants;
- the repository is designed for research workflows and has not been validated as clinical software.

---

## Versioning

Current software version:

```text
v0.1.0
```

Version meaning:

- `0.1.0`: initial public research version.
- `0.x.x`: experimental research-stage development.
- `1.0.0`: stable release after formal validation and documentation review.

---

## Suggested Citation

If you use this repository in academic work, please cite the associated manuscript or software release.

```bibtex
@software{synthetic_clinical_progress_notes_2026,
  title        = {Generation and Evaluation of Realistic Synthetic Clinical Progress Notes for Prostate Cancer Using Large Language Models},
  author       = {Rey-Blanes, Álvaro and Moreno-Barea, Francisco J. and Veredas, Francisco J.},
  year         = {2026},
  version      = {0.1.0},
  note         = {Research software for Spanish clinical NLP and synthetic clinical note generation}
}
```

---

## License

This repository is released under the MIT License.

Before public release, verify that the selected license is compatible with all institutional, data governance, and third-party dependency requirements.

---

## Contact

For questions regarding the repository, reproducibility, or research use, please contact the corresponding author or repository maintainer.

```text
Maintainer: Álvaro Rey-Blanes
Institution: Inteligencia Computacional en Biomedicina (Universidad de Málaga)
Email: alvaroreyb@uma.es
```