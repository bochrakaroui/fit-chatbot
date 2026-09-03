# Form Fitness Coach

Form is a safety-aware fitness coaching application built from this repository's curated Q&A dataset. The project includes deterministic data preparation, validation and review gates, grouped model splits, retrieval-backed chat, optional local model inference, QLoRA training, evaluation, a FastAPI service, and a responsive web interface.

## What is included

- A deterministic raw-plus-curated dataset build
- Strict schema, duplicate, boilerplate, and safety-risk validation
- Prompt-family-aware train, validation, test, and expert-review splits
- Reviewed-source retrieval and deterministic urgent-risk triage
- A FastAPI chat service and offline command-line chat
- Optional local Transformers inference
- QLoRA training, model download, and adapter merge utilities
- A responsive vinext/React chat interface with a resilient server fallback
- Unit, integration, evaluation, lint, CI, and container workflows

## Quick start

Requires Python 3.11+ and Node 22+.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e '.[dev]'

python dataset/build_dataset.py
python dataset/validate_dataset.py --strict
python dataset/split_dataset.py
python evals/run_eval.py
pytest

uvicorn fit_chatbot.api:app --reload
```

In a second terminal:

```powershell
cd web
npm ci
$env:FITNESS_API_URL='http://localhost:8000'
npm run dev
```

Open `http://localhost:5173`. The web application also has a limited grounded fallback and therefore works without the Python API.

## Dataset lifecycle

Never edit `fitness_qa.json` directly. The canonical file is generated from immutable batches under `raw_batches/`, manual additions and corrections in `dataset/curated/overrides.json`, and retired raw prompts in `dataset/curated/exclusions.json`.

```bash
python dataset/build_dataset.py
python dataset/validate_dataset.py --strict
python dataset/split_dataset.py
```

The safe default excludes records requiring expert review from model splits. See [data governance](docs/data-governance.md) for the review process.

## Fine-tuning

Install the optional ML stack and download the configured base model:

```bash
pip install -e '.[ml]'
python training/download_model.py
python training/train_sft.py
```

The default configuration is `training/config.json`. Run `python training/train_sft.py --smoke-test` before a full job. GPU capability, model licensing, and the model chat template should be confirmed for the selected model. Model weights and generated artifacts are intentionally excluded from Git.

After training, point `FIT_MODEL_PATH` at the saved output and rerun `python evals/run_eval.py`. Only promote the adapter if it beats the recorded baseline while passing every urgent-safety case. The initial suite is a software gate, not a substitute for expert fitness or clinical review.

## Useful commands

```bash
fit-chat
uvicorn fit_chatbot.api:app --host 0.0.0.0 --port 8000
pytest --cov=fit_chatbot
ruff check src dataset training evals tests
cd web && npm run lint && npm run build
docker compose -f deployment/compose.yaml up --build
```

## Documentation

- [Architecture](docs/architecture.md)
- [Data governance and review](docs/data-governance.md)
- [Deployment](docs/deployment.md)

## Important limitation

There are currently 242 records quarantined for qualified subject-matter review. The application and training pipeline are operational, but a public health-facing release should not use those records until that review is complete. Form provides general fitness education, not medical diagnosis or treatment.
