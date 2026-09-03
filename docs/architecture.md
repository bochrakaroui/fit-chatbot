# Architecture

Form separates safety policy, reviewed knowledge, model behavior, and presentation so each can be tested and updated independently.

```text
Web client
    |
    v
POST /api/chat
    |
    +--> deterministic urgent-risk triage --> fixed escalation response
    |
    +--> reviewed knowledge retrieval
    |          |
    |          v
    +--> grounded generator (zero-setup or local model)
               |
               v
       answer + risk level + sources
```

The web application has a server-side fallback so it remains usable when the Python inference service is unavailable. Set `FITNESS_API_URL` to connect it to the full service. The Python service uses the dependency-free grounded generator by default; set `FIT_MODEL_PATH` to a local base model, merged model, or adapter-compatible model directory to enable Transformers inference.

## Safety boundary

Emergency-like language is intercepted before retrieval or generation. Clinically sensitive language receives a scope warning and reviewed sources. This is defense in depth, not a medical-device claim. The assistant provides general education and must not diagnose, prescribe, or replace a clinician.

## Data flow

```text
raw_batches/*.json
       +
dataset/curated/overrides.json
       -
dataset/curated/exclusions.json
       |
       v
fitness_qa.json + build report
       |
       +--> review queue
       +--> grouped train/validation/test JSONL
```

The canonical output is deterministic. Prompt-family hashing keeps the same normalized family in a single split. Records flagged by medical-risk or questionable-advice heuristics are quarantined in `review.jsonl` rather than used for training.
