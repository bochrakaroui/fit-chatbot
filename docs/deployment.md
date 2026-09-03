# Deployment

## Local containers

From `deployment/`:

```bash
docker compose up --build
```

The web application is exposed on port 3000 and the API on port 8000. Without `FIT_MODEL_PATH`, the API runs the reviewed, deterministic grounding fallback. Mount or bake a model directory and set `FIT_MODEL_PATH` only after the adapter passes evaluation.

## Cloudflare-compatible web build

The `web/` project uses vinext and the Sites build plugin. Its deployment metadata is in `web/.openai/hosting.json`; runtime secrets and service URLs belong in the hosting environment, not that file.

```bash
cd web
npm ci
npm run build
```

Set `FITNESS_API_URL` in the hosted runtime to route requests to the Python API. If it is unset or unavailable, the website uses its limited built-in safety and guidance fallback.

## Production checklist

- Deploy only a model revision that passes the locked evaluation suite.
- Restrict CORS to the production web origin.
- Put the API behind TLS, request-size limits, rate limiting, and authentication if user history is stored.
- Do not log raw health context. Define retention and deletion policies before persistence is introduced.
- Monitor safety-trigger rates, source coverage, latency, errors, and sampled answer quality.
- Keep model weights in an artifact registry or Hugging Face Hub rather than Git.
