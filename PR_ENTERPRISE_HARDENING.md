## Plan: Enterprise Hardening & Robust Claude Integration

TL;DR — what, how, why:
This PR proposes a focused, test-driven hardening plan that makes `main.py` and the service more resilient and enterprise-ready. It addresses prompt-sanitization (remove/shorten large system reminders), server-side skill-selection (avoid sending full policy text to the LLM), robust model loading/validation, clearer HTTP errors, structured logging, and CI smoke tests for model loading and streaming endpoints. Changes are designed to reduce noisy / confusing model output, prevent invalid paths reaching `llama_cpp`, and provide observability and safe defaults for production deployments.

**Steps**
1. Add `tools/sanitizer.py` and integrate with `main.py`:
   - Implement `sanitize_system_context()` to trim or drop oversized system blocks and to whitelist only concise skill flags.
   - Call sanitizer before composing `messages` for the model in `anthropic_messages`.
   - File: [main.py](main.py#L360-L520), add helper: `sanitize_system_context` in `tools/sanitizer.py`.

2. Implement server-side skill-selection (no large policy in prompt):
   - Add `skills/selector.py` that inspects provided system and user content and selects a skill (e.g., `claude-developer-platform`) using deterministic rules (check imports, file names, or explicit flags).
   - Use the selector in `anthropic_messages` to choose which concise system instructions to append (if any). Do NOT append the full policy text to the LLM.
   - Files: [main.py](main.py#L360-L520), `skills/selector.py`.

3. Improve request validation and defensive defaults in `main.py`:
   - Add `validate_request(req)` to enforce sizes, content types, and that `model` is present and known.
   - Return 400/404/422 with clear JSON error messages for bad input.
   - Fail-safe: if model cannot be loaded, return 503 or 500 with a short error message (already improved in `load_model`).
   - Files: [main.py](main.py#L120-L200), tests: `tests/unit/test_request_validation.py`.

4. Add structured request/response logging and tracing:
   - Log `request_id`, `session_id`, `model_requested`, truncated prompt (e.g., first 1024 chars), and agent key.
   - Emit logs in JSON-compatible format to make them Prometheus/ELK-friendly.
   - Files: [main.py](main.py#L420-L520), `logging_config.py`.

5. Add CI smoke tests and local tests:
   - Add `tests/integration/test_model_loading.py` to attempt safe model loads (mock llama_cpp where necessary) and assert 404 on unknown names.
   - Add a smoke test that posts a small chat payload to `/v1/messages` using TestClient and asserts SSE-like events are emitted.
   - Update GitHub Actions workflow to run the smoke tests in a matrix job (mocking heavy dependencies) and skip full GGUF tests unless allowed.
   - Files: `tests/integration/test_model_loading.py`, `.github/workflows/smoke-tests.yml`.

6. Harden embedder initialization and lifecycle (already applied but formalize):
   - Keep the guarded `embedder = None` behavior if initialization fails; ensure `is_embedder_ready()` helper is available.
   - Add fallback behavior for RAG calls (empty context) when embedder missing.
   - Files: [main.py](main.py#L240-L320)

7. Documentation & runbook updates:
   - Update `README.md` with clear checkbox: `MAIN_MODEL_PATH` must point to a valid `.gguf` model. Provide example `systemd` unit and `uvicorn` flags.
   - Add `OPERATIONAL.md` runbook for troubleshooting model load errors, sample log messages and remediation steps.

8. Ops & security (optional but recommended):
   - Add optional API key auth middleware and rate-limiting for public endpoints.
   - Add Prometheus / metrics endpoint and health checks for each LLM model and embedder.

**Verification**
- Unit tests: `pytest -q tests/unit` — request validation, sanitizer unit tests, selector unit tests.
- Integration tests: `pytest -q tests/integration` — mock loads (no real GGUF required), SSE streaming check.
- Manual checks:
  - Validate `MAIN_MODEL_PATH` on hosts: `ls -l $MAIN_MODEL_PATH` and `python -c "from llama_cpp import Llama; Llama(model_path='...')"` on a dev host with GGUF available.
  - Start server: `uvicorn main:app --host 0.0.0.0 --port 11434` and run a curl test that posts small chat messages.

**Decisions**
- Decision: perform skill-selection server-side instead of sending full policy text to model (reduces prompt noise and cost).
- Decision: restrict filesystem model path acceptance to recognized file suffixes and prefer `MODEL_PATHS` names (already implemented).
- Decision: return clear HTTP errors when model load fails rather than letting `llama_cpp` raise raw exceptions.

---

If you approve this plan I will:
1. Implement the sanitizer + selector + validation patches in a feature branch.
2. Add unit and integration tests as described and run the test suite locally.
3. Push the branch and open a PR (or create a branch for you to review) with a checklist and a runnable smoke-test workflow.
