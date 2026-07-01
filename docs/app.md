# CEMSPIMS experiment graph app

Local UI for browsing the experiment DAG, branching with config edits, launching multi-pipeline jobs, and cross-run comparison.

## Requirements

```bash
pip install -e ".[app]"
cd app/web && npm install && npm run build
```

For training steps launched from the app, also install `pip install -e ".[train]"`.

## Launch

From repo root:

```bash
python scripts/run.py app
```

Opens `http://127.0.0.1:8765` (API + built static frontend). Hard-refresh the browser after rebuilding the frontend (`npm run build`).

## Architecture (event-driven)

The graph is driven by an in-memory **state store** + **SSE** stream, not polling or hot-path disk scans.

| Piece | Role |
|-------|------|
| `GET /api/stream` | Server-Sent Events: initial graph snapshot, then deltas (`active_step`, `node_update`, `step_log`, `job_update`, …) |
| `data/.app_state/state.json` | Debounced persistence of run/step/job state |
| `POST /api/state/rescan` | Rebuild state from manifests + `step_journal.json` on disk |
| `scripts/sync_app_state.py` | CLI rescan; `--reload` POSTs rescan to a running app |

After pipeline runs **outside** the app (plain `python scripts/run.py run …`), sync so the UI matches disk:

```bash
python scripts/run.py app          # terminal 1
python scripts/sync_app_state.py --reload   # terminal 2 (app must be up)
```

Or use **Rescan from disk** on the Graph page.

## Primary workflow (Graph page)

1. **Click a per-run step** — drawer shows status, lazy config fields / artifacts, live log for the active step.
2. **Branch & run…** — child run id, config patch from this step onward, pipelines to run (sets `pipeline_scope`), starts job.
3. **Extend run…** on a child — add pipelines to scope and optionally run them. **Overwrite** is opt-in (checkbox).
4. Graph updates over SSE when steps start/finish; running nodes pulse blue.

**Branch copy rule:** branching at step *S* copies only **upstream inputs** to *S* (not downstream `model_train/` or `processed/`). See `src/mss/run/branch.py`.

Shared prepare nodes (ingest / returns / targets) are **read-only** in the UI.

## Visual encoding

- **Left stripe** = pipeline stage (graph, train, eval, …)
- **Status dot** = complete / missing / **running**
- Inherited parent steps on a child lane are labeled `(parent_run_id)`

## Activity bar

Shows live headline parsed from job log lines when you have selected/focused a run (epoch bars, evaluate tqdm, etc.).

## Jobs page

- **Start run** — pick run, select pipelines (auto-extends `pipeline_scope`), optional overwrite.
- **Live log** — WebSocket `WS /api/jobs/{id}/stream` for expanded job output.
- Job list also streams via SSE (`job_update` events).

## Config page (advanced)

Full TOML editor. Prefer **Branch & run** for experiments.

## Compare page

Lists runs with completed `analysis.summarize`. Output under `data/<anchor>/processed/comparisons/<comparison_id>/`.

## API highlights

| Endpoint | Purpose |
|----------|---------|
| `GET /api/stream` | SSE: graph snapshot + live deltas |
| `GET /api/graph` | One-shot graph projection (same as snapshot payload) |
| `GET /api/nodes/{id}/config-fields` | Step-scoped config schema |
| `GET /api/nodes/{id}/artifacts` | Lazy artifact previews |
| `POST /api/runs/branch-and-run` | Branch + patch + scope + job |
| `POST /api/runs/{id}/extend-scope` | Widen `pipeline_scope` + optional job |
| `POST /api/jobs` | Start background pipeline job |
| `POST /api/state/rescan` | Rebuild in-memory state from disk |
| `WS /api/jobs/{id}/stream` | Streaming job log (Jobs page) |

Removed (do not use): `/api/graph/revision`, `/api/runs/{id}/progress` — replaced by SSE.

## Manifest `pipeline_scope`

Child runs store which pipelines appear on the graph:

```json
"pipeline_scope": ["model.train", "model.evaluate"]
```

Downstream pipelines in scope appear on the child lane even before first run. Terminal node appears when `analysis.summarize` is in scope and that step has completed.
